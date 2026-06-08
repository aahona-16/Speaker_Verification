import gradio as gr
import os
import shutil
import torch
import math
import database
import librosa
import numpy as np

# ---- WINDOWS SYMLINK PATCH FOR SPEECHBRAIN ----
if os.name == 'nt':
    import pathlib
    _orig_symlink = os.symlink
    _orig_path_symlink_to = pathlib.Path.symlink_to
    
    def _patched_symlink(src, dst, target_is_directory=False):
        try:
            _orig_symlink(src, dst, target_is_directory=target_is_directory)
        except OSError:
            src_path = os.path.abspath(src)
            dst_path = os.path.abspath(dst)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                
    def _patched_path_symlink_to(self, target, target_is_directory=False):
        try:
            _orig_path_symlink_to(self, target, target_is_directory=target_is_directory)
        except OSError:
            src_path = os.path.abspath(str(target))
            dst_path = os.path.abspath(str(self))
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                
    os.symlink = _patched_symlink
    pathlib.Path.symlink_to = _patched_path_symlink_to
# ------------------------------------------------

# ---- HF_HUB_DOWNLOAD PATCH FOR SPEECHBRAIN ----
import huggingface_hub
import requests.exceptions
_orig_hf_hub_download = huggingface_hub.hf_hub_download
def _patched_hf_hub_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        if 'token' not in kwargs: kwargs['token'] = kwargs.pop('use_auth_token')
        else: kwargs.pop('use_auth_token')
    try:
        return _orig_hf_hub_download(*args, **kwargs)
    except Exception as e:
        if type(e).__name__ == "RemoteEntryNotFoundError" or "404 Client Error" in str(e):
            raise requests.exceptions.HTTPError("404 Client Error: Not Found")
        raise
huggingface_hub.hf_hub_download = _patched_hf_hub_download
# ------------------------------------------------

from speechbrain.inference.speaker import SpeakerRecognition

print("Initializing Vector Database...", flush=True)
database.init_db()

print("Loading SpeechBrain Model (ResNet)...", flush=True)
model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-resnet-voxceleb", 
    savedir="model_cache_resnet"
)
print("Model loaded successfully!", flush=True)

# THRESHOLDS
THRESHOLD_RAW = 0.25 # Same as model.verify_files threshold
SIGMOID_K = 12.0
SIGMOID_X0 = 0.15

def get_embedding_tensor(audio_path):
    """Preprocesses audio and returns a 3-second padded tensor for batching along with its relative length."""
    sig, _ = librosa.load(audio_path, sr=16000)
    sig = torch.tensor(sig)
    
    frame_size = int(16000 * 0.1)
    start_idx = 0
    for i in range(0, sig.shape[0], frame_size):
        frame = sig[i:i+frame_size]
        if torch.sqrt(torch.mean(frame**2)).item() > 0.002: 
            start_idx = i
            break
            
    if start_idx > 0:
        sig = sig[start_idx:]
        
    max_samples = 16000 * 3
    if sig.shape[0] >= max_samples:
        sig = sig[:max_samples]
        rel_len = 1.0
    else:
        rel_len = sig.shape[0] / max_samples
        sig = torch.nn.functional.pad(sig, (0, max_samples - sig.shape[0]))
        
    return sig, rel_len

def get_embedding(audio_path, truncate=True):
    """Helper to extract a 192-d embedding vector using SpeechBrain."""
    if truncate:
        sig, rel_len = get_embedding_tensor(audio_path)
    else:
        sig, _ = librosa.load(audio_path, sr=16000)
        sig = torch.tensor(sig)
        rel_len = 1.0
        
    batch = sig.unsqueeze(0)
    rel_length = torch.tensor([rel_len])
    
    # Disable gradient calculation to speed up CPU inference
    with torch.no_grad():
        emb = model.encode_batch(batch, wav_lens=rel_length)
        
    return emb

def save_audio_permanently(username, temp_path):
    """Copies an audio file from a temporary location to a permanent drive storage directory."""
    if not temp_path or not os.path.exists(temp_path):
        return ""
        
    storage_dir = os.path.join(os.path.dirname(__file__), "drive_storage", username.replace(" ", "_"))
    os.makedirs(storage_dir, exist_ok=True)
    
    filename = os.path.basename(temp_path)
    permanent_path = os.path.join(storage_dir, filename)
    
    try:
        shutil.copy2(temp_path, permanent_path)
        return permanent_path
    except Exception as e:
        print(f"Error saving file permanently: {e}")
        return ""

def enroll_user(name, audio_path):
    if not name or name.strip() == "":
        return "⚠️ Please enter a Name."
    if not audio_path:
        return "⚠️ Please upload an Audio clip."
        
    try:
        emb = get_embedding(audio_path)
        # Convert tensor to a standard Python list
        emb_list = emb.squeeze().tolist()
        
        database.enroll_user(name.strip(), emb_list)
        permanent_path = save_audio_permanently(name.strip(), audio_path)
        database.log_file_to_user(name.strip(), os.path.basename(audio_path), permanent_path)
        return f"✅ Successfully enrolled '{name}' into the Vector Database!"
    except Exception as e:
        return f"❌ Error during enrollment: {str(e)}"

def delete_user_ui(name):
    if not name:
        return "⚠️ Please select a user to delete."
    try:
        database.delete_user(name)
        return f"✅ Successfully deleted '{name}' from the database."
    except Exception as e:
        return f"❌ Error deleting user: {str(e)}"

def verify_identity(claimed_name, audio_path):
    if not claimed_name:
        return "⚠️ Please select a User.", ""
    if not audio_path:
        return "⚠️ Please upload an Audio clip to verify.", ""
        
    try:
        stored_emb_list = database.get_user_embedding(claimed_name)
        if not stored_emb_list:
            return f"❌ Error: User '{claimed_name}' not found in database.", ""
            
        stored_emb = torch.tensor(stored_emb_list).unsqueeze(0).unsqueeze(0)
        incoming_emb = get_embedding(audio_path)
        
        score = torch.nn.functional.cosine_similarity(incoming_emb, stored_emb, dim=-1)
        raw_score = score.item()
        
        confidence = 1.0 / (1.0 + math.exp(-SIGMOID_K * (raw_score - SIGMOID_X0)))
        pct = confidence * 100
        
        is_match = raw_score > THRESHOLD_RAW
        
        if is_match:
            result_text = f"✅ YES - Identity Verified ({claimed_name})"
        else:
            result_text = f"❌ NO - Identity Mismatch"
            
        details = f"Confidence: {pct:.1f}%\n(Raw Cosine Score: {raw_score:.3f})"
        return result_text, details
        
    except Exception as e:
        return f"❌ Error processing verification: {str(e)}", ""

def direct_compare(audio_path_1, audio_path_2):
    """Directly compares two audio files without storing them."""
    if not audio_path_1 or not audio_path_2:
        return "⚠️ Please upload both Audio 1 and Audio 2.", ""
        
    try:
        # For direct 1-to-1 comparison, speed is not a bottleneck.
        # We can analyze the ENTIRE audio files (truncate=False) to get maximum mathematical precision!
        emb1 = get_embedding(audio_path_1, truncate=False)
        emb2 = get_embedding(audio_path_2, truncate=False)
        
        score = torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1)
        raw_score = score.item()
        
        confidence = 1.0 / (1.0 + math.exp(-SIGMOID_K * (raw_score - SIGMOID_X0)))
        pct = confidence * 100
        
        is_match = raw_score > THRESHOLD_RAW
        
        if is_match:
            result_text = f"✅ YES - It is the SAME person."
        else:
            result_text = f"❌ NO - DIFFERENT speakers."
            
        details = f"Confidence: {pct:.1f}%\n(Raw Cosine Score: {raw_score:.3f})"
        return result_text, details
        
    except Exception as e:
        return f"❌ Error processing direct comparison: {str(e)}", ""

def get_initial_cluster_text():
    """Generates the initial HTML text for Tab 5 showing all existing enrolled users."""
    users = database.get_all_users()
    if not users:
        return "<i>No users enrolled yet. Go to Tab 1 to enroll users.</i>"
    
    lines = ["<h3>📁 Existing Enrolled Users in Database:</h3><ul>"]
    for user in users:
        if not user.startswith("Unknown"):
            count = database.get_file_count(user)
            historical_files = database.get_historical_files(user)
            lines.append(f"<li><b>🗣️ {user}</b> (Total files historically: {count})")
            if historical_files:
                lines.append("<ul style='color: gray; font-size: 0.9em; list-style-type: none; padding-left: 20px;'>")
                for hf, hp in historical_files:
                    if hp and os.path.exists(hp):
                        try:
                            import base64
                            with open(hp, "rb") as f:
                                b64_audio = base64.b64encode(f.read()).decode("utf-8")
                            lines.append(f"<li style='margin-bottom: 5px; display: flex; align-items: center; gap: 10px;'>- 📄 {hf} <audio src='data:audio/wav;base64,{b64_audio}' controls style='height: 30px;'></audio></li>")
                        except:
                            lines.append(f"<li>- 📄 {hf} <i style='color:red;'>(Audio Load Error)</i></li>")
                    else:
                        lines.append(f"<li>- 📄 {hf}</li>")
                lines.append("</ul>")
            lines.append("</li>")
    lines.append("</ul><p><i>Note: The database only tracks filenames of files uploaded from this point forward.</i></p>")
    return "".join(lines)

def batch_cluster(files, state, append=False):
    """Processes multiple audio files and clusters them by enrolled users, optionally appending to existing state."""
    if not files:
        return "⚠️ Please upload at least one audio file.", state
        
    try:
        user_names = database.get_all_users()
        if not user_names and not append:
            return "⚠️ No users are enrolled! Please enroll users in Tab 1 first.", state
            
        stored_embs = {}
        for name in user_names:
            emb_list = database.get_user_embedding(name)
            if emb_list:
                stored_embs[name] = torch.tensor(emb_list).unsqueeze(0).unsqueeze(0)
                
        if not append or state is None:
            clusters = {name: [] for name in user_names}
            unknown_profiles = {}
            unknown_counter = 1
            for name in user_names:
                if name.startswith("Unknown "):
                    try:
                        num = int(name.split(" ")[1])
                        if num >= unknown_counter:
                            unknown_counter = num + 1
                    except:
                        pass
        else:
            clusters = state.get("clusters", {name: [] for name in user_names})
            unknown_profiles = state.get("unknown_profiles", {})
            unknown_counter = state.get("unknown_counter", 1)
            # Ensure new DB users are in clusters
            for name in user_names:
                if name not in clusters:
                    clusters[name] = []
                    
        # Pre-calculate existing files to prevent duplicate processing
        existing_files = set()
        if append:
            for file_list in clusters.values():
                existing_files.update(file_list)
                    
        # 1. Preprocess all NEW files into tensors
        batch_tensors = []
        batch_rel_lens = []
        valid_files = []
        file_path_map = {}
        newly_added = []
        for file_obj in files:
            file_path = file_obj if isinstance(file_obj, str) else getattr(file_obj, 'name', str(file_obj))
            filename = os.path.basename(file_path)
            
            if filename in existing_files:
                continue # Skip files we already processed in this session
                
            try:
                sig, rel_len = get_embedding_tensor(file_path)
                batch_tensors.append(sig)
                batch_rel_lens.append(rel_len)
                valid_files.append(filename)
                file_path_map[filename] = file_path
            except Exception as e:
                if "Unknown Errors" not in clusters:
                    clusters["Unknown Errors"] = []
                clusters["Unknown Errors"].append(f"{filename} (Error loading: {str(e)})")
                
        if not batch_tensors:
            return "❌ Could not load any valid audio files.", state
            
        # 2. Extract embeddings in Mini-Batches (10 files at a time)
        batch_size = 10
        all_embs = []
        for i in range(0, len(batch_tensors), batch_size):
            batch = torch.stack(batch_tensors[i:i+batch_size])
            rel_length = torch.tensor(batch_rel_lens[i:i+batch_size])
            with torch.no_grad():
                embs = model.encode_batch(batch, wav_lens=rel_length)
                all_embs.append(embs)
                
        all_embs = torch.cat(all_embs, dim=0) # Shape: [Num_Files, 1, 192]
        
        # 3. Process the Extracted Embeddings
        for idx in range(len(valid_files)):
            filename = valid_files[idx]
            emb = all_embs[idx].unsqueeze(0)
            
            try:
                best_score = -1.0
                best_match = None
                
                # Check Enrolled Users
                for name, stored_emb in stored_embs.items():
                    score = torch.nn.functional.cosine_similarity(emb, stored_emb, dim=-1).item()
                    if score > best_score:
                        best_score = score
                        best_match = name
                        
                # Dynamic Clustering for Unknowns (Single-Linkage Clustering for maximum accuracy)
                if best_score < 0.31:
                    best_unknown_score = -1.0
                    best_unknown_match = None
                    for unk_name, unk_embs in unknown_profiles.items():
                        # Compare against ALL previous embeddings in this cluster, not just an average
                        for unk_emb in unk_embs:
                            unk_score = torch.nn.functional.cosine_similarity(emb, unk_emb, dim=-1).item()
                            if unk_score > best_unknown_score:
                                best_unknown_score = unk_score
                                best_unknown_match = unk_name
                                
                    if best_unknown_score > 0.25: 
                        best_match = best_unknown_match
                        unknown_profiles[best_match].append(emb)
                    else:
                        new_unk_name = f"Unknown {unknown_counter}"
                        unknown_profiles[new_unk_name] = [emb]
                        best_match = new_unk_name
                        unknown_counter += 1
                        clusters[best_match] = []
                
                clusters[best_match].append(filename)
                newly_added.append(filename)
                
            except Exception as e:
                if "Unknown Errors" not in clusters:
                    clusters["Unknown Errors"] = []
                clusters["Unknown Errors"].append(f"{filename} (Error scoring: {str(e)})")
                newly_added.append(filename)
                
        # 4. Update Database Counts for Enrolled Users Only
        for speaker, file_list in clusters.items():
            if file_list and not speaker.startswith("Unknown"):
                # Only increment by the newly added files in THIS specific run
                new_in_cluster = [f for f in file_list if f in newly_added]
                if new_in_cluster:
                    database.increment_file_count(speaker, len(new_in_cluster))
                    for filename in new_in_cluster:
                        temp_path = file_path_map.get(filename)
                        permanent_path = save_audio_permanently(speaker, temp_path) if temp_path else ""
                        database.log_file_to_user(speaker, filename, permanent_path)

        # 5. Format output as HTML for animation support
        output_lines = []
        for speaker, file_list in clusters.items():
            if not speaker.startswith("Unknown"):
                # Always show enrolled DB users
                total_db_count = database.get_file_count(speaker)
                output_lines.append(f"<h3 style='margin-bottom:4px;'>🗣️ {speaker} (Total files historically: {total_db_count})</h3>")
                output_lines.append("<ul style='margin-top:0px;'>")
                if file_list:
                    for f in file_list:
                        if f in newly_added:
                            anim_style = "color: #10b981; font-weight: bold; animation: pulse 1.5s infinite;"
                            output_lines.append(f"<li><span style='{anim_style}'>✨ {f} (Just Added!) ✨</span></li>")
                        else:
                            output_lines.append(f"<li>{f}</li>")
                else:
                    output_lines.append("<li><i>No new files in this session</i></li>")
                output_lines.append("</ul><br>")
                
            elif file_list:
                # Only show Unknowns if they have files in the current session
                if speaker != "Unknown Errors":
                    output_lines.append(f"<h3 style='margin-bottom:4px;'>❓ {speaker} (Files in Current Session: {len(file_list)})</h3>")
                else:
                    output_lines.append(f"<h3 style='margin-bottom:4px;'>⚠️ {speaker}</h3>")
                    
                output_lines.append("<ul style='margin-top:0px;'>")
                for f in file_list:
                    if f in newly_added:
                        anim_style = "color: #10b981; font-weight: bold; animation: pulse 1.5s infinite;"
                        output_lines.append(f"<li><span style='{anim_style}'>✨ {f} (Just Added!) ✨</span></li>")
                    else:
                        output_lines.append(f"<li>{f}</li>")
                output_lines.append("</ul><br>")
                
        # Add CSS pulse animation manually since Gradio doesn't include it by default
        css = "<style>@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }</style>"
        
        state = {
            "clusters": clusters,
            "unknown_profiles": unknown_profiles,
            "unknown_counter": unknown_counter
        }
        return css + "".join(output_lines), state
        
    except Exception as e:
        return f"<b style='color:red;'>❌ Error during batch clustering: {str(e)}</b>", state

def clear_batch_state():
    return get_initial_cluster_text(), {"clusters": {}, "unknown_profiles": {}, "unknown_counter": 1}, None

def batch_cluster_fresh(files, state):
    return batch_cluster(files, state, append=False)

def batch_cluster_append(files, state):
    return batch_cluster(files, state, append=True)

def diarize_conversation(audio_path):
    """Sliding-window diarization using the enrolled users in the Vector DB."""
    if not audio_path:
        return "⚠️ Please upload a conversation audio file."
        
    try:
        user_names = database.get_all_users()
        if not user_names:
            return "⚠️ No users are enrolled! Please enroll users in Tab 1 first so the system knows who to look for."
            
        # Load all enrolled vectors into memory for fast comparison
        stored_embs = {}
        for name in user_names:
            emb_list = database.get_user_embedding(name)
            if emb_list:
                stored_embs[name] = torch.tensor(emb_list).unsqueeze(0).unsqueeze(0)
                
        sig, _ = librosa.load(audio_path, sr=16000)
        sig = torch.tensor(sig)
        total_samples = sig.shape[0]
        
        # 1. SMART VAD: Scan audio in 0.1s frames to find "Utterances" (continuous speech blocks)
        frame_size = int(16000 * 0.1) 
        is_speech = []
        for i in range(0, total_samples, frame_size):
            frame = sig[i:i+frame_size]
            rms = torch.sqrt(torch.mean(frame**2)).item()
            is_speech.append(rms > 0.002) # Fast Energy Threshold
            
        utterances = [] # List of (start_sample, end_sample)
        in_speech = False
        start_frame = 0
        silence_counter = 0
        MAX_SILENCE_FRAMES = 5 # 0.5s of silence marks the end of a block
        MAX_UTTERANCE_FRAMES = 100 # Force split after 10.0 seconds to track fast speaker changes
        
        for idx, active in enumerate(is_speech):
            if active:
                if not in_speech:
                    in_speech = True
                    start_frame = idx
                silence_counter = 0
                
                if (idx - start_frame) >= MAX_UTTERANCE_FRAMES:
                    utterances.append((start_frame * frame_size, idx * frame_size))
                    start_frame = idx
            else:
                if in_speech:
                    silence_counter += 1
                    if silence_counter >= MAX_SILENCE_FRAMES:
                        # End of utterance
                        utterances.append((start_frame * frame_size, (idx - silence_counter) * frame_size))
                        in_speech = False
                        
        if in_speech:
            utterances.append((start_frame * frame_size, len(is_speech) * frame_size))
            
        if not utterances:
            return "No recognizable speech found."
            
        # 2. MASSIVE SPEEDUP: Extract only the first 3 seconds of each utterance for the AI!
        batch_tensors = []
        batch_rel_lens = []
        MAX_EMB_SAMPLES = int(16000 * 3.0)
        
        for start_s, end_s in utterances:
            utt_sig = sig[start_s:end_s]
            # Limit to 3 seconds for lightning-fast processing
            if utt_sig.shape[0] >= MAX_EMB_SAMPLES:
                utt_sig = utt_sig[:MAX_EMB_SAMPLES]
                batch_rel_lens.append(1.0)
            else:
                batch_rel_lens.append(utt_sig.shape[0] / MAX_EMB_SAMPLES)
                utt_sig = torch.nn.functional.pad(utt_sig, (0, MAX_EMB_SAMPLES - utt_sig.shape[0]))
            batch_tensors.append(utt_sig)
            
        # 3. Process in Mini-Batches (10 utterances at a time)
        batch_size = 10
        all_embs = []
        for i in range(0, len(batch_tensors), batch_size):
            batch = torch.stack(batch_tensors[i:i+batch_size])
            rel_length = torch.tensor(batch_rel_lens[i:i+batch_size])
            with torch.no_grad():
                embs = model.encode_batch(batch, wav_lens=rel_length)
                all_embs.append(embs)
                
        all_embs = torch.cat(all_embs, dim=0) # Shape: [Num_Utterances, 1, 192]
        
        # 4. Timeline merging & Unknown Clustering
        timeline = []
        current_speaker = None
        current_start_time = 0.0
        current_end_time = 0.0
        
        unknown_profiles = {}
        unknown_counter = 1
        
        for idx in range(len(utterances)):
            emb = all_embs[idx].unsqueeze(0)
            
            best_score = -1.0
            best_match = None
            
            # Compare against ENROLLED users
            for name, stored_emb in stored_embs.items():
                score = torch.nn.functional.cosine_similarity(emb, stored_emb, dim=-1).item()
                if score > best_score:
                    best_score = score
                    best_match = name
                    
            # Compare against dynamically tracked UNKNOWN users
            if best_score < 0.31:
                best_unknown_score = -1.0
                best_unknown_match = None
                
                for unk_name, unk_embs in unknown_profiles.items():
                    for unk_emb in unk_embs:
                        unk_score = torch.nn.functional.cosine_similarity(emb, unk_emb, dim=-1).item()
                        if unk_score > best_unknown_score:
                            best_unknown_score = unk_score
                            best_unknown_match = unk_name
                        
                # Use Single-Linkage clustering with 0.25 threshold for utterances
                if best_unknown_score > 0.25:
                    best_match = best_unknown_match
                    unknown_profiles[best_match].append(emb)
                else:
                    new_unk_name = f"Unknown {unknown_counter}"
                    unknown_profiles[new_unk_name] = [emb]
                    best_match = new_unk_name
                    unknown_counter += 1
            
            utt_start_sec = utterances[idx][0] / 16000.0
            utt_end_sec = utterances[idx][1] / 16000.0
            
            def fmt(sec):
                m = int(sec // 60)
                s = int(sec % 60)
                return f"{m:02d}:{s:02d}"
            
            # Merge consecutive utterances of the same speaker
            if current_speaker is None:
                current_speaker = best_match
                current_start_time = utt_start_sec
                current_end_time = utt_end_sec
            elif best_match != current_speaker or (utt_start_sec - current_end_time) > 2.0:
                timeline.append(f"[{fmt(current_start_time)} - {fmt(current_end_time)}] : 🗣️ {current_speaker}")
                current_speaker = best_match
                current_start_time = utt_start_sec
                current_end_time = utt_end_sec
            else:
                current_end_time = utt_end_sec
                
            # Handle the very last chunk properly
            if idx == len(utterances) - 1:
                timeline.append(f"[{fmt(current_start_time)} - {fmt(current_end_time)}] : 🗣️ {current_speaker}")
                
        return "\n".join(timeline)

    except Exception as e:
        return f"❌ Error during diarization: {str(e)}"


def smart_search(audio_path):
    """Identifies the uploaded audio and returns all historical files for that user."""
    if not audio_path:
        return "<i>⚠️ Please upload an audio file to search.</i>"
    
    try:
        user_names = database.get_all_users()
        if not user_names:
            return "<i>⚠️ No enrolled users in the database to match against.</i>"
            
        stored_embs = {}
        for name in user_names:
            emb_list = database.get_user_embedding(name)
            if emb_list:
                stored_embs[name] = torch.tensor(emb_list).unsqueeze(0).unsqueeze(0)
                
        incoming_emb = get_embedding(audio_path)
        
        best_score = -1.0
        best_match = None
        for name, stored_emb in stored_embs.items():
            score = torch.nn.functional.cosine_similarity(incoming_emb, stored_emb, dim=-1).item()
            if score > best_score:
                best_score = score
                best_match = name
                
        if best_score < THRESHOLD_RAW:
            return f"<div style='padding: 15px; background-color: #fef2f2; border-left: 5px solid #ef4444; border-radius: 4px;'><h2 style='margin:0; color:#b91c1c;'>❌ No Match Found</h2><p style='margin:5px 0 0 0; color:#991b1b;'>The highest match was <b>{best_match}</b> at {(best_score * 100):.1f}%, which is below the required threshold.</p></div>"
            
        # Success match
        confidence = 1.0 / (1.0 + math.exp(-SIGMOID_K * (best_score - SIGMOID_X0)))
        pct = confidence * 100
        
        historical_files = database.get_historical_files(best_match)
        
        html = f"""
        <div style="padding: 15px; background-color: #f0fdf4; border-left: 5px solid #22c55e; border-radius: 4px; margin-bottom: 20px;">
            <h2 style="margin:0; color:#166534;">✅ Match Found: 🗣️ {best_match}</h2>
            <p style="margin:5px 0 0 0; color:#15803d;"><b>Confidence: {pct:.1f}%</b> (Raw similarity: {(best_score*100):.1f}%)</p>
        </div>
        """
        
        html += f"<h3>📁 {best_match}'s File History (Drive):</h3>"
        if not historical_files:
            html += "<p><i>No historical filenames recorded for this user (or files were uploaded before history tracking was enabled).</i></p>"
        else:
            html += "<ul style='list-style-type: none; padding-left: 0;'>"
            for hf, hp in historical_files:
                if hp and os.path.exists(hp):
                    try:
                        import base64
                        with open(hp, "rb") as f:
                            b64_audio = base64.b64encode(f.read()).decode("utf-8")
                        html += f"<li style='margin-bottom: 10px; display: flex; align-items: center; gap: 15px;'><div style='width: 200px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>📄 {hf}</div> <audio src='data:audio/wav;base64,{b64_audio}' controls style='height: 40px;'></audio></li>"
                    except:
                        html += f"<li style='margin-bottom: 10px;'>📄 {hf} <i style='color:red;'>(Audio Load Error)</i></li>"
                else:
                    html += f"<li style='margin-bottom: 10px;'>📄 {hf} <i style='color:gray; font-size: 0.8em;'>(File physically unavailable)</i></li>"
            html += "</ul>"
            
        return html
    except Exception as e:
        return f"<b style='color:red;'>❌ Error during Smart Search: {str(e)}</b>"

def update_dropdown():
    """Helper to dynamically refresh the dropdown list of users."""
    users = database.get_all_users()
    return gr.Dropdown(choices=users)

# Create Gradio UI with Tabs
with gr.Blocks(title="AI Speaker Verification & Diarization") as interface:
    gr.Markdown("# 🎙️ AI Voice Ecosystem (Verify & Diarize)")
    
    with gr.Tab("1. Enroll & Manage Users"):
        gr.Markdown("### 👤 Add a new voice to the memory database")
        with gr.Row():
            enroll_name = gr.Textbox(label="User Name", placeholder="e.g. John Doe")
            enroll_audio = gr.Audio(type="filepath", label="Voice Clip to Enroll")
            
        enroll_btn = gr.Button("Save to Vector DB", variant="primary")
        enroll_output = gr.Textbox(label="Enrollment Status")
        
        gr.Markdown("### 🗑️ Delete an existing user")
        with gr.Row():
            initial_users = database.get_all_users()
            delete_name = gr.Dropdown(label="Select User to Delete", choices=initial_users)
            delete_btn = gr.Button("Delete User", variant="stop")
        delete_output = gr.Textbox(label="Deletion Status")
        
    with gr.Tab("2. Verify Identity") as verify_tab:
        gr.Markdown("### 🔍 Verify an incoming voice against the database")
        with gr.Row():
            verify_name = gr.Dropdown(label="Claimed Identity (Select Enrolled User)", choices=initial_users)
            verify_audio = gr.Audio(type="filepath", label="Voice Clip to Test")
            
        verify_btn = gr.Button("Verify Identity", variant="primary")
        
        with gr.Row():
            verify_result = gr.Textbox(label="Verdict", text_align="center")
            verify_score = gr.Textbox(label="Details")
            
    with gr.Tab("3. Conversation Timeline") as diarize_tab:
        gr.Markdown("### ⏱️ Who Spoke When? (Diarization)")
        gr.Markdown("Upload a conversation. The AI will scan it and identify exactly when enrolled users were speaking.")
        diarize_audio = gr.Audio(type="filepath", label="Conversation Audio")
        diarize_btn = gr.Button("Generate Timeline", variant="primary")
        diarize_output = gr.Textbox(label="Timeline Output", lines=15)
        
    with gr.Tab("4. Direct Comparison"):
        gr.Markdown("### ⚖️ Compare two voice clips directly (No Database)")
        gr.Markdown("Upload two audio files to instantly check if they belong to the same person without saving anything.")
        with gr.Row():
            direct_audio_1 = gr.Audio(type="filepath", label="Audio Clip 1")
            direct_audio_2 = gr.Audio(type="filepath", label="Audio Clip 2")
            
        direct_btn = gr.Button("Compare Voices", variant="primary")
        
        with gr.Row():
            direct_result = gr.Textbox(label="Verdict", text_align="center")
            direct_score = gr.Textbox(label="Details")
            
    with gr.Tab("5. Batch Clustering"):
        gr.Markdown("### 🗂️ Sort multiple audio files by Speaker")
        gr.Markdown("Upload a batch of audio files (e.g. 50 files). The AI will analyze every single file and group them by the Enrolled Users they belong to. It will even cluster strangers into 'Unknown 1', 'Unknown 2', etc.")
        batch_files = gr.File(file_count="multiple", label="Upload Multiple Audio Files")
        with gr.Row():
            batch_btn = gr.Button("Cluster Files (Start Fresh)", variant="primary")
            batch_append_btn = gr.Button("Add to Existing Batch", variant="secondary")
            clear_batch_btn = gr.Button("Clear Batch Memory", variant="stop")
        batch_output = gr.HTML(label="Clustering Results", value=get_initial_cluster_text())
        batch_state = gr.State(value=None)

    with gr.Tab("6. Smart Search & History"):
        gr.Markdown("### 🔎 Reverse Audio Search (Drive)")
        gr.Markdown("Upload an unknown voice clip. The AI will instantly identify who it belongs to, and magically pull up a 'Drive Folder' showing **every single audio file** that person has ever uploaded to the system.")
        search_audio = gr.Audio(type="filepath", label="Mystery Voice Clip")
        search_btn = gr.Button("Identify Person & Open Drive", variant="primary")
        search_output = gr.HTML(label="Search Results")

    # --- EVENT BINDINGS ---
    enroll_btn.click(
        fn=enroll_user,
        inputs=[enroll_name, enroll_audio],
        outputs=[enroll_output]
    ).then(
        fn=update_dropdown,
        outputs=[verify_name]
    ).then(
        fn=update_dropdown,
        outputs=[delete_name]
    )
    
    delete_btn.click(
        fn=delete_user_ui,
        inputs=[delete_name],
        outputs=[delete_output]
    ).then(
        fn=update_dropdown,
        outputs=[verify_name]
    ).then(
        fn=update_dropdown,
        outputs=[delete_name]
    )

    verify_btn.click(
        fn=verify_identity,
        inputs=[verify_name, verify_audio],
        outputs=[verify_result, verify_score]
    )
    
    diarize_btn.click(
        fn=diarize_conversation,
        inputs=[diarize_audio],
        outputs=[diarize_output]
    )
    
    direct_btn.click(
        fn=direct_compare,
        inputs=[direct_audio_1, direct_audio_2],
        outputs=[direct_result, direct_score]
    )
    
    batch_btn.click(
        fn=batch_cluster_fresh,
        inputs=[batch_files, batch_state],
        outputs=[batch_output, batch_state]
    ).then(
        fn=update_dropdown,
        outputs=[verify_name]
    ).then(
        fn=update_dropdown,
        outputs=[delete_name]
    )
    
    batch_append_btn.click(
        fn=batch_cluster_append,
        inputs=[batch_files, batch_state],
        outputs=[batch_output, batch_state]
    ).then(
        fn=update_dropdown,
        outputs=[verify_name]
    ).then(
        fn=update_dropdown,
        outputs=[delete_name]
    )
    
    clear_batch_btn.click(
        fn=clear_batch_state,
        outputs=[batch_output, batch_state, batch_files]
    )
    
    search_btn.click(
        fn=smart_search,
        inputs=[search_audio],
        outputs=[search_output]
    )
    
    # Refresh dropdowns
    verify_tab.select(fn=update_dropdown, outputs=[verify_name]).then(fn=update_dropdown, outputs=[delete_name])

if __name__ == "__main__":
    allowed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "drive_storage"))
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False, allowed_paths=[allowed_dir])
