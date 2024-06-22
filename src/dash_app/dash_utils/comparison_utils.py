import os
from utils.config import load_config
from data.audio_loader import AudioFile

def get_audio_folder_names():
    config = load_config()
    data_folder = config['data_paths']['train_raw_audio_path']
    if not os.path.exists(data_folder):
        return []

    folder_names = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
    return folder_names

def display_searchable_dropdown():
    folder_names = get_audio_folder_names()
    if not folder_names:
        return []

    # Return all folder names
    options = [{"label": name, "value": name} for name in folder_names]

    return options

def print_selected_folder_name(selected_folder):
    print(f"Selected folder name: {selected_folder}")

def get_audio_files(audio_id, fake_indices):
    """
    Get AudioFile objects for the real and specified fake audios.
    
    Parameters:
    - audio_id: The ID of the audio.
    - fake_indices: A list of indices of the fake audios (e.g., [1, 2] for F01, F02).
    
    Returns:
    - real_audio: AudioFile object for the real audio.
    - fake_audios: A list of AudioFile objects for the fake audios.
    """
    config = load_config()
    train_raw_audio_path = config['data_paths']['train_raw_audio_path']
    
    real_audio_path = os.path.join(train_raw_audio_path, audio_id, f"{audio_id}.flac")
    
    # Load real audio
    real_audio = AudioFile(real_audio_path, label='real')
    
    fake_audios = []
    for fake_index in fake_indices:
        fake_audio_path = os.path.join(train_raw_audio_path, audio_id, f"F{fake_index:02d}_{audio_id}.flac")
        if os.path.exists(fake_audio_path):
            fake_audios.append(AudioFile(fake_audio_path, label='fake', generation_method=f"F{fake_index:02d}"))
    
    return real_audio, fake_audios