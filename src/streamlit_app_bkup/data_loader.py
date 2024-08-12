import requests
from datasets import load_dataset

def fetch_audio_id_to_file_map(username, dataset_name):
    repo_id = f"{username}/{dataset_name}"
    json_file_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/audio_id_to_file_map.json"

    response = requests.get(json_file_url)
    response.raise_for_status()
    return response.json()

def get_dataset(audio_id, audio_id_to_file_map):
    parquet_file = audio_id_to_file_map[audio_id]
    iterable_ds = load_dataset("parquet", data_files=parquet_file, split="train", streaming=True)
    dataset = iterable_ds.filter(lambda example: example['audio_id'] == audio_id)
    return dataset

def create_display_data(audio_id, audio_id_to_file_map):
    data = []
    for example in get_dataset(audio_id, audio_id_to_file_map):
        audio_info = {
            "audio": example['audio']['array'],
            "sampling_rate": example['audio']['sampling_rate'],
            "audio_id": example['audio_id'],
            "real_or_fake": example['real_or_fake']
        }
        data.append(audio_info)

    # Sort data by real_or_fake field
    data.sort(key=lambda x: (x['real_or_fake'] != 'R', x['real_or_fake']))
    
    return data
