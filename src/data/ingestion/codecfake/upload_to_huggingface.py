import os
import pandas as pd
from tqdm import tqdm
from getpass import getpass
from datasets import Dataset, DatasetDict, Audio
from huggingface_hub import HfApi, HfFolder
from utils.config import load_config

# Load configuration
config = load_config()
audio_dir = config.get('data_paths', {}).get('codecfake', {}).get('audio_files', '/app/data/codecfake/audio_files')
repo_id = config.get('huggingface_repo_id', {}).get('codecfake', 'ajaykarthick/codecfake-audio')

# Retrieve the Hugging Face token from the environment variable or Hugging Face CLI session
token = os.getenv('HF_TOKEN')
if token is None:
    token = HfFolder.get_token()
    if token is None:
        token = getpass("Please enter your Hugging Face token: ")

# Initialize Hugging Face API with the token
hf_api = HfApi(token)

def create_df(audio_dir):
    data = []
    for root, _, files in tqdm(os.walk(audio_dir), total=len(os.listdir(audio_dir)), desc="Processing audio files"):
        for file in files:
            if file.endswith('.flac'):
                file_path = os.path.join(root, file)
                audio_id = os.path.basename(os.path.dirname(file_path))

                if audio_id.startswith('SSB'):
                    print(f'Skipping {audio_id}')
                    continue
                
                if file.startswith('F0'):
                    real_or_fake = file[:3]
                else:
                    real_or_fake = 'R'
                data.append({"audio": file_path, "audio_id": audio_id, "real_or_fake": real_or_fake})
                
    df = pd.DataFrame(data)
    return df

df = create_df(audio_dir)

# Calculate the number of unique audio IDs
unique_audio_ids = df['audio_id'].nunique()

desired_chunk_size_min = 90
desired_chunk_size_max = 110

def find_best_chunk_size(total_ids, min_size, max_size):
    for size in range(max_size, min_size - 1, -1):
        if total_ids % size == 0:
            return size
    raise ValueError("No suitable chunk size found within the given range.")

try:
    chunk_size = find_best_chunk_size(unique_audio_ids, desired_chunk_size_min, desired_chunk_size_max)
    num_shards = unique_audio_ids // chunk_size
    print(f"Chunk Size: {chunk_size}, Number of Chunks: {num_shards}")
except ValueError as e:
    print(e)
    exit(1)

chunk_index = 0
datasets = []
current_chunk = []
audio_ids_so_far = set()

# Group the DataFrame by audio_id
grouped = df.copy().groupby('audio_id')

def read_audio_file(path):
    with open(path, 'rb') as f:
        return {'bytes': f.read()}

def save_chunk(chunk_data, chunk_index):
    chunk_name = f"partition{chunk_index}"
    df_chunk = pd.DataFrame(chunk_data)
    ds_chunk = Dataset.from_pandas(df_chunk)
    ds_chunk = ds_chunk.cast_column("audio", Audio(decode=True))
    datasets.append((chunk_name, ds_chunk))

# Iterate over the grouped data and create chunks
for audio_id, group in tqdm(grouped, desc="Creating chunks"):
    if audio_id in audio_ids_so_far:
        continue

    if len(audio_ids_so_far) + 1 > chunk_size:
        save_chunk(current_chunk, chunk_index)
        current_chunk = []
        audio_ids_so_far = set()
        chunk_index += 1

    current_chunk.extend(group.to_dict('records'))
    audio_ids_so_far.add(audio_id)

# Save the last chunk if it has remaining data
if current_chunk:
    save_chunk(current_chunk, chunk_index)

# Create a DatasetDict from the chunks
dataset_dict = DatasetDict({shard_name: ds for shard_name, ds in datasets})

# Push to Hugging Face Hub
dataset_dict.push_to_hub(repo_id)

print("Dataset pushed to Hugging Face Hub successfully.")
