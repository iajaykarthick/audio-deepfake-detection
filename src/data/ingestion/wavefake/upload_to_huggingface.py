import os
import pandas as pd
from tqdm import tqdm
from getpass import getpass
from datasets import Dataset, DatasetDict, Audio
from huggingface_hub import HfApi, HfFolder
from utils.config import load_config

# Load configuration
config = load_config()
audio_dir = config.get('data_paths', {}).get('wavefake', {}).get('audio_files', '/app/data/wavefake/audio_files')
repo_id = config.get('huggingface_repo_id', {}).get('wavefake', 'ajaykarthick/wavefake-audio')


dir_list = [
    'ljspeech_melgan', 
    'ljspeech_full_band_melgan', 
    'ljspeech_hifiGAN', 
    'ljspeech_melgan_large', 
    'ljspeech_multi_band_melgan',
    'ljspeech_parallel_wavegan', 
    'ljspeech_waveglow', 
    'LJSpeech-1.1/wavs'
]

fake_label_mapping = {
    'ljspeech_melgan' : 'WF1',
    'ljspeech_full_band_melgan' : 'WF2',
    'ljspeech_hifiGAN' : 'WF3',
    'ljspeech_melgan_large' : 'WF4',
    'ljspeech_multi_band_melgan' : 'WF5',
    'ljspeech_parallel_wavegan' : 'WF6',
    'ljspeech_waveglow' : 'WF7',
    'LJSpeech-1.1/wavs' : 'R'
}


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
    for dir in dir_list:
        print('processing dir:',dir)
        real_or_fake = fake_label_mapping[dir]
        print('real or fake label:', real_or_fake)
        if dir == "LJSpeech-1.1/wavs":
            cur_audio_dir = audio_dir+ '/LJSpeech-1.1/wavs'
        else:
            cur_audio_dir = audio_dir+ '/generated_audio/'+ dir
        print('audio dir:', cur_audio_dir)
        for root, _, files in tqdm(os.walk(cur_audio_dir), total=len(os.listdir(cur_audio_dir)), desc="Processing audio files"):
            for file in files:                
                file_path = os.path.join(root, file)
                cleaned_filename = os.path.basename(file_path)
                audio_id = cleaned_filename.replace('_generated','').replace('_gen','').replace('.wav', '')
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
