import time
import requests
from collections import defaultdict
from datasets import load_dataset


USERNAME      = "ajaykarthick"
DATASET_NAME  = "wavefake-audio"
REPO_ID       = f"{USERNAME}/{DATASET_NAME}"
JSON_FILE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/audio_id_to_file_map.json"

def get_parquet_file_path(partition_id):
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/data/partition{partition_id}-00000-of-00001.parquet"

def fetch_audio_id_list():
    """
    Fetch the list of audio IDs from the dataset.
    """
    response = requests.get(JSON_FILE_URL)
    response.raise_for_status()
    audio_id_to_file_map = response.json()
    
    return list(audio_id_to_file_map.keys())

def get_audio_dataset(audio_ids, cache_dir=None, max_retries=3, backoff_factor=0.5):
    """
    Fetch the dataset for given audio ID or list of audio IDs.
    """
    response = requests.get(JSON_FILE_URL)
    response.raise_for_status()
    audio_id_to_file_map = response.json()

    if isinstance(audio_ids, str):
        audio_ids = [audio_ids]

    # Create a dictionary to map parquet files to audio IDs
    parquet_to_audio_ids = defaultdict(list)
    for audio_id in audio_ids:
        parquet_file = audio_id_to_file_map[audio_id]['train']
        parquet_to_audio_ids[audio_id].append(parquet_file)

    # Create a generator to yield filtered examples from each parquet file
    def dataset_generator():
        for audio_id, parquet_files in parquet_to_audio_ids.items():
            for parquet_file in parquet_files:
                retry_attempts = 0
                while retry_attempts < max_retries:
                    try:
                        if cache_dir:
                            dataset = load_dataset("parquet", data_files={'train': parquet_file}, split="train", cache_dir=cache_dir)
                        else:
                            dataset = load_dataset("parquet", data_files={'train': parquet_file}, split="train", streaming=True)

                        filtered_ds = dataset.filter(lambda example: example['audio_id'] == audio_id)
                        for example in filtered_ds:
                            yield example
                        break  # Exit the retry loop if no exceptions
                    except Exception as e:
                        retry_attempts += 1
                        if retry_attempts >= max_retries:
                            print(f"Failed to process parquet file {parquet_file} after {max_retries} attempts. Error: {e}")
                            raise
                        else:
                            wait_time = backoff_factor * (2 ** (retry_attempts - 1))
                            print(f"Retrying {parquet_file} in {wait_time} seconds. Attempt {retry_attempts} of {max_retries}.")
                            time.sleep(wait_time)

    return dataset_generator()


def get_dataset_from_single_parquet(partition_id, cache_dir=None, max_retries=3, backoff_factor=0.5):
    """
    Fetch the dataset from a single parquet file and return a generator that iterates through all audios.
    """
    parquet_file = get_parquet_file_path(partition_id)
    def dataset_generator():
        retry_attempts = 0
        while retry_attempts < max_retries:
            try:
                if cache_dir:
                    dataset = load_dataset("parquet", data_files={'train': parquet_file}, split="train", cache_dir=cache_dir)
                else:
                    dataset = load_dataset("parquet", data_files={'train': parquet_file}, split="train", streaming=True)

                for example in dataset:
                    yield example
                break  # Exit the retry loop if no exceptions
            except Exception as e:
                retry_attempts += 1
                if retry_attempts >= max_retries:
                    print(f"Failed to process parquet file {parquet_file} after {max_retries} attempts. Error: {e}")
                    raise
                else:
                    wait_time = backoff_factor * (2 ** (retry_attempts - 1))
                    print(f"Retrying {parquet_file} in {wait_time} seconds. Attempt {retry_attempts} of {max_retries}.")
                    time.sleep(wait_time)

    return dataset_generator()
