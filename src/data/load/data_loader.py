from . import codecfake, wavefake

def get_wavefake_audio_id_list():
    """
    Fetch the list of audio IDs in the wavefake dataset.
    """
    return wavefake.fetch_audio_id_list()

def get_codecfake_audio_id_list():
    """
    Fetch the list of audio IDs in the codecfake dataset.
    """
    return codecfake.fetch_audio_id_list()

def load_audio_data(audio_ids, dataset='codecfake', cache_dir=None):
    """
    Load audio data for the given audio IDs.
    """
    if dataset == 'codecfake':
        return codecfake.get_audio_dataset(audio_ids, cache_dir=cache_dir)
    else:
        raise ValueError(f"Invalid dataset: {dataset}") 
    
def load_parquet_data(partition_id, dataset='codecfake', cache_dir=None):
    """
    Load audio data from a single parquet file.
    """
    if dataset == 'codecfake':
        return codecfake.get_dataset_from_single_parquet(partition_id, cache_dir=cache_dir)
    if dataset == 'wavefake':
        return wavefake.get_dataset_from_single_parquet(partition_id, cache_dir=cache_dir)
