import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.config import load_config
from download import download_parts
from extract import unzip_file, list_files_in_archive, sort_files, extract_and_convert_files

def main():
    parser = argparse.ArgumentParser(description="Process the files based on user input.")
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip downloading files if they are already present and verified')

    args = parser.parse_args()
    
    config = load_config()
    
    if not args.skip_download:
        download_parts(config) 
    else:
        print("Skipping download as requested.")
        
    # Extract labels
    label_zip_path =  f"{config['data_paths']['raw_data_path']['codecfake']['zip_files']}/label.zip"
    label_output_path = f"{config['data_paths']['raw_data_path']['codecfake']['label_files']}"
    if not os.path.exists(label_output_path):
        unzip_file(label_zip_path, label_output_path)
        
    # Proceed to extract specific files
    archive_path = f"{config['data_paths']['raw_data_path']['codecfake']['zip_files']}/train_split.zip"
    train_flac_base = f"{config['data_paths']['raw_data_path']['codecfake']['audio_files']}"
    
    files = list_files_in_archive(archive_path)
    files = sort_files(files)
    
    file_mapping = {}
    for file in files:
        if file.replace('train/', '').startswith('F0'):
            real_file = f"train/{file.replace('train/', '')[4:]}"
            if real_file not in file_mapping:
                print(f"Real file missing for {file}")
                continue
            else:
                file_mapping[real_file].append(file)
        else:
            if file not in file_mapping:
                file_mapping[file] = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(extract_and_convert_files, real_file, fake_files, archive_path, train_flac_base)
                   for real_file, fake_files in file_mapping.items()]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()