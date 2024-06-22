import os
import subprocess
import zipfile
import hashlib
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.config import load_config


def calculate_md5(filename, block_size=4096):
    """Calculate the MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b""):
            md5.update(block)
    return md5.hexdigest()


def download_from_zenodo(direct_url, destination, expected_checksum):
    """Download a large file from Zenodo in chunks to avoid loading it all into memory."""
    if os.path.exists(destination):
        if calculate_md5(destination) == expected_checksum:
            print(f"File already exists and is verified: {destination}")
            return True
        else:
            print(f"Checksum mismatch or file corrupted. Re-downloading: {destination}")
            os.remove(destination) 
    try:
        with requests.get(direct_url, stream=True) as response:
            response.raise_for_status() 
            with open(destination, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        print(f"Downloaded file to {destination}")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code} {e.response.reason}")
    except requests.exceptions.ConnectionError:
        print("Connection Error. Please check your internet connection.")
    except requests.exceptions.Timeout:
        print("The request timed out.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        

def download_parts(config):
    repo_urls = [
        "https://zenodo.org/record/11171708",
        "https://zenodo.org/records/11171720",
        "https://zenodo.org/records/11171724"
    ]

    files = [
        [f'train_split.z{str(num).zfill(2)}' for num in range(1, 7)] + ['train_split.zip', 'label.zip'],
        [f'train_split.z{str(num).zfill(2)}' for num in range(7, 15)],
        [f'train_split.z{str(num).zfill(2)}' for num in range(15, 20)]
    ]

    checksums = {
        'train_split.z01': '61423ee0c7e9991b7272b9e50b234439',
        'train_split.z02': '938387b24c700fd3167caff5b6c4c2cc',
        'train_split.z03': '6ed3919559200bfa2e09416816a748ab',
        'train_split.z04': 'cffba4cd8a551e1da36e821e3db1137b',
        'train_split.z05': 'c90ea493d8bfda6cf0fb7713e2bdf628',
        'train_split.z06': 'a8363316c2db890f62d9a3f05ffa882b',
        'train_split.z07': '8c89c7b19c2860dc360e53cf484f7844',
        'train_split.z08': '069fb8d4ff70deafe2b23e70f67c255f',
        'train_split.z09': '208fa914647e7519bf93eb04427e94ab',
        'train_split.z10': '3441024afe061775a29d49292d6b94f6',
        'train_split.z11': 'ef9b40ff9145bbe925944aa5a97a6060',
        'train_split.z12': 'c9a30c2d9c4d0fd59c23058990e79c68',
        'train_split.z13': '2fa3c4f13cad47c1a2c8da6b02593197',
        'train_split.z14': 'd4b19b65945532a1192cfdaea45fe6e5',
        'train_split.z15': 'f1416171017fe86806c1642f36865d22',
        'train_split.z16': '4005490382925a7dde0df498831d4595',
        'train_split.z17': '4aabe67a30484ab45919e58250f1d2c7',
        'train_split.z18': '24fc5547fb782d59a8f94e53eb9fd2bc',
        'train_split.z19': '2ded1a7fda786a04743923790a27f39f',
        'train_split.zip': '600a9ab2c5d820004fecc0e67ac2f645',
        'label.zip'      : '1886fa25a8e018307e709da28bdc57b2'
    }
    
    # Set up the ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for repo_url, file_group in zip(repo_urls, files):
            for file in file_group:
                download_url = f"{repo_url}/files/{file}"
                zip_file_path = f"{config['data_paths']['raw_data_path']}/codecfake/{file}"
                expected_checksum = checksums.get(file, None)
                if expected_checksum:
                    print(f"Downloading {download_url} to {zip_file_path}")
                    futures.append(executor.submit(download_from_zenodo, download_url, zip_file_path, expected_checksum))
    
    # Wait for all futures to complete
    for future in futures:
        future.result() 
    print("All downloads completed.")

        
def unzip_file(zip_path, output_path):
    """Unzip a single zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)
        print(f"Files extracted from {zip_path} to {output_path}")
        
        
def extract_specific_files_with_7z(archive_path, output_path, files_to_extract, verbose=False):
    """Extract specific files from a multi-part zip archive using 7z from a subprocess."""
    try:
        # Prepare the command to include specific files
        command = ['7z', 'x', archive_path, f'-o{output_path}', '-aos'] + files_to_extract
        if verbose:
            subprocess.run(command, check=True)
        else:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if verbose:
            print(f"Extracted specified files to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to extract files: {str(e)}")
        

def list_files_in_archive(archive_path):
    try:
        # Prepare the command to list files in the archive
        command = ['7z', 'l', archive_path]
        
        # Run the command and capture the output
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Decode the output and split by lines
        output = result.stdout.decode('utf-8').splitlines()
        
        # Find the start of the file listing (usually after the headers)
        files = []
        for line in output:
            if line.endswith('.wav'):
                files.append(line.split()[-1])
        
        return files
    except subprocess.CalledProcessError as e:
        print(f"Failed to list files: {str(e)}")
        return []


def sort_files(files):
    real_files = sorted([f for f in files if not f.replace('train/', '').startswith('F0')])
    fake_files = sorted([f for f in files if f.replace('train/', '').startswith('F0')])
    return real_files + fake_files

        
def extract_and_convert_files(real_file, fake_files, archive_path, output_base):
    """
    Extract files from the archive, convert to FLAC, and save in the specified directory structure.
    
    :param file_mapping: Dictionary mapping real audio files to their corresponding fake audio files.
    :param archive_path: Path to the 7z archive.
    :param output_base: Base directory for the output files.
    """
    
    # Extract ID from the real file
    file_id = real_file.replace('train/', '').replace('.wav', '')
    output_directory = os.path.join(output_base, file_id)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Files to extract
    files_to_extract = [real_file] + fake_files
    
    for file in files_to_extract:
        
        # Construct the FLAC filename
        flac_filename = os.path.basename(file).replace('.wav', '.flac')
        flac_file_path = os.path.join(output_directory, flac_filename)
        
        # Skip conversion if the FLAC file already exists
        if os.path.exists(flac_file_path):
            continue 
        
        # Prepare the temporary file path
        temp_wav_file = os.path.join(output_base, file)
        
        # Extract the file
        extract_specific_files_with_7z(archive_path, output_base, [file])
        
        # Convert to FLAC
        ffmpeg_command = ['ffmpeg', '-i', temp_wav_file, '-c:a', 'flac', flac_file_path]
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Remove the temporary .wav file
        os.remove(temp_wav_file)


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
    label_zip_path =  f"{config['data_paths']['raw_data_path']}/codecfake/label.zip"
    label_output_path = f"{config['data_paths']['raw_data_path']}/codecfake/label"
    if not os.path.exists(label_output_path):
        unzip_file(label_zip_path, label_output_path)
        
    # Proceed to extract specific files
    archive_path = f"{config['data_paths']['raw_data_path']}/codecfake/train_split.zip"
    # output_path = f"{config['data_paths']['raw_data_path']}/codecfake/"
    train_flac_base = f"{config['data_paths']['raw_data_path']}/codecfake/train_flac"
    
    files = list_files_in_archive(archive_path)
    files = sort_files(files)
    
    file_mapping = {}
    for file in files:
        # Fake Audio
        if file.replace('train/', '').startswith('F0'):
            # Real Audio
            real_file = f"train/{file.replace('train/', '')[4:]}"
            if real_file not in file_mapping.keys():
                print(f"Real file missing for {file}")
                continue
            else:
                file_mapping[real_file].append(file)
        else:
            if file not in file_mapping:
                file_mapping[file] = []
    
    # extract_and_convert_files(file_mapping, archive_path, train_flac_base)
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(extract_and_convert_files, real_file, fake_files, archive_path, train_flac_base)
                   for real_file, fake_files in file_mapping.items()]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {e}")
    
    # files_to_extract = [
    #     # SSB13650058.wav
    #     'train/SSB13650058.wav',
    #     'train/F01_SSB13650058.wav',
    #     'train/F02_SSB13650058.wav',
    #     'train/F03_SSB13650058.wav',
    #     'train/F04_SSB13650058.wav',
    #     'train/F05_SSB13650058.wav',
    #     'train/F06_SSB13650058.wav',
    #     # SSB13280206.wav
    #     'train/SSB13280206.wav',
    #     'train/F01_SSB13280206.wav',
    #     'train/F02_SSB13280206.wav',
    #     'train/F03_SSB13280206.wav',
    #     'train/F04_SSB13280206.wav',
    #     'train/F05_SSB13280206.wav',
    #     'train/F06_SSB13280206.wav',
    # ]
    # print(f"Extracting specific files from {archive_path} to {output_path}")
    # extract_specific_files_with_7z(archive_path, output_path, files_to_extract)


if __name__ == "__main__":
    main()
