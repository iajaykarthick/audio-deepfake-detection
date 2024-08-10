import os
import subprocess
import zipfile

### Extracting files from the ZIP file ###
def unzip_file(zip_path, output_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)
        print(f"Files extracted from {zip_path} to {output_path}")

### Extracting specific files from the 7z archive ###
def extract_specific_files_with_7z(archive_path, output_path, files_to_extract, verbose=False):
    try:
        command = ['7z', 'x', archive_path, f'-o{output_path}', '-aos'] + files_to_extract
        if verbose:
            subprocess.run(command, check=True)
        else:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if verbose:
            print(f"Extracted specified files to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to extract files: {str(e)}")

### List all files from the 7z archive ###
def list_files_in_archive(archive_path):
    try:
        command = ['7z', 'l', archive_path]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8').splitlines()
        files = [line.split()[-1] for line in output if line.endswith('.wav')]
        return files
    except subprocess.CalledProcessError as e:
        print(f"Failed to list files: {str(e)}")
        return []

### Sort the files in the order of real files with p, real files with SSB, and then fake files ###
def sort_files(files):
    real_files_st_w_p = sorted([f for f in files if f.replace('train/', '').startswith('p')])
    real_files_st_w_SSB = sorted([f for f in files if f.replace('train/', '').startswith('SSB')])
    fake_files = sorted([f for f in files if f.replace('train/', '').startswith('F0')])
    return real_files_st_w_p + real_files_st_w_SSB + fake_files

### Extract and convert the files to FLAC format ###
def extract_and_convert_files(real_file, fake_files, archive_path, output_base):
    file_id = real_file.replace('train/', '').replace('.wav', '')
    output_directory = os.path.join(output_base, file_id)
    os.makedirs(output_directory, exist_ok=True)
    files_to_extract = [real_file] + fake_files

    for file in files_to_extract:
        flac_filename = os.path.basename(file).replace('.wav', '.flac')
        flac_file_path = os.path.join(output_directory, flac_filename)
        if os.path.exists(flac_file_path):
            continue 
        temp_wav_file = os.path.join(output_base, file)
        extract_specific_files_with_7z(archive_path, output_base, [file])
        ffmpeg_command = ['ffmpeg', '-i', temp_wav_file, '-c:a', 'flac', flac_file_path]
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(temp_wav_file)