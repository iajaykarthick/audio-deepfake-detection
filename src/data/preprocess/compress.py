import os
import subprocess

def compress_audio_files(file_paths, output_directory, codec='flac'):
    """
    Compress a list of audio files using the specified lossless codec.

    Parameters:
    - file_paths (list): A list of paths to the audio files.
    - output_directory (str): The directory where compressed files will be stored.
    - codec (str): The codec to use for compression ('flac' or 'alac').

    This function converts audio files to the specified lossless format and stores them
    in the provided output directory. It supports both FLAC and ALAC formats.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    for file_path in file_paths:
        filename, ext = os.path.splitext(os.path.basename(file_path))
        compressed_file_path = os.path.join(output_directory, f"{filename}.{codec}")

        if os.path.exists(compressed_file_path):
            print(f"File {compressed_file_path} already exists. Skipping compression.")
            continue

        if codec == 'flac':
            ffmpeg_command = ['ffmpeg', '-i', file_path, '-c:a', 'flac', compressed_file_path]
        elif codec == 'alac':
            ffmpeg_command = ['ffmpeg', '-i', file_path, '-c:a', 'alac', compressed_file_path]
        else:
            raise ValueError(f"Unsupported codec: {codec}")

        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Compressed {file_path} to {compressed_file_path} using {codec.upper()}.")
