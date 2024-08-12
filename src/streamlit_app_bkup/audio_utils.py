import io
from scipy.io.wavfile import write

def convert_audio_to_wav(audio_array, sampling_rate):
    virtualfile = io.BytesIO()
    write(virtualfile, sampling_rate, audio_array)
    virtualfile.seek(0)
    return virtualfile
