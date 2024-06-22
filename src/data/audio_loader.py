import librosa

class AudioFile:
    def __init__(self, file_path, label, generation_method=None):
        self.file_path = file_path
        self.label = label
        self.generation_method = generation_method
        self.data, self.sr = self.load_audio(file_path)
        self.duration = len(self.data) / self.sr
    
    def load_audio(self, file_path):
        data, sr = librosa.load(file_path, sr=None)
        return data, sr
    
    def get_info(self):
        return {
            'file_path': self.file_path,
            'label': self.label,
            'generation_method': self.generation_method,
            'sample_rate': self.sr,
            'duration': self.duration
        }
