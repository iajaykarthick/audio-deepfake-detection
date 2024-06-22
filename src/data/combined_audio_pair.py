from .audio_loader import AudioFile
from features.feature_extractor import FeatureExtractor
from features.visualizer import Visualizer


class CombinedAudioPair:
    def __init__(self, real_file_path, fake_file_paths, generation_methods=None):
        self.real_audio = AudioFile(real_file_path, label='real')
        self.fake_audios = [AudioFile(fake_file_paths[i], label='fake', generation_method=(generation_methods[i] if generation_methods else None))
                            for i in range(len(fake_file_paths))]
        
        if not generation_methods:
            self.real_audio.generation_method = "Real Audio"
            for fake_audio in self.fake_audios:
                fake_audio.generation_method = self.infer_generation_method(fake_audio.file_path.split('/')[-1])

    def get_info(self):
        real_info = self.real_audio.get_info()
        fake_info = [fake_audio.get_info() for fake_audio in self.fake_audios]
        return {'real': real_info, 'fake': fake_info}

    def infer_generation_method(self, file_name):
        if file_name.startswith('F01'):
            return 'SoundStream'
        elif file_name.startswith('F02'):
            return 'SpeechTokenizer'
        elif file_name.startswith('F03'):
            return 'FunCodec'
        elif file_name.startswith('F04'):
            return 'EnCodec'
        elif file_name.startswith('F05'):
            return 'AudioDec'
        elif file_name.startswith('F06'):
            return 'AcademicCodec'
        elif file_name.startswith('F07'):
            return 'DAC'
        else:
            return 'Unknown'

    def extract_features(self):
        real_extractor = FeatureExtractor(self.real_audio.data, self.real_audio.sr)
        fake_extractors = [FeatureExtractor(fake_audio.data, fake_audio.sr) for fake_audio in self.fake_audios]

        real_features = real_extractor.extract_all_features()
        fake_features = [extractor.extract_all_features() for extractor in fake_extractors]

        return {'real': real_features, 'fake': fake_features}
    
    def plot_real_vs_fake(self, fake_index=None):
        for i, fake_audio in enumerate(self.fake_audios):
            if fake_index is not None and i != fake_index:
                continue
            supertitle = f'Real Audio vs Fake Audio ({fake_audio.generation_method})'
            title2 = f'Fake Audio (Method: {fake_audio.generation_method})'
            self._plot_comparison(self.real_audio, fake_audio, supertitle, title2=title2)

    def plot_fake_vs_fake(self):
        for i in range(len(self.fake_audios)):
            for j in range(i + 1, len(self.fake_audios)):
                supertitle = f'Fake Audio Comparison ({self.fake_audios[i].generation_method} vs {self.fake_audios[j].generation_method})'
                title1 = f'Fake Audio (Method: {self.fake_audios[i].generation_method})'
                title2 = f'Fake Audio (Method: {self.fake_audios[j].generation_method})'
                self._plot_comparison(self.fake_audios[i], self.fake_audios[j], supertitle, title1=title1, title2=title2)
                
    def _plot_comparison(self, audio1, audio2, supertitle, title1='Real Audio', title2='Fake Audio'):
        # Plot waveforms
        Visualizer.plot_waveform_side_by_side(audio1.data, audio2.data, audio1.sr, audio2.sr, f'{supertitle} - Waveform', title1, title2)
        
        # Plot spectrograms
        Visualizer.plot_spectrogram_side_by_side(audio1.data, audio2.data, audio1.sr, audio2.sr, f'{supertitle} - Spectrogram', title1, title2)

        # Plot MFCCs
        mfcc1 = FeatureExtractor(audio1.data, audio1.sr).extract_mfcc()
        mfcc2 = FeatureExtractor(audio2.data, audio2.sr).extract_mfcc()
        Visualizer.plot_mfcc_side_by_side(mfcc1, mfcc2, audio1.sr, audio2.sr, f'{supertitle} - MFCCs', title1, title2)

    def plot_all(self):
        # Plot waveforms
        Visualizer.plot_waveform(self.real_audio.data, self.real_audio.sr, 'Real Audio Waveform')
        for fake_audio in self.fake_audios:
            Visualizer.plot_waveform(fake_audio.data, fake_audio.sr, f'Fake Audio Waveform (Method: {fake_audio.generation_method})')
        
        # Plot spectrograms
        Visualizer.plot_spectrogram(self.real_audio.data, self.real_audio.sr, 'Real Audio Spectrogram')
        for fake_audio in self.fake_audios:
            Visualizer.plot_spectrogram(fake_audio.data, fake_audio.sr, f'Fake Audio Spectrogram (Method: {fake_audio.generation_method})')

        # Plot MFCCs
        real_mfcc = FeatureExtractor(self.real_audio.data, self.real_audio.sr).extract_mfcc()
        Visualizer.plot_mfcc(real_mfcc, self.real_audio.sr, 'MFCCs of Real Audio')
        for fake_audio in self.fake_audios:
            fake_mfcc = FeatureExtractor(fake_audio.data, fake_audio.sr).extract_mfcc()
            Visualizer.plot_mfcc(fake_mfcc, fake_audio.sr, f'MFCCs of Fake Audio (Method: {fake_audio.generation_method})')
