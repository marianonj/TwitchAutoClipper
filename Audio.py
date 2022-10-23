import librosa, subprocess, shlex, os
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


def return_audio_time_markers():
    compare_audio, test_audio = [], None
    path = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
    path = f'{path}/game_specific_params/valorant/audio'
    comparison_files = os.listdir(path)
    for file in os.listdir(path):
        if file == 'test.wav':
            test_audio = librosa.load(f'{path}/{file}')
        else:
            compare_audio.append(librosa.load(f'{path}/{file}'))

    for file_name, audio in zip(comparison_files, compare_audio):
        print(file_name)
        audio_correlation = signal.correlate(test_audio[0], audio[0], mode='valid', method='fft')
        x_indicies = np.arange(0, audio_correlation.shape[0]) / test_audio[1]
        plt.plot(x_indicies, audio_correlation)
        plt.show()
        print('b')


def get_audio(comparison_sounds: [np.ndarray, ...], vod_id) -> np.ndarray or None:
    temp_audio_str = 'temp.mkv'
    download_cmd = shlex.split(f'twitch-dl download -q audio_only {vod_id} -o {temp_audio_str} --overwrite')
    subprocess.run(download_cmd, stdout=subprocess.PIPE, shell=False)

    audio_data = None
    if os.path.exists(temp_audio_str):
        audio_data = librosa.load(temp_audio_str)
    return audio_data


return_audio_time_markers()
