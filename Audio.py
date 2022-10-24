import librosa, subprocess, shlex, os
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

import Dir
import errors

def audio_process_child(vod_id, test_audio, time_stamps_mp, max_time_stamp_count, time_stamp_p_value):
    audio, sr = get_audio(vod_id)
    if audio is not None:
        return_audio_time_markers(test_audio, max_time_stamp_count, time_stamp_p_value)
    else:
        time_stamps_mp[-1] = -1
        raise errors.AudioDownloadError(f'Audio failed to download for vod_id <{vod_id}>')




def return_audio_time_markers(test_audio, max_time_stamp_count, time_stamp_p_value):
    compare_audio, test_audio = [], None
    path = f'{Dir.game_params}/valorant/audio'
    comparison_files = os.listdir(path)
    for file in os.listdir(path):
        if file == 'test.wav':
            audio_y, audio_sr = librosa.load(f'{path}/{file}')
        else:
            compare_audio.append(librosa.load(f'{path}/{file}'))

    corr_max = np.mean(audio_y) * ()

    for file_name, audio in zip(comparison_files, compare_audio):


        audio_correlation = signal.correlate(audio_y, audio[0], mode='valid', method='fft')
        x_indicies = np.arange(0, audio_correlation.shape[0]) / audio_sr
        cor_max = np.max()
        max_val = np.max(audio_correlation)
        print(audio_correlation)
        plt.plot(x_indicies, audio_correlation)
        plt.show()
        print('b')


def get_audio(vod_id) -> np.ndarray or None:
    temp_audio_str = 'temp.mkv'
    download_cmd = shlex.split(f'twitch-dl download -q audio_only {vod_id} -o {temp_audio_str} --overwrite')
    subprocess.run(download_cmd, stdout=subprocess.PIPE, shell=False)
    audio, sr = None, None
    if os.path.exists(temp_audio_str):
        audio, sr, = librosa.load(temp_audio_str)
    return audio, sr
