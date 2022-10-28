import librosa, subprocess, shlex, os
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

import ChatClasses
import Dir
import errors


def get_time_str(seconds):
    hours = seconds // 3600
    seconds -= hours * 3600

    minutes = seconds // 60
    seconds -= minutes * 60

    return f'{str(hours).zfill(2)}:{str(minutes).zfill(2)}:{str(seconds).zfill(2)}'


def audio_process_child(vod_id, test_audio, time_stamps_mp, max_time_stamp_count, time_stamp_p_value):
    audio, sr = get_audio(vod_id)
    if audio is not None:
        return_audio_time_markers(test_audio, max_time_stamp_count, time_stamp_p_value)
    else:
        time_stamps_mp[-1] = -1
        raise errors.AudioDownloadError(f'Audio failed to download for vod_id <{vod_id}>')


def get_audio_process():
    pass


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


def download_child_process(clip_timings_mp, stream_data, chat_analysis_finished, editing_finished, max_clip_count, seconds_per_bucket):
    date_str = date.today().strftime("%m-%d-%y")
    vod_path = f'{Dir.clips_dir}/{date_str}'
    timing_view = np.ndarray((max_clip_count, 2), dtype=np.uint16, buffer=clip_timings_mp._obj)

    for i, vod_id in enumerate(stream_data.keys()):
        while chat_analysis_finished.value == 0:
            pass
        print(f'Download {i + 1} of {len(stream_data)} started')
        timing_copy = timing_view.copy()
        chat_analysis_finished.value = 1

        path = f'{vod_path}/{stream_data[vod_id]["streamer"]}/{stream_data[vod_id]["title"]}'

        for clip_i, time in enumerate(timing_copy):
            if np.all(time == 0):
                print('break')
                break

            if ((time[1] - time[0]) / seconds_per_bucket) > 10:
                continue
            download_clip(get_time_str(time[0]), get_time_str(time[1]), path, vod_id, clip_i)
        print(f'Download {i + 1} of {i / len(stream_data)} finished')

    print('Download child exited')


def download_clip(start_time, end_time, path, vod_id, vod_i):
    download_cmd = f'twitch-dl download -q source -s {start_time} -e {end_time} {vod_id} -o {path}/clip_{vod_i}.mkv --overwrite'
    subprocess.run(shlex.split(download_cmd), stdout=subprocess.PIPE, shell=False)


def get_clips(vod_id, streamer, title, time_start_stop):
    path = f'{Dir.clips_dir}/{date.today().strftime("%m-%d-%y")}/'
    for path_extension in (date.today().strftime("%m-%d-%y"), streamer, title):
        path = f'{path}/{path_extension}'
        if not os.path.exists(path):
            os.mkdir(path)

    for i, time in enumerate(time_start_stop):
        start, end = get_time_str(time[0]), get_time_str(time[1])
        download_cmd = f'twitch-dl download -q source -s {get_time_str(time[0])} -e {get_time_str(time[1])} {vod_id} -o {path}/clip_{i}.mkv --overwrite'
        subprocess.run(shlex.split(download_cmd), stdout=subprocess.PIPE, shell=False)
