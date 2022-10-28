import os, pickle, Dir
import numpy as np
from multiprocessing import Process
import multiprocessing.sharedctypes as mpc
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from enum import Enum
import time
import Config
from ChatClasses import ChatGenerator, StreamData
from Downloader import download_child_process
import unicodedata
from datetime import date


class MpIdxs(Enum):
    text_array_idx = 0
    text_length = 1
    text_analyzed_start_i = 2

class CTypeSharedArray:
    data_type_conversion = {
        'b': np.int8,
        'B': np.uint8,
        'h': np.int16,
        'H': np.uint16,
        'i': np.int16,
        'I': np.uint16,
        'l': np.int32,
        'L': np.uint32,
        'f': np.float32,
        'd': np.float64
    }

    def __init__(self, data_type, length, view_shape=None, mp=None, order='C'):
        self.data_type_mp = data_type
        self.data_type_np = self.data_type_conversion[data_type]
        self.mp = None

        if mp is not None:
            self.mp = mp
        else:
            self.mp = mpc.Array(data_type, length)

        if view_shape is not None:
            self.view = np.ndarray(view_shape, buffer=self.mp._obj, dtype=self.data_type_np, order=order)
        else:
            self.view = np.ndarray(length, buffer=self.mp._obj, dtype=self.data_type_np, order=order)
        self.length = length

    def clear(self, start_i=0, end_i=None):
        if end_i is None:
            end_i = self.length
        length = end_i - start_i
        if self.data_type_mp == 'u':
            self.view[start_i:end_i] = ' ' * length
        else:
            self.view[start_i:end_i] = np.zeros(length, dtype=self.data_type_np)

    def fill(self, value, start_i=0, end_i=None):
        if end_i is None:
            end_i = self.length

        length = end_i - start_i
        if isinstance(value, str):
            self.mp[0:self.length] = value * (length // len(str))
        else:
            self.mp[start_i:end_i] = np.full(length, value, dtype=self.data_type_np)


def return_timestamps(frequency_array, bin_size_seconds, desired_clip_count=8):
    data_np = np.load('data_test.npy')
    max_positive = np.argsort(data_np[:, 3])[::-1]

    count, array_idx = 0, 7
    idxs_sort = None

    while count < desired_clip_count:
        idxs = max_positive[0:array_idx + 1]
        idxs_sort = np.sort(idxs)
        diff = np.abs(np.diff(idxs_sort))
        count = np.count_nonzero(np.argwhere(diff) > 1)
        if array_idx == max_positive.shape[0]:
            break
        array_idx += 1

    idxs_all_sorted = np.sort(np.hstack((idxs_sort - 1, idxs_sort, idxs_sort + 1)))
    diff = np.abs(np.diff(idxs_all_sorted))
    greater_than_one = np.sort(np.argwhere(diff > 1).flatten())
    start_stop_idxs = np.column_stack((np.hstack((0, greater_than_one + 1)), np.hstack((greater_than_one, idxs_all_sorted.shape[0] - 1))))
    start_stop_times = np.column_stack(([idxs_all_sorted[start_stop_idxs[:, 0]], idxs_all_sorted[start_stop_idxs[:, 1]]])) * bin_size_seconds
    return start_stop_times


def set_string_comparisons_dict(directory: str, comparison_dict: dict):
    for file_name in os.listdir(directory):
        if not (file_name[-3:] == 'txt'):
            continue

        file_path = f'{directory}/{file_name}'
        with open(file_path) as f:
            key = file_name[0:-4]
            strings = f.read().strip().lower().replace('\n', ',').strip().split(sep=',')
            if not strings or (len(strings) == 1 and not strings[-1]):
                continue

            if key in comparison_dict:
                comparison_dict[key] = comparison_dict[key] + strings[0:-2] if not strings[-1] else strings
            else:
                comparison_dict[key] = strings[0:-2] if not strings[-1] else strings


def set_clip_timings(timing_view, frequency_array, bucket_seconds, max_clip_count):
    q25_cutoff_idxs = np.argwhere(frequency_array[0] > np.percentile(frequency_array[0], 25)).flatten()
    frequency_q25 = frequency_array[0:, q25_cutoff_idxs]
    sort_idxs = np.vstack(([np.argsort(frequency_q25[row_i])[::-1] for row_i in range(1, frequency_q25.shape[0])]))
    current_idxs = np.column_stack((np.arange(0, sort_idxs.shape[0]), np.zeros(sort_idxs.shape[0], dtype=np.uint16)))
    count = 0
    selected_time_idxs = None

    while count < max_clip_count:
        current_arr_idxs = sort_idxs[current_idxs[:, 0], current_idxs[:, 1]]
        proportion = frequency_q25[current_idxs[:, 0] + 1, current_arr_idxs] / frequency_q25[0][current_arr_idxs]
        max_proportion_i = np.argmax(proportion)
        current_idxs[max_proportion_i, 1] += 1
        selected_idx = q25_cutoff_idxs[current_arr_idxs[max_proportion_i]]
        if selected_time_idxs is None:
            selected_time_idxs = np.arange(selected_idx - 2, selected_idx + 1).astype(np.uint16)
        else:
            selected_time_idxs = np.sort(np.unique(np.hstack((selected_time_idxs, np.arange(selected_idx - 2, selected_idx + 2)))))
            count = np.count_nonzero(np.abs(np.diff(selected_time_idxs)) > 1)

    selected_time_idxs = selected_time_idxs[np.logical_and(np.argwhere(selected_time_idxs > 0).flatten(), np.argwhere(selected_time_idxs < q25_cutoff_idxs[-1]).flatten())]
    end_is = np.argwhere(np.diff(selected_time_idxs) > 1).flatten()
    start_times, end_times = np.hstack((0, end_is[0:-1] + 1)), np.hstack((end_is[0:-1], selected_time_idxs.shape[0] - 1))

    timing_view[:, 0][0:start_times.shape[0]] = selected_time_idxs[start_times] * bucket_seconds
    timing_view[:, 1][0:len(end_times)] = selected_time_idxs[end_times] * bucket_seconds


def save_panda_data(stream_data, panda_data):
    path = f'{Dir.clips_dir}'
    for path_extension in (date.today().strftime("%m-%d-%y"), stream_data.streamer, stream_data.title):
        path = f'{path}/{path_extension}'
        if not os.path.exists(path):
            os.mkdir(path)

    with open(f'{path}/panda_data.pkl', 'wb') as f:
        pickle.dump(panda_data, f)


def string_comparison_child(frequency_array_mp, chat_msg_mp, communication_array, comparison_str, array_start_idx, text_analyzed_i, mp_idxs):
    while 1:
        while communication_array[text_analyzed_i]:
            pass
        chat_msg = chat_msg_mp[0:communication_array[mp_idxs.text_length.value]]
        # print(communication_array[mp_idxs.text_length.value])
        for text in comparison_str:
            if text in chat_msg:
                idx = array_start_idx + communication_array[mp_idxs.text_array_idx.value]
                frequency_array_mp[idx] = frequency_array_mp[idx] + 1
                break

        communication_array[text_analyzed_i] = 1


def chat_analysis_process(urls, clip_timings_mp, game_params, streamer_params, chat_analysis_finished, max_clip_count, bucket_seconds):
    timing_view = np.ndarray((max_clip_count, 2), dtype=np.uint16, buffer=clip_timings_mp._obj)
    stream_data_all = [ChatGenerator().return_stream_data(url) for url in urls]
    comparison_strs_common = {}
    set_string_comparisons_dict(Dir.common_params, comparison_strs_common)
    for i, stream_data in enumerate(stream_data_all):
        print(f'Chat analysis {i} of {i / len(stream_data_all)} started')
        # Awaits download child to copy the timings and then set the value back to 0
        while chat_analysis_finished.value == 1:
            pass

        comparison_strings = comparison_strs_common.copy()
        for param_value, params, directory in zip((stream_data.streamer, stream_data.game), (streamer_params, game_params), (Dir.streamer_params, Dir.game_params)):
            if param_value in params:
                set_string_comparisons_dict(f'{directory}/{param_value}', comparison_strings)
        array_size = stream_data.duration // bucket_seconds if (stream_data.duration % bucket_seconds) == 0 else (stream_data.duration // bucket_seconds) + 1
        current_text_mp = mpc.Array('u', Config.twitch_character_limit)
        chat_frequency = CTypeSharedArray('H', array_size * (len(comparison_strings) + 1), view_shape=(array_size, (len(comparison_strings) + 1)), order='F')
        child_comm = CTypeSharedArray('H', MpIdxs.text_analyzed_start_i.value + len(comparison_strings), MpIdxs.text_analyzed_start_i.value + len(comparison_strings))
        child_comm.fill(1, start_i=MpIdxs.text_analyzed_start_i.value)

        children = [Process(target=string_comparison_child, args=(chat_frequency.mp, current_text_mp, child_comm.mp, comparison_strings[key], (i + 1) * array_size, MpIdxs.text_analyzed_start_i.value + i, MpIdxs))
                    for i, key in enumerate(comparison_strings.keys())]
        panda_data = []

        for child in children:
            child.start()
        for child in children:
            while not child.is_alive():
                time.sleep(.01)

        for msg in stream_data:
            try:
                current_text_mp._obj[0:len(msg['msg'])] = msg['msg']
            # Ignores Emojis
            except TypeError:
                continue
            child_comm.view[MpIdxs.text_length.value] = len(msg['msg'])
            child_comm.view[MpIdxs.text_array_idx.value] = int(msg['seconds'] // bucket_seconds)
            chat_frequency.view[int(msg['seconds'] // bucket_seconds), 0] += 1
            child_comm.clear(start_i=MpIdxs.text_analyzed_start_i.value)

            panda_dict = {
                'seconds': msg['seconds'],
                'msg': msg['msg'],
            }
            panda_data.append(panda_dict)
            while np.any(child_comm.view[MpIdxs.text_analyzed_start_i.value:] != 1):
                pass

        np.save('data_test.npy', chat_frequency.view.copy())
        save_panda_data(stream_data, panda_data)
        set_clip_timings(timing_view, chat_frequency, bucket_seconds, max_clip_count)
        chat_analysis_finished.value = 1
        print(f'Chat analysis {i} of {i/ len(stream_data_all)} finished')
    print('Chat child exited')


def get_subprocess_stream_dict(urls) -> (list, list, list):
    stream_data_all = [ChatGenerator().return_stream_data(url) for url in urls]
    stream_data_dict = {}
    for stream_data in stream_data_all:
        stream_data_dict[stream_data.vod_id] = {'streamer': stream_data.streamer, 'title': stream_data.title,
                                               'game': stream_data.game}

    return stream_data_dict


def main():
    max_clip_count, bucket_size = 16, 15

    urls = []
    streamer_params, game_params = os.listdir(Dir.streamer_params)[1:], os.listdir(Dir.game_params)[1:]

    with open('urls.txt', 'r') as f:
        for url in f:
            urls.append(url.strip())

    chat_analysis_finished, download_finished, editing_finished = mpc.Value('B', 0), mpc.Value('B', 0), mpc.Value('B', 0)
    stream_data = get_subprocess_stream_dict(urls)
    clip_timings = CTypeSharedArray('H', max_clip_count * 2, view_shape=(max_clip_count, 2))

    download_child = Process(target=download_child_process, args=(clip_timings.mp, stream_data, chat_analysis_finished, editing_finished, max_clip_count))
    analysis_child = Process(target=chat_analysis_process, args=(urls, clip_timings.mp, game_params, streamer_params, chat_analysis_finished, max_clip_count, bucket_size))

    for p in (analysis_child, download_child):
        p.start()

    for p in (analysis_child, download_child):
        p.join()


    '''for url in urls:
        chat = ChatGenerator().return_stream_data(url)
        # frequency_data, panda_arr = return_frequency_data(stream_data, comparison_strs_common, comparison_audio_common, game_params, streamer_params)
        # np.save('data_test.npy', frequency_data)
        frequency_data = np.load('data_test.npy')
        time_stamps = return_timestamps(frequency_data, 15)
        get_clips(chat.vod_id, chat.streamer, chat.title, time_stamps)'''


def return_data_sample(data_bucket_seconds, chat_object):
    data_arr_length = chat_object.duration // data_bucket_seconds
    if (chat_object.duration % data_bucket_seconds) != 0:
        data_arr_length += 1

    data_array, panda_data = np.zeros((data_arr_length, 2), dtype=np.uint16), []

    for msg in chat_object:
        data_array[int(msg['seconds'] // data_bucket_seconds), 0] += 1
        panda_dict = {
            'seconds': msg['seconds'],
            'msg': msg['msg'],
        }
        panda_data.append(panda_dict)

    return data_array, pd.DataFrame().from_records(panda_data)


if __name__ == '__main__':
    main()
    # data = np.load('data_test.npy')
    # return_timestamps(data, 15, desired_clip_count=8)
