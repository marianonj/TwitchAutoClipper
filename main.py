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
from ChatClasses import ChatGenerator, Chat
from Audio import audio_process_child
import unicodedata


class MpIdxs(Enum):
    text_array_idx = 0
    text_length = 1
    text_analyzed_start_i = 2


def plot_frequency(bin_size):
    data_np = np.load('data_test.npy')
    counts, bins = np.histogram(data_np)
    bins = np.arange(0, data_np.shape[0]) * bin_size

    for i, data in np.ndenumerate(data_np[:, 0]):
        plt.bar(i, data, linewidth=0, bottom=0)
    plt.show()

    print('b')

    pass


def set_comparison_dict(directory: str, comparison_strs: dict, audio_comparison_directories: list):
    for file_name in os.listdir(directory):
        if not (file_name[-3:] == 'txt' or file_name == 'audio'):
            continue

        file_path = f'{directory}/{file_name}'
        if file_name == 'audio':
            for audio in os.listdir(file_path):
                audio_comparison_directories.append(f'{file_path}/{audio}')
            pass
        else:
            with open(file_path) as f:
                key = file_name[0:-4]
                strings = f.read().strip().lower().replace('\n', ',').strip().split(sep=',')
                if not strings or (len(strings) == 1 and not strings[-1]):
                    continue

                if key in comparison_strs:
                    comparison_strs[key] = comparison_strs[key] + strings[0:-2] if not strings[-1] else strings
                else:
                    comparison_strs[key] = strings[0:-2] if not strings[-1] else strings


def start_children(comparison_strings, comparison_dict):
    pass


def string_comparison_child(frequency_array_mp, string_array_mp, communication_array, comparison_str, array_start_idx, text_analyzed_i, mp_idxs):
    while 1:
        while communication_array[text_analyzed_i]:
            pass

        chat_msg = string_array_mp[0:communication_array[mp_idxs.text_length.value]]
        for text in comparison_str:
            if text in chat_msg:
                idx = array_start_idx + communication_array[mp_idxs.text_array_idx.value]
                frequency_array_mp[idx] = frequency_array_mp[idx] + 1
                break

        communication_array[text_analyzed_i] = 1


def main():
    bucket_seconds = 10
    urls, comparison_strs_common, comparison_audio_common = [], {}, []
    streamer_params, game_params = os.listdir(Dir.streamer_params)[1:], os.listdir(Dir.game_params)[1:]
    set_comparison_dict(Dir.common_params, comparison_strs_common, comparison_audio_common)

    with open('urls.txt', 'r') as f:
        for url in f:
            urls.append(url.strip())

    current_text_mp = mpc.Array('u', Config.twitch_character_limit)
    for url in urls:
        chat = ChatGenerator().return_chat(url)
        comparison_strings, comparison_audio = comparison_strs_common.copy(), comparison_audio_common.copy()
        for param_value, params, directory in zip((chat.streamer, chat.game), (streamer_params, game_params), (Dir.streamer_params, Dir.game_params)):
            if param_value in params:
                set_comparison_dict(f'{directory}/{param_value}', comparison_strings, comparison_audio)
        array_size = chat.duration // bucket_seconds if (chat.duration % bucket_seconds) == 0 else (chat.duration // bucket_seconds) + 1
        frequency_array_mp = mpc.Array('H', array_size * (len(comparison_strings) + 1))
        frequency_array_view = np.ndarray((array_size, (len(comparison_strings) + 1)), dtype=np.uint16, buffer=frequency_array_mp._obj, order='F')
        child_communication_mp = mpc.Array('H', MpIdxs.text_analyzed_start_i.value + len(comparison_strings))
        child_communication_view = np.ndarray(MpIdxs.text_analyzed_start_i.value + len(comparison_strings), dtype=np.uint16, buffer=child_communication_mp._obj)
        child_communication_view[MpIdxs.text_analyzed_start_i.value:] = 1

        children = [Process(target=string_comparison_child, args=(frequency_array_mp, current_text_mp, child_communication_mp, comparison_strings[key], (i + 1) * array_size, MpIdxs.text_analyzed_start_i.value + i, MpIdxs))
                    for i, key in enumerate(comparison_strings.keys())]
        panda_data = []

        if comparison_audio:
            children.append(Process)
            pass

        for child in children:
            child.start()
        for child in children:
            while not child.is_alive():
                time.sleep(.01)

        for msg in chat:
            try:
                current_text_mp._obj[0:len(msg['msg'])] = msg['msg']
            except TypeError:
                continue
            print(msg['time'])
            child_communication_mp[MpIdxs.text_length.value] = len(msg['msg'])
            child_communication_mp[MpIdxs.text_array_idx.value] = int(msg['time'] // bucket_seconds)
            frequency_array_view[int(msg['time'] // bucket_seconds), 0] += 1
            child_communication_view[MpIdxs.text_analyzed_start_i.value:] = 0

            panda_dict = {
                'time': msg['time'],
                'msg': msg['msg'],
            }
            panda_data.append(panda_dict)
            while np.any(child_communication_view[MpIdxs.text_analyzed_start_i.value:] != 1):
                pass

        frequency_data, chat_data = return_data_sample(5, chat)
        np.save('data_test.npy', frequency_data)
        with open('panda_test.pickle', 'wb') as f:
            pickle.dump(chat_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def return_data_sample(data_bucket_seconds, chat_object):
    data_arr_length = chat_object.duration // data_bucket_seconds
    if (chat_object.duration % data_bucket_seconds) != 0:
        data_arr_length += 1

    data_array, panda_data = np.zeros((data_arr_length, 2), dtype=np.uint16), []

    for msg in chat_object:
        data_array[int(msg['time'] // data_bucket_seconds), 0] += 1
        panda_dict = {
            'time': msg['time'],
            'msg': msg['msg'],
        }
        panda_data.append(panda_dict)

    return data_array, pd.DataFrame().from_records(panda_data)


if __name__ == '__main__':
    main()
    plot_frequency(5)
