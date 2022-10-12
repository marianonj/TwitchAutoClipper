import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from ChatClasses import ChatGenerator, Chat


def plot_frequency(bin_size):
    data_np = np.load('data_test.npy')
    counts, bins = np.histogram(data_np)
    bins = np.arange(0, data_np.shape[0]) * bin_size

    for i, data in np.ndenumerate(data_np[:,0]):
        plt.bar(i, data, linewidth=0, bottom=0)
    plt.show()

    print('b')

    pass


def main():
    urls = []
    with open('urls.txt', 'r') as f:
        for url in f:
            urls.append(url.strip())

    for url in urls:
        chat = ChatGenerator().return_chat(url)
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
