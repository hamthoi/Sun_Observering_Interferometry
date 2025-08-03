import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
import math
import tkinter as tk
from tkinter import filedialog

def get_folder_file_list(path, folder):
    _file_list = os.listdir(os.path.join(path, folder))
    _file_list.sort()
    _number_file = 0
    _file_index = 0

    for file in _file_list:
        if file < f"{folder}_10_00_00_PSD.csv":
            _file_index += 1
            continue
        elif file > f"{folder}_14_00_00_PSD.csv":
            continue
        _number_file += 1

    if not _file_list:
        return [], None, None, None, 0

    _frequency = np.loadtxt(os.path.join(path, folder, _file_list[0]), delimiter=',')[:,0]
    _timeStamp_list = np.zeros(_number_file)
    _timeString_list = ["" for _ in range(_number_file)]

    idx = 0
    for file in _file_list:
        if file < f"{folder}_10_00_00_PSD.csv" or file > f"{folder}_14_00_00_PSD.csv":
            continue
        components = file.split("_")
        date_str = f"{components[0]}-{components[1]}-{components[2]}"
        time_str = f"{components[3]}:{components[4]}:{components[5]}"
        _timeString_list[idx] = time_str
        dateTime_str = f"{date_str} {time_str}"
        dateTime = dt.datetime.strptime(dateTime_str, '%Y-%m-%d %H:%M:%S')
        _timeStamp_list[idx] = dateTime.timestamp()
        idx += 1

    return _file_list, _frequency, _timeStamp_list, _timeString_list, _number_file

def calculate_power(path, folder, file_list, number_file, Frequency):
    power_list = np.zeros(number_file)
    data_matrix = np.zeros((len(Frequency), number_file))
    _file_index = 0

    for file in file_list:
        if file < f"{folder}_10_00_00_PSD.csv":
            _file_index += 1

    idx = 0
    for file in file_list:
        if file < f"{folder}_10_00_00_PSD.csv" or file > f"{folder}_14_00_00_PSD.csv":
            continue
        data = np.loadtxt(os.path.join(path, folder, file), delimiter=',')
        data_matrix[:, idx] = data[:, 1]
        power_list[idx] = data[:, 1].mean()
        idx += 1

    return power_list, data_matrix

def remove_spike(x, threshold=3, window_size=11):
    x_mean = np.mean(x)
    x_std = np.std(x)
    spike = np.abs(x - x_mean) > threshold * x_std
    x_smooth = np.copy(x)
    x_smooth[spike] = np.mean(x[~spike])
    x_smooth = np.convolve(x_smooth, np.ones(window_size)/window_size, mode='same')
    return x_smooth

def plot_graph(time_list, power_list, date, output_folder):
    fig, ax = plt.subplots(1, 1, figsize=[24, 8])
    ax.set_title(f'{date}', fontsize=20)
    ax.set(xlabel='Time(s)', ylabel='Signal(dB)')
    ax.grid()
    ax.plot(time_list, power_list, color='green', linestyle='-')
    x = np.arange(0, len(time_list))
    step = max(1, len(time_list)//25)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(time_list[::step], rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(os.path.join(output_folder, f'avg_power_{date}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_histogram(data_matrix, Frequency, time_list, date, output_folder):
    fig, ax = plt.subplots(1, 1, figsize=[24, 8], layout='constrained')
    ax.set_title(f'{date}', fontsize=20)
    ax.set_xlabel('Time(hh:mm:ss) UTC+07', fontsize=16)
    ax.set_ylabel('Frequency(MHz)', fontsize=16)
    im = ax.imshow(data_matrix, cmap='inferno', interpolation='nearest', aspect='auto', origin='lower')
    x = np.arange(0, len(time_list))
    step_x = max(1, len(time_list)//50)
    ax.set_xticks(x[::step_x])
    ax.set_xticklabels(time_list[::step_x], rotation=45, ha='right')
    y = np.arange(0, len(Frequency))
    step_y = max(1, len(Frequency)//8)
    ax.set_yticks(y[::step_y])
    ax.set_yticklabels(np.round(Frequency[::step_y], 3), rotation=0, ha='right')
    plt.colorbar(im, ax=ax)
    fig.savefig(os.path.join(output_folder, f'spectrogram_{date}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    root = tk.Tk()
    root.withdraw()
    data_root = filedialog.askdirectory(title="Select the root folder containing day folders")
    if not data_root:
        print("No data folder selected. Exiting.")
        return
    output_folder = filedialog.askdirectory(title="Select the output folder for PNGs")
    if not output_folder:
        print("No output folder selected. Exiting.")
        return

    # List all folders in the selected root
    data_folder_list = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    data_folder_list.sort()

    for folder in data_folder_list:
        file_list, Frequency, timeStamp_list, timeString_list, number_file = get_folder_file_list(data_root, folder)
        if Frequency is None or number_file == 0 or not file_list:
            print(f"Skipping folder {folder}: no valid data.")
            continue
        power_list, data_matrix = calculate_power(data_root, folder, file_list, number_file, Frequency)
        if power_list is None or np.all(power_list == 0) or np.isnan(power_list).all():
            print(f"Skipping folder {folder}: power_list is all zeros or NaN.")
            continue
        # Remove spikes from power_list
        power_list = remove_spike(power_list, threshold=3, window_size=11)
        # Remove spikes from each frequency bin in data_matrix
        for i in range(data_matrix.shape[0]):
            data_matrix[i, :] = remove_spike(data_matrix[i, :], threshold=3, window_size=11)
        # Use folder name as date string
        date_str = folder
        plot_graph(timeString_list, power_list, date_str, output_folder)
        plot_histogram(data_matrix, Frequency, timeString_list, date_str, output_folder)
        print(f"Plots generated for {folder}")

if __name__ == "__main__":
    main()