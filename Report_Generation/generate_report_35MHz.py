import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
import tkinter as tk
from tkinter import filedialog

def get_folder_file_list(path, folder):
    _file_list = os.listdir(os.path.join(path, folder))
    _file_list.sort()
    _number_file = len(_file_list)

    if not _file_list:
        return [], None, None, None, 0

    _frequency = np.loadtxt(os.path.join(path, folder, _file_list[0]), delimiter=',')[:,0]
    _timeStamp_list = np.zeros(_number_file)
    _timeString_list = ["" for _ in range(_number_file)]

    for idx, file in enumerate(_file_list):
        components = file.split("_")
        if len(components) < 6:
            continue
        date_str = f"{components[0]}-{components[1]}-{components[2]}"
        time_str = f"{components[3]}:{components[4]}:{components[5]}"
        _timeString_list[idx] = time_str
        dateTime_str = f"{date_str} {time_str}"
        try:
            dateTime = dt.datetime.strptime(dateTime_str, '%Y-%m-%d %H:%M:%S')
            _timeStamp_list[idx] = dateTime.timestamp()
        except Exception:
            _timeStamp_list[idx] = 0

    return _file_list, _frequency, _timeStamp_list, _timeString_list, _number_file

def calculate_power(path, folder, file_list, number_file, Frequency):
    power_list = np.zeros(number_file)
    data_matrix = np.zeros((len(Frequency), number_file))

    for idx, file in enumerate(file_list):
        data = np.loadtxt(os.path.join(path, folder, file), delimiter=',')
        data_matrix[:, idx] = data[:, 1]
        power_list[idx] = data[:, 1].mean()

    return power_list, data_matrix

def remove_spike(x, threshold=3, window_size=11):
    x_mean = np.mean(x)
    x_std = np.std(x)
    spike = np.abs(x - x_mean) > threshold * x_std
    x_smooth = np.copy(x)
    x_smooth[spike] = np.mean(x[~spike])
    x_smooth = np.convolve(x, np.ones(window_size)/window_size, mode='same')
    return x_smooth

def plot_graph(time_list, power_list, date, output_folder):
    fig, ax = plt.subplots(1, 1, figsize=[24, 8])
    ax.set(ylabel='Signal(dB)')
    ax.grid()
    ax.plot(time_list, power_list, color='green', linestyle='-')
    ax.set_title(f'{date}', fontsize=13)
    x = np.arange(len(time_list))
    step = max(1, len(time_list) // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([time_list[i] for i in x[::step]], rotation=45, ha='right')
    fig.savefig(os.path.join(output_folder, f"avg_power_{date}.png"), dpi=600, bbox_inches='tight')
    plt.close()

def plot_spectrogram(data_matrix, Frequency, time_list, date, output_folder):
    fig, ax = plt.subplots(1, 1, figsize=[24, 8])
    ax.set(ylabel='Signal (dB)')
    ax.imshow(data_matrix, cmap='inferno', interpolation='nearest', aspect='auto', origin='lower')
    ax.set_title(date, fontsize=13)

    # Trục X – thời gian
    step_x = max(1, len(time_list) // 20)
    x = np.arange(len(time_list))
    ax.set_xticks(x[::step_x])
    ax.set_xticklabels(time_list[::step_x], rotation=45, ha='right')
    fig.savefig(os.path.join(output_folder, f"spectrogram_{date}.png"), dpi=600, bbox_inches='tight')
    plt.close()

def main():
    root = tk.Tk()
    root.withdraw()
    data_folder = filedialog.askdirectory(title="Select the daily data folder (YYYY_MM_DD)")
    if not data_folder:
        print("No data folder selected. Exiting.")
        return
    output_folder = filedialog.askdirectory(title="Select the output folder for PNGs")
    if not output_folder:
        print("No output folder selected. Exiting.")
        return

    folder = os.path.basename(data_folder)
    parent_path = os.path.dirname(data_folder)
    file_list, Frequency, timeStamp_list, timeString_list, number_file = get_folder_file_list(parent_path, folder)
    if Frequency is None or number_file == 0 or not file_list:
        print(f"Skipping folder {folder}: no valid data.")
        return
    power_list, data_matrix = calculate_power(parent_path, folder, file_list, number_file, Frequency)
    if power_list is None or np.all(power_list == 0) or np.isnan(power_list).all():
        print(f"Skipping folder {folder}: power_list is all zeros or NaN.")
        return
    power_list = remove_spike(power_list, threshold=3, window_size=11)
    for i in range(data_matrix.shape[0]):
        data_matrix[i, :] = remove_spike(data_matrix[i, :], threshold=3, window_size=11)
    date_str = folder
    plot_graph(timeString_list, power_list, date_str, output_folder)
    plot_spectrogram(data_matrix, Frequency, timeString_list, date_str, output_folder)
    print(f"Processed folder: {folder}")

if __name__ == "__main__":
    main()