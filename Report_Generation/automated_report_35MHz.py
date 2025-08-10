import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os

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

def generate_daily_pngs(data_folder, avg_power_dir, spectrogram_dir):
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
    plot_graph(timeString_list, power_list, date_str, avg_power_dir)
    plot_spectrogram(data_matrix, Frequency, timeString_list, date_str, spectrogram_dir)
    print(f"Processed folder: {folder}")

def main():
    # Get yesterday's date
    yesterday = dt.datetime.now() - dt.timedelta(days=1)
    folder_name = yesterday.strftime("%Y_%m_%d")
    data_folder = f"/home/radio/Desktop/Dlite/{folder_name}"
    output_folder = "/home/radio/Desktop/Data Plots"

    avg_power_dir = os.path.join(output_folder, "avg_power")
    spectrogram_dir = os.path.join(output_folder, "spectrogram")
    os.makedirs(avg_power_dir, exist_ok=True)
    os.makedirs(spectrogram_dir, exist_ok=True)

    avg_power_path = os.path.join(avg_power_dir, f'avg_power_{folder_name}.png')
    spectrogram_path = os.path.join(spectrogram_dir, f'spectrogram_{folder_name}.png')
    
    # Check if both plots already exist
    if os.path.isfile(avg_power_path) and os.path.isfile(spectrogram_path):
        print("Plots already exist for yesterday. Exiting.")
        sys.exit(0)

    if not os.path.isdir(data_folder):
        print(f"Data folder {data_folder} does not exist. Exiting.")
        sys.exit(1)

    # Pass both subdirectories to the plotting function
    generate_daily_pngs(data_folder, avg_power_dir, spectrogram_dir)
    print("Plots generated successfully.")

if __name__ == "__main__":
    main()