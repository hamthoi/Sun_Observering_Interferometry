import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
import sys

def extract_timestamp(file):
    base = os.path.basename(file)
    parts = base.split('_')
    date_str = f"{parts[0]}-{parts[1]}-{parts[2]}"
    time_str = f"{parts[3]}:{parts[4]}:{parts[5]}"
    return dt.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")

def remove_spike(x, threshold=3, window_size=11):
    x_mean = np.mean(x)
    x_std = np.std(x)
    spike = np.abs(x - x_mean) > threshold * x_std
    x_smooth = np.copy(x)
    x_smooth[spike] = np.mean(x[~spike])
    x_smooth = np.convolve(x_smooth, np.ones(window_size)/window_size, mode='same')
    return x_smooth

def generate_daily_pngs(data_folder, output_dirs):
    avg_power_dir, spectrogram_dir = output_dirs
    os.makedirs(avg_power_dir, exist_ok=True)
    os.makedirs(spectrogram_dir, exist_ok=True)
    file_list = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    file_list.sort()
    if not file_list:
        print("No .csv files found in the selected data folder.")
        return
    number_file = len(file_list)
    time_list = np.zeros(number_file)
    time1_list = ["" for _ in range(number_file)]
    power_list = np.zeros(number_file)
    frequency = None
    data_matrix = None

    # Get frequency from first file
    frequency = np.loadtxt(os.path.join(data_folder, file_list[0]), delimiter=',')[:,0]
    data_matrix = np.zeros((len(frequency), number_file))

    # Parse times and load data
    for idx, file in enumerate(file_list):
        components = file.split("_")
        date_str = f"{components[0]}-{components[1]}-{components[2]}"
        time_str = f"{components[3]}:{components[4]}:{components[5]}"
        time1_list[idx] = time_str
        dateTime_str = f"{date_str} {time_str}"
        dateTime = dt.datetime.strptime(dateTime_str, '%Y-%m-%d %H:%M:%S')
        time_list[idx] = dateTime.timestamp()
        data = np.loadtxt(os.path.join(data_folder, file), delimiter=',')
        data_matrix[:, idx] = data[:, 1]
        power_list[idx] = data[:, 1].mean()

    date_str = file_list[0][:10]  # YYYY_MM_DD

    # Plot 1: Average power vs. time (with spike removal)
    power_smooth = remove_spike(power_list, threshold=3, window_size=11)
    fig, ax = plt.subplots(1, 1, figsize=[24, 8])
    ax.set_title(f'{date_str}', fontsize=20)
    ax.set(xlabel='Time(s)', ylabel='Signal(dB)')
    ax.grid()
    ax.plot(time_list, power_list, color='green', linestyle='-', label='Raw')
    ax.plot(time_list, power_smooth, color='red', linestyle='--', linewidth=2.5, label='Smoothed')
    ymin = np.min(power_list)
    ymax = np.max(power_list)
    margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.1
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.legend()
    x = np.arange(0, len(time1_list))
    ax.set_xticks(time_list[::max(1, len(time1_list)//50)])
    ax.set_xticklabels([t for t in time1_list[::max(1, len(time1_list)//50)]], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(avg_power_dir, f'avg_power_{date_str}.png'), dpi=300)
    plt.close()

    # Plot 2: Spectrogram (frequency vs. time)
    fig, ax_2 = plt.subplots(1, 1, figsize=[24, 8], layout='constrained')
    ax_2.set_title(f'{date_str}', fontsize=20)
    ax_2.set_xlabel('Time(hh:mm:ss) UTC+07', fontsize=16)
    ax_2.set_ylabel('Frequency(MHz)', fontsize=16)
    im = ax_2.imshow(data_matrix, cmap='inferno', interpolation='nearest', aspect='auto', origin='lower')
    x = np.arange(0, len(time1_list))
    ax_2.set_xticks(x[::max(1, len(time1_list)//50)])
    ax_2.set_xticklabels([t for t in time1_list[::max(1, len(time1_list)//50)]], rotation=45, ha='right')
    y = np.arange(0, len(frequency))
    ax_2.set_yticks(y[::max(1, len(frequency)//8)])
    ax_2.set_yticklabels(np.round(frequency[::max(1, len(frequency)//8)], 3), rotation=0, ha='right')
    plt.colorbar(im, ax=ax_2)
    plt.savefig(os.path.join(spectrogram_dir, f'spectrogram_{date_str}.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
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
    generate_daily_pngs(data_folder, (avg_power_dir, spectrogram_dir))
    print("Plots generated successfully.")