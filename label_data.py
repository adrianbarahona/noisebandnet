import matplotlib.pyplot as plt
import numpy as np
import argparse
import librosa as li
import os

def label_time_series(time_series, sampling_rate, n_fft=1024, noverlap=256):
    fig, (ax1, ax2) = plt.subplots(nrows=2)

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    ax1.plot(time_series, c='salmon')
    ax1.set_xlabel('Time (samples)')
    ax1.set_ylabel('Amplitude')
    ax1.grid()
    ax1.set_xlim(left=0, right=len(time_series))

    Pxx, freqs, bins, im = ax2.specgram(time_series, NFFT=n_fft, Fs=sampling_rate, noverlap=noverlap, cmap='magma')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')

    points = []
    values = []

    def onclick(event):
        nonlocal points, values
        x, y = event.xdata, event.ydata
        if event.button == 1 and (len(points) == 0 or x > points[-1]):
            # Left mouse button: add a new point
            points.append(x)
            values.append(y)
            ax2.plot(points, values, 'ro-', c='cyan', linewidth=5)
            plt.draw()
        elif event.button == 3:
            # Right mouse button: delete the last placed point
            if len(points) > 0:
                points.pop()
                values.pop()
                ax2.clear()
                Pxx, freqs, bins, im = ax2.specgram(time_series, NFFT=n_fft, Fs=sampling_rate, noverlap=noverlap, cmap='magma')
                ax2.plot(points, values, 'ro-', c='cyan', linewidth=5)
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Frequency (Hz)')
                plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    x, y = zip(*sorted(zip(points, values)))
    duration = len(time_series) / sampling_rate
    new_x = np.linspace(0, duration, num=len(time_series))
    new_y = np.interp(new_x, x, y, left=np.nan, right=np.nan)
    mask = np.isnan(new_y)
    new_y[mask] = np.interp(new_x[mask], new_x[~mask], new_y[~mask])
    new_y = (new_y - np.nanmin(new_y)) / (np.nanmax(new_y) - np.nanmin(new_y))
    return new_y

def load_audio(path, fs, norm=True):
    x = li.load(path, sr=fs, mono=True)[0]
    if norm:
        x = li.util.normalize(x)
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', help='Directory of the training sound', default='training_sounds')
    parser.add_argument('--audio_name', help='Name of the training sound', default='metal')
    parser.add_argument('--output_directory', help='Where to put the file', default="labels_train")
    parser.add_argument('--feature_name', help='Name of the feature for the output file', default="control_1")
    parser.add_argument('--sampling_rate', help='Fs of the sounds', default=44100)

    config = parser.parse_args()
    audio_in = f'{config.audio_path}/{config.audio_name}.wav'
    audio = load_audio(path=audio_in, fs=config.sampling_rate)
    audio = audio+1e-8
    #create a folder if it doesn't exist
    if not os.path.exists(f'{config.output_directory}/{config.audio_name}'):
        os.makedirs(f'{config.output_directory}/{config.audio_name}')

    # Label the time series with mouse clicks and interpolate between points
    y_values = label_time_series(audio, config.sampling_rate)

    # Plot the control parameter 
    fig, ax = plt.subplots()
    ax.set_title(f'Control parameter for {config.audio_name}')
    ax.plot(y_values)
    ax.legend()
    plt.show()
    #save the file in the output directory
    np.save(f'{config.output_directory}/{config.audio_name}/{config.feature_name}', y_values)
