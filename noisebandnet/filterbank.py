import torch.nn as nn
from scipy import signal
import numpy as np

class FilterBank(nn.Module):
    """Filterbank class that builds a filterbank with linearly and logarithmically distributed filters.

    Args:
    ----------
    n_filters_linear : int
        Number of linearly distributed filters
    n_filters_log : int
        Number of logarithmically distributed filters
    linear_min_f : float
        Low pass filter cutoff frequency
    linear_max_f_cutoff_fs : float
        Portion of the spectrum that is linearly distributed in a fraction of the sampling rate
    fs : int
        Sampling rate
    attenuation : float
        FIR filter attenuation used in the Kaiser window (in dB)
    """
    def __init__(self, n_filters_linear = 1024, n_filters_log = 1024, linear_min_f = 20, linear_max_f_cutoff_fs = 4,  fs = 44100, attenuation=50):
        super().__init__()
        print("Building filterbank...")
        frequency_bands = self.get_frequency_bands(n_filters_linear=n_filters_linear, n_filters_log=n_filters_log, linear_min_f=linear_min_f, linear_max_f_cutoff_fs=linear_max_f_cutoff_fs,  fs=fs)
        self.band_centers = self.get_band_centers(frequency_bands=frequency_bands, fs=fs)
        self.filters = self.build_filterbank(frequency_bands=frequency_bands, fs=fs, attenuation=attenuation)
        self.max_filter_len = max(len(array) for array in self.filters)

        print(f"Done. {len(self.filters)} filters, max filter length: {self.max_filter_len}")

    def get_linear_bands(self, n_filters_linear, linear_min_f, linear_max_f_cutoff_fs, fs):
        linear_max_f = (fs/2)/linear_max_f_cutoff_fs
        linear_bands = np.linspace(linear_min_f, linear_max_f, n_filters_linear)    
        linear_bands = np.vstack((linear_bands[:-1], linear_bands[1:])).T
        return linear_bands
    
    def get_log_bands(self, n_filters_log, linear_max_f_cutoff_fs, fs):
        linear_max_f = (fs/2)/linear_max_f_cutoff_fs
        log_bands = np.geomspace(start=linear_max_f, stop=fs/2, num=n_filters_log, endpoint=False)
        log_bands = np.vstack((log_bands[:-1], log_bands[1:])).T
        return log_bands

    def get_frequency_bands(self, n_filters_linear, n_filters_log, linear_min_f, linear_max_f_cutoff_fs,  fs):
        linear_bands = self.get_linear_bands(n_filters_linear=n_filters_linear, linear_min_f=linear_min_f, linear_max_f_cutoff_fs=linear_max_f_cutoff_fs, fs=fs)
        if linear_max_f_cutoff_fs==1:
            return linear_center_f
        log_bands = self.get_log_bands(n_filters_log=n_filters_log, linear_max_f_cutoff_fs=linear_max_f_cutoff_fs, fs=fs)
        return np.concatenate((linear_bands, log_bands))

    def get_band_centers(self, frequency_bands, fs):
        mean_frequencies = np.mean(frequency_bands, axis=1)
        lower_edge = frequency_bands[0,0]/2    
        upper_edge = ((fs/2)+frequency_bands[-1,-1])/2
        return np.concatenate(([lower_edge],mean_frequencies,[upper_edge]))

    def get_filter(self, cutoff, fs, attenuation, pass_zero, transition_bandwidth=0.2, scale=True):
        if isinstance(cutoff, np.ndarray): #BPF
            bandwidth = abs(cutoff[1]-cutoff[0])
        elif pass_zero==True: #LPF
            bandwidth = cutoff
        elif pass_zero==False: #HPF
            bandwidth = abs((fs/2)-cutoff)
        width = (bandwidth/(fs/2))*transition_bandwidth
        N, beta = signal.kaiserord(ripple=attenuation, width=width)
        N = 2 * (N // 2) + 1 #make odd
        h = signal.firwin(numtaps=N, cutoff=cutoff, window=('kaiser', beta), scale=scale, fs=fs, pass_zero=pass_zero)
        return h
    
    def build_filterbank(self, frequency_bands, fs, attenuation):
        filters = []
        for i in range(frequency_bands.shape[0]):
            #low pass filter
            if i == 0:
                h = self.get_filter(cutoff=frequency_bands[i,0], fs=fs, attenuation=attenuation, pass_zero=True)
                filters.append(h)
            #band pass filter
            h = self.get_filter(cutoff=frequency_bands[i], fs=fs, attenuation=attenuation, pass_zero=False)
            filters.append(h)
            #high pass filter
            if i == frequency_bands.shape[0]-1:
                h = self.get_filter(cutoff=frequency_bands[i,-1], fs=fs, attenuation=attenuation, pass_zero=False)
                filters.append(h)
        return filters