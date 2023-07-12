from . import filterbank
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

#some snippets from https://github.com/acids-ircam/ddsp_pytorch
def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)

def gru(n_input, hidden_size):
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)

def compute_magnitude_filters(filters):
    magnitude_filters = torch.fft.rfft(filters)
    magnitude_filters = torch.abs(magnitude_filters)
    return magnitude_filters

def check_power_of_2(x):
    return 2 ** int(math.log(x, 2)) == x

def get_next_power_of_2(x):
    return int(math.pow(2, math.ceil(math.log(x)/math.log(2))))

def pad_filters(filters, n_samples):
    for i in range(len(filters)):
        filters[i] = np.pad(filters[i], (n_samples-len(filters[i]),0))
    return torch.from_numpy(np.array(filters))

def get_noise_bands(fb, min_noise_len, normalize):
    #build deterministic loopable noise bands
    if fb.max_filter_len > min_noise_len:
        noise_len = get_next_power_of_2(fb.max_filter_len)
    else:
        noise_len = min_noise_len
    filters = pad_filters(fb.filters, noise_len)
    magnitude_filters = compute_magnitude_filters(filters=filters)
    torch.manual_seed(42) #enforce deterministic noise
    phase_noise = torch.FloatTensor(magnitude_filters.shape[0], magnitude_filters.shape[-1]).uniform_(-math.pi, math.pi).to(magnitude_filters.device)
    phase_noise = torch.exp(1j*phase_noise)
    phase_noise[:,0] = 0
    phase_noise[:,-1] = 0
    magphase = magnitude_filters*phase_noise
    noise_bands = torch.fft.irfft(magphase)
    if normalize:
        noise_bands = (noise_bands / torch.max(noise_bands.abs())) 
    return noise_bands.unsqueeze(0).float(), noise_len

class NoiseBandNet(nn.Module):
    """NoiseBandNet synthesiser class.

    Args:
    ----------
    hidden_size : int
        Internal hidden size for all the layer
    n_band : int
        Number of bands in the filterbank
    synth_window : int
        Synthesis window size (in samples), the internal sampling rate 
        is sampling rate divided by synth_window
    filterbank_attenuation : float
        FIR filter attenuation used in the Kaiser window (in dB)
    fs : int
        Sampling rate
    attenuation : float
        FIR filter attenuation used in the Kaiser window (in dB)
    min_noise_len : int
        Minimum noise length (in samples) for the noise bands, must be a power of 2
    n_control_params : int
        Number of control parameters
    linear_min_f : float
        Low pass filter cutoff frequency
    linear_max_f_cutoff_fs : float
        Portion of the spectrum that is linearly distributed in a fraction of the sampling rate
    normalize_noise_bands : bool
        Normalize the noise bands to the abslute maximum value. Scale up the noise bands amplitude.
    """
    def __init__(self, hidden_size, n_band, synth_window, filterbank_attenuation=50, fs=44100, min_noise_len = 2**16, 
                n_control_params=2, linear_min_f = 20, linear_max_f_cutoff_fs = 4, normalize_noise_bands = True):
        super().__init__()
        assert min_noise_len > 0 and isinstance(min_noise_len, int) and check_power_of_2(min_noise_len), "min_noise_len must be a positive integer and a power of 2"
        self.n_band = n_band
        self.synth_window = synth_window
        fb  = filterbank.FilterBank(n_filters_linear = n_band//2, n_filters_log = n_band//2, linear_min_f = linear_min_f, linear_max_f_cutoff_fs = linear_max_f_cutoff_fs,  fs = fs, attenuation = filterbank_attenuation)
        self.center_frequencies = fb.band_centers #store center frequencies for reference
        self.noise_bands, self.noise_len = get_noise_bands(fb=fb, min_noise_len=min_noise_len, normalize=normalize_noise_bands)

        in_mlps = []
        for i in range(n_control_params):
            in_mlps.append(mlp(1, hidden_size, 1))
        self.in_mlps = nn.ModuleList(in_mlps)
        self.gru = gru(n_control_params, hidden_size)
        self.out_mlp = mlp(hidden_size + n_control_params, hidden_size, 3)
        self.amplitudes_layer = nn.Linear(hidden_size, n_band)

    def scale_function(self, x):
        return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-18

    def synth_batch(self, amplitudes):
        """Apply the predicted amplitudes to the noise bands.
        Args:
        ----------
        amplitudes : torch.Tensor
            Predicted amplitudes

        Returns:
        ----------  
        signal : torch.Tensor
            Output audio signal
        """

        #synth in noise_len frames to fit longer sequences on GPU memory
        frame_len = int(self.noise_len/self.synth_window)
        n_frames = math.ceil(amplitudes.shape[-1]/frame_len)
        self.noise_bands = self.noise_bands.to(amplitudes.device)
        #avoid overfitting to noise values
        self.noise_bands = torch.roll(self.noise_bands, shifts=int(torch.randint(low=0, high=self.noise_bands.shape[-1], size=(1,))), dims=-1)
        signal_len = amplitudes.shape[-1]*self.synth_window
        #smaller amp len than noise_len
        if amplitudes.shape[-1]/frame_len < 1:
            upsampled_amplitudes = F.interpolate(amplitudes, scale_factor=self.synth_window, mode='linear')
            signal = (self.noise_bands[..., :signal_len]*upsampled_amplitudes).sum(1, keepdim=True)
        else:
            for i in range(n_frames):
                if i == 0:
                    upsampled_amplitudes = F.interpolate(amplitudes[..., :frame_len], scale_factor=self.synth_window, mode='linear')
                    signal = (self.noise_bands*upsampled_amplitudes).sum(1, keepdim=True)
                #last iteration
                elif i == (n_frames-1):
                    upsampled_amplitudes = F.interpolate(amplitudes[..., i*frame_len:], scale_factor=self.synth_window, mode='linear')
                    signal = torch.cat([signal, (self.noise_bands[...,:upsampled_amplitudes.shape[-1]]*upsampled_amplitudes).sum(1, keepdim=True)], dim=-1)
                else:
                    upsampled_amplitudes = F.interpolate(amplitudes[..., i*frame_len:(i+1)*frame_len], scale_factor=self.synth_window, mode='linear')
                    signal = torch.cat([signal, (self.noise_bands*upsampled_amplitudes).sum(1, keepdim=True)], dim=-1)
        return signal 


    def forward_amplitudes(self, control_params):
        """Predict filterbank amplitudes from control parameters.
        Args:
        ----------
        control_params : list of torch.Tensor
            List of control parameters

        Returns:
        ----------  
        amplitudes : torch.Tensor
            Predicted filterbank amplitudes
        """
        hidden = []
        for i in range(len(self.in_mlps)):
            hidden.append(self.in_mlps[i](control_params[i]))
        hidden = torch.cat(hidden, dim=-1)
        hidden = self.gru(hidden)[0]
        for i in range(len(control_params)):
            hidden = torch.cat([hidden, control_params[i]], dim=-1)
        hidden = self.out_mlp(hidden)
        amplitudes = self.amplitudes_layer(hidden).permute(0,2,1)
        amplitudes = self.scale_function(amplitudes)
        return amplitudes

    def forward_random(self, control_params, frame_len, frequency_shifts, k_amplitudes, k_low_mult=0.5, k_high_mult=1.5, init_f_shifts=0):
        """Apply randomisation to the predicted amplitudes.
            Args:
        ----------
        control_params : list of torch.Tensor
            List of control parameters
        frame_len : int
            Frame lenght to apply the randomisation scheme. The audio is divided in frames of this length.
        frequency_shifts : int
            Maximum frequency shift in Hz (positive and negative)
        k_amplitudes : int
            Number of maximum amplitude values to randomise in each frame
        k_low_mult : float
            Mininum amplitude multiplier
        k_high_mult : float
            Maximum amplitude multiplier
        init_f_shifts : int
            Maximum frequency shift in Hz (positive and negative) to apply to the first frame
        
        Returns:
        ----------  
        signal : torch.Tensor
            Predicted audio with randomised amplitudes

        """
        amplitudes = self.forward_amplitudes(control_params)
        frames = amplitudes.shape[-1]//frame_len
        shift = 0
        if init_f_shifts>0:
            init_shift = torch.randint(low=-init_f_shifts, high=init_f_shifts+1, size=(1,)).item()
            amplitudes = torch.roll(amplitudes, init_shift, dims=1)
        for i in range(frames):
            shift = shift+torch.randint(low=-frequency_shifts, high=frequency_shifts+1, size=(1,)).item()
            chunk = amplitudes[:,:,(i*frame_len):(i*frame_len)+frame_len].sum(-1)
            _, topk_idx = torch.topk(chunk, k=k_amplitudes, dim=-1)
            amplitudes[:,topk_idx.squeeze(0),(i*frame_len):(i*frame_len)+frame_len] *= torch.FloatTensor(amplitudes.shape[0], k_amplitudes).uniform_(k_low_mult, k_high_mult).unsqueeze(-1).to(amplitudes.device)
            amplitudes[:,:,(i*frame_len):(i*frame_len)+frame_len] = torch.roll(amplitudes[:,:,(i*frame_len):(i*frame_len)+frame_len], shift, dims=1)
        #randomise the leftover frame (if any)
        if amplitudes.shape[-1]%frame_len != 0:
            shift = shift+torch.randint(low=-frequency_shifts, high=frequency_shifts+1, size=(1,)).item()
            chunk = amplitudes[:,:,((i*frame_len)+frame_len):].sum(-1)
            _, topk_idx = torch.topk(chunk, k=k_amplitudes, dim=-1)
            amplitudes[:,topk_idx.squeeze(0),((i*frame_len)+frame_len):] *= torch.FloatTensor(amplitudes.shape[0], k_amplitudes).uniform_(k_low_mult, k_high_mult).unsqueeze(-1).to(amplitudes.device)
            amplitudes[:,:,((i*frame_len)+frame_len):] = torch.roll(amplitudes[:,:,((i*frame_len)+frame_len):], shift, dims=1)
        signal = self.synth_batch(amplitudes=amplitudes)
        return signal

    def forward(self, control_params):
        """Outputs audio from control parameters.
        Args:
        ----------
        control_params : list of torch.Tensor
            List of control parameters

        Returns:
        ----------  
        signal : torch.Tensor
            Predicted audio
        """
        amplitudes = self.forward_amplitudes(control_params)
        signal = self.synth_batch(amplitudes=amplitudes)
        return signal