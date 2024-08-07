<h1 align="center">NoiseBandNet: Controllable Time-Varying Neural Synthesis of Sound Effects Using Filterbanks
</h1>
<div align="center">

<h4>
        <a href="https://ieeexplore.ieee.org/abstract/document/10440034/" target="_blank">Paper</a> |  <a href="https://www.adrianbarahonarios.com/noisebandnet/" target="_blank">Website</a> </a>
    </h4>
    <p>
    </p>
</div>
<p align="center"><img src="https://www.adrianbarahonarios.com/files/NBN/nbn_arch.png" width="512" /></p>

# **Installation**
Please install the requirements by running:
```
pip install -r requirements.txt
```

# **Training**
Please place all the training .wav files inside the same directory. 

To train a model just run the commands below depending on the desired control scheme. The training configuration options (batch size, number of filters, training epochs, learning rate, etc.) can be seen by typing:

```bash
python train.py --help
```

The progress is logged in a `trained_models/dataset_name/current_date` directory, where `dataset_name` is taken from the `--dataset_path` and `current_date` is the current date and time (to avoid overriding). The directory contains the checkpoints (model, training audio examples, synthesised audio examples) taken during training and a `config.pickle` file with the training configuration (for inference). 


### **Training using loudness and spectral centroid**
Used to compare NoiseBandNet to the original DDSP noise synthesiser.
```bash
python train.py --dataset_path path_to_wav_files_directory --auto_control_params loudness centroid
```

### **Training using loudness**
Used to perform loudness transfer.

```bash
python train.py --dataset_path path_to_wav_files_directory --auto_control_params loudness
```
### **Training using user-defined control parameters**
Used to control the synthesiser with user-defined control parameters. This is limited to a single audio file.

First, label the training audio by running:

```bash
python label_data.py --audio_path path_to_wav_file_directory --audio_name name_of_the_audio_file --output_directory output_directory --feature_name name_of_the_labelled_feature --sampling_rate sampling_rate_of_the_audio
```

The `label_data.py` tool will show an image with the training audio waveform at the top and its spectrogram at the bottom. The control parameters are defined by clicking on top of the spectrogram. To allow for a finer control, the right click removes the last added control point. Please see below for an example, where the cyan curve on top of the spectrogram is the user-defined control parameter:

<p align="center"><img src="https://www.adrianbarahonarios.com/files/NBN/drill_ui.png" width="256" /></p>

This will create a `feature_name.npy` file with the control parameters in a `output_directory/audio_name` directory. To train a model using this control curve, simply run:

```bash
python train.py --dataset_path path_to_wav_file_directory --control_params_path output_directory/audio_name
```

# **Inference**

We provide 3 notebooks with different inference schemes.

### **Amplitude randomisation**

The `inference_randomisation` notebook contains a demo of randomising the predicted amplitudes from the model, including generating stereo signals (Section V-A of the paper).

### **Loudness transfer**

The `inference_loudness_transfer` notebook shows how to perform loudness transfer (Section V-B of the paper).

### **User-defined control curves**

First, an inference control curve can be generated by running:

```bash
python inference_create_control_param.py --n_samples length_of_the_control_signal --output_directory control_curve_directory --feature_name name_of_the_control_curve
```

Which will create a `feature_name.npy` file with the control parameters in a `output_directory` directory. The `inference_control_param` notebook shows how to employ that curve as the input of the synthesiser (section V-C of the paper). Keep in mind that if you trained a model with a single user-defined control curve, the directory should contain only one `feature_name.npy` inference control vector.


### Acknowledgements
NoiseBandNet uses code snippets from the following repositories: [ACIDS DDSP implementation](https://github.com/acids-ircam/ddsp_pytorch).