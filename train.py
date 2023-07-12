from noisebandnet.model import NoiseBandNet
import dataset_tool
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import auraloss
import soundfile as sf
import os
import argparse
from tqdm import tqdm
import datetime
import dateutil.tz
import pickle

def create_save_dir(dataset_path):
    #create save dir with date and model name
    dataset_path = os.path.basename(os.path.normpath(dataset_path))
    current_time = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = f'trained_models/{dataset_path}/{current_time}'
    if os.path.exists(output_dir) == False:
        print(f'Creating saving directory in {output_dir}')
        os.makedirs(output_dir)
    else:
        print(f'Saving model in: {output_dir}')
    return output_dir

def save_config(config, save_dir):
    print(f'Saving model config in {save_dir}')
    with open(f'{save_dir}/config.pickle', 'wb') as handle:
        pickle.dump(config, handle)
    
def save_model(epoch, save_dir, model, x_audio, y_audio, sampling_rate):
    torch.save(obj=model.state_dict(), f=f'{save_dir}/model_{epoch}.ckpt')
    sf.write(file=f'{save_dir}/y_audio_epoch_{epoch}.wav', data=y_audio[0].squeeze(0).detach().cpu().numpy(), samplerate=sampling_rate)
    sf.write(file=f'{save_dir}/x_audio_epoch_{epoch}.wav', data=x_audio[0].squeeze(0).detach().cpu().numpy(), samplerate=sampling_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', help='Directory of the training sound/sounds', default='metal')
    parser.add_argument('--auto_control_params', nargs='+', help='Automatic control parameters', default='loudness', choices=['loudness', 'centroid'])
    parser.add_argument('--control_params_path', help='Directory of the training sound control parameter', default=None)
    parser.add_argument('--device', help='Device to use', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--epochs', type=int, default=10000, help='How many epochs to train on')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_band', type=int, default=2048, help='Number of bands of the filter bank')
    parser.add_argument('--synth_window', type=int, default=32, help='How many samples to get a new amplitude value')
    parser.add_argument('--sampling_rate', type=int, default=44100, help='Sampling rate of the model')
    parser.add_argument('--audio_size_samples', type=int, default=65536, help='Number of samples of the training audio chunks')
    parser.add_argument('--hidden_size', type=int, default=128, help='Model hidden size')
    parser.add_argument('--print_loss_step', type=int, default=100, help='How often print the loss in fraction of epochs (epochs/print_loss)')
    parser.add_argument('--save_model_step', type=int, default=10, help='How often save the model and output training audio in fraction of epochs (epochs/save_model)')
    
    config = parser.parse_args()
    save_dir = create_save_dir(config.dataset_path)
    
    audio_dataset = dataset_tool.AudioDataset(dataset_path=config.dataset_path, audio_size_samples=config.audio_size_samples, min_batch_size=config.batch_size,
        sampling_rate=config.sampling_rate, device=config.device, auto_control_params=config.auto_control_params, control_params_path=config.control_params_path)
    dataloder = DataLoader(dataset=audio_dataset, batch_size=config.batch_size, shuffle=True)
    
    n_control_params = len(audio_dataset.control_params)
    config.n_control_params = n_control_params
    save_config(config, save_dir)

    synth = NoiseBandNet(hidden_size=config.hidden_size, n_band=config.n_band, synth_window=config.synth_window, n_control_params=config.n_control_params).to(config.device).float()
    print(f'Model parameters: {sum(p.numel() for p in synth.parameters() if p.requires_grad)}')

    print_loss = config.epochs//config.print_loss_step
    model_snapshot = config.epochs//(config.save_model_step)
    

    opt = torch.optim.Adam(synth.parameters(), lr=config.lr)
    mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[8192, 4096, 1024, 2048, 512, 128, 32],
                                                hop_sizes=[8192//4, 4096//4, 1024//4, 2048//4, 512//4, 128//4, 32//4],
                                                win_lengths=[8192, 4096, 1024, 2048, 512, 128, 32])
    
    
    print(f'Starting training for {config.epochs} epochs.')
    print(f'Saving model every {model_snapshot} epochs.')
    for epoch in tqdm(range(config.epochs)):
        for x_audio, control_params in dataloder:
            opt.zero_grad()
            for i in range(len(control_params)):
                #downsample to synth window
                control_params[i] = F.interpolate(input=control_params[i], scale_factor=1/config.synth_window, mode='linear').permute(0,2,1).float()
            y_audio = synth(control_params)
            loss = mrstft(y_audio, x_audio)
            loss.backward()
            opt.step()
        if epoch%print_loss==0:
            print(f'\nLoss: {loss.item()}')
        if epoch%model_snapshot==0:
            print(f'Taking model snapshot at epoch {epoch}')
            save_model(epoch=epoch, save_dir=save_dir, model=synth, x_audio=x_audio, y_audio=y_audio, sampling_rate=config.sampling_rate)
    print(f'Taking model snapshot at end of traning')
    save_model(epoch=epoch, save_dir=save_dir, model=synth, x_audio=x_audio, y_audio=y_audio, sampling_rate=config.sampling_rate)   
