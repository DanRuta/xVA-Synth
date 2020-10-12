import os
import argparse
# import models
from python import models
import sys
import warnings


import torch
from scipy.io.wavfile import write
from torch.nn.utils.rnn import pad_sequence

# from common.text import text_to_sequence
from python.common.text import text_to_sequence
# from waveglow import model as glow
from python import model as glow
# from waveglow.denoiser import Denoiser
from python.denoiser import Denoiser
sys.modules['glow'] = glow




def load_and_setup_model(model_name, parser, checkpoint, device, forward_is_infer=False, ema=True, jitable=False):
    model_parser = models.parse_model_args(model_name, parser, add_help=False)
    model_args, model_unk_args = model_parser.parse_known_args()

    model_config = models.get_model_config(model_name, model_args)
    model = models.get_model(model_name, model_config, device, forward_is_infer=forward_is_infer, jitable=jitable)

    if checkpoint is not None:
        checkpoint_data = torch.load(checkpoint)
        status = ''

        if 'state_dict' in checkpoint_data:
            sd = checkpoint_data['state_dict']
            if any(key.startswith('module.') for key in sd):
                sd = {k.replace('module.', ''): v for k,v in sd.items()}

            model.load_state_dict(sd, strict=False)
        else:
            model = checkpoint_data['model']
        print(f'Loaded {model_name}{status}')

    if model_name == "WaveGlow":
        model = model.remove_weightnorm(model)
    model.eval()
    return model.to(device)



def init (use_gpu):
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if use_gpu else 'cpu')
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference', allow_abbrev=False)

    fastpitch = load_and_setup_model('FastPitch', parser, None, device, forward_is_infer=True, jitable=False)
    fastpitch.device = device

    try:
        os.remove("./FASTPITCH_LOADING")
    except:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wg_ckpt_path = "./resources/app/models/waveglow_256channels_universal_v4.pt"
        if not os.path.exists(wg_ckpt_path):
            wg_ckpt_path = "./models/waveglow_256channels_universal_v4.pt"
        waveglow = load_and_setup_model('WaveGlow', parser, wg_ckpt_path, device, forward_is_infer=True)
        denoiser = Denoiser(waveglow).to(device)
        waveglow = getattr(waveglow, 'infer', waveglow)

    fastpitch.waveglow = waveglow
    fastpitch.denoiser = denoiser

    return fastpitch#, waveglow, denoiser


def loadModel (fastpitch, ckpt):
    print(f'Loading FastPitch model: {ckpt}')

    checkpoint_data = torch.load(ckpt+".pt")
    fastpitch.load_state_dict(checkpoint_data['state_dict'], strict=False)

    fastpitch.eval()
    return fastpitch

# def infer(text, output, fastpitch, waveglow, denoiser, pace, pitch_data=None):
def infer(text, output, fastpitch, pace=1.0, pitch_data=None):

    print(f'Inferring: "{text}" ({len(text)})')

    sigma_infer = 0.9
    stft_hop_length = 256
    sampling_rate = 22050
    denoising_strength = 0.01

    text = torch.LongTensor(text_to_sequence(text, ['english_cleaners']))
    text = pad_sequence([text], batch_first=True).to(fastpitch.device)

    with torch.no_grad():

        mel, mel_lens, dur_pred, pitch_pred = fastpitch.infer_advanced(text, pace=pace, pitch_data=pitch_data)

        audios = fastpitch.waveglow(mel, sigma=sigma_infer)
        audios = fastpitch.denoiser(audios.float(), strength=denoising_strength).squeeze(1)

        for i, audio in enumerate(audios):
            audio = audio[:mel_lens[i].item() * stft_hop_length]
            audio = audio/torch.max(torch.abs(audio))
            write(output, sampling_rate, audio.cpu().numpy())

    return [pitch_pred.cpu().detach().numpy()[0], dur_pred.cpu().detach().numpy()[0]]


if __name__ == '__main__':
    use_gpu = True
    # fastpitch, waveglow, denoiser = init(use_gpu=use_gpu, fp_ckpt=f'./output/ralph_cosham/FastPitch_checkpoint_46-9706.pt')
    fastpitch = init(use_gpu=use_gpu, fp_ckpt=f'./output/ralph_cosham/FastPitch_checkpoint_46-9706.pt')
    fastpitch = loadModel(fastpitch, ckpt=f'./output/ralph_cosham/FastPitch_checkpoint_46-9706.pt')

    pitch_data = [
        [ 0.5661585, 0.01267741, 0.05258048, 0.29871154, 0.925132, 1.7379575, 1.0263913, 0.05215776, 0.03624946, 0.07995621, 0.84763604, 0.38124663, 0.01558742, -0.10329278, 0.0082381, 0.15878905, 0.4814809, -0.23024671, -0.2614041, -0.63224566, -0.6544972, -0.28373814, 0.05362733, 0.05719803, 0.02127304, -0.0418622, 0.00267242, -0.2817365, -0.12668462, -0.61234844, -0.2831583, -0.6914924, 0.05280653, 0.0484153],
        [ 8.171971, 1.8040161, 8.369684,2.8598971, 5.514156,7.6847506, 10.210282, 5.580053, 1.6950781, 12.085466, 7.600868, 9.511061, 0.,4.337601, 0.2941885, 9.077057, 6.4160748, 3.1779437, 3.7425022, 6.697532, 6.114112, 3.4995742, 1.7744527, 0.23444939, 0.16238248,  7.3306684, 0.3910483, 5.1543617, 0.77957904, 5.8416853, 14.261931, 8.463928, 3.8534894, 3.29784 ]
    ]
    pitch_data = None

    # [pitch, durations] = infer("A short, sweet sentence, for test.", f'./output/ralph_cosham/audio_devset1.tsv/out_minimal.wav', fastpitch, waveglow, denoiser, pitch_data=pitch_data, pace=1.0)
    [pitch, durations] = infer("A short, sweet sentence, for test.", f'./output/ralph_cosham/audio_devset1.tsv/out_minimal.wav', fastpitch, pitch_data=pitch_data, pace=1.0)
    print(f'pitch, {pitch} {pitch.shape}')
    print(f'durations, {durations} {durations.shape}')
