import os
import argparse
from python import models_fp as models

import traceback
import torch
from scipy.io.wavfile import write
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from python.common.text import text_to_sequence, sequence_to_text


def load_and_setup_model(model_name, parser, checkpoint, device, logger, forward_is_infer=False, ema=True, jitable=False):
    model_parser = models.parse_model_args(model_name, parser, add_help=False)
    model_args, model_unk_args = model_parser.parse_known_args()

    model_config = models.get_model_config(model_name, model_args)
    model = models.get_model(model_name, model_config, device, logger, forward_is_infer=forward_is_infer, jitable=jitable)
    model.eval()

    if checkpoint is not None:
        checkpoint_data = torch.load(checkpoint, map_location="cpu")

        if 'state_dict' in checkpoint_data:
            sd = checkpoint_data['state_dict']
            if any(key.startswith('module.') for key in sd):
                sd = {k.replace('module.', ''): v for k,v in sd.items()}

            TEMP_NUM_SPEAKERS = 5
            symbols_embedding_dim = 384
            model.speaker_emb = nn.Embedding(TEMP_NUM_SPEAKERS, symbols_embedding_dim).to(device)
            if "speaker_emb.weight" not in sd:
                sd["speaker_emb.weight"] = torch.rand((TEMP_NUM_SPEAKERS, 384))

            model.load_state_dict(sd, strict=False)
        if 'model' in checkpoint_data:
            model = checkpoint_data['model']
        else:
            model = checkpoint_data
        print(f'Loaded {model_name}')

    if model_name == "WaveGlow":
        model = model.remove_weightnorm(model)
    model.eval()
    model.device = device
    return model.to(device)


def init (PROD, use_gpu, vocoder, logger):
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if use_gpu else 'cpu')
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference', allow_abbrev=False)

    fastpitch = load_and_setup_model('FastPitch', parser, None, device, logger, forward_is_infer=True, jitable=False)
    fastpitch.device = device
    fastpitch.wg_type = None
    fastpitch.ckpt_path = None

    if os.path.exists(f'{"./resources/app" if PROD else "."}/FASTPITCH_LOADING'):
        try:
            os.remove(f'{"./resources/app" if PROD else "."}/FASTPITCH_LOADING')
        except:
            logger.info(traceback.format_exc())
            pass

    return fastpitch

def loadModel (fastpitch, ckpt, n_speakers, device):
    print(f'Loading FastPitch model: {ckpt}')

    checkpoint_data = torch.load(ckpt+".pt", map_location="cpu")
    if 'state_dict' in checkpoint_data:
        checkpoint_data = checkpoint_data['state_dict']

    symbols_embedding_dim = 384
    if n_speakers is not None:
        fastpitch.speaker_emb = nn.Embedding(n_speakers, symbols_embedding_dim).to(device)
    else:
        try:
            del fastpitch.speaker_emb
        except:
            pass
    fastpitch.load_state_dict(checkpoint_data, strict=False)
    fastpitch.ckpt_path = ckpt
    fastpitch = fastpitch.float()

    fastpitch.eval()
    del checkpoint_data
    return fastpitch

def infer_batch(PROD, user_settings, models_manager, linesBatch, fastpitch, vocoder, speaker_i, logger=None, old_sequence=None):
    print(f'Inferring batch of {len(linesBatch)} lines')

    sigma_infer = 0.9
    stft_hop_length = 256
    sampling_rate = 22050
    denoising_strength = 0.01

    text_sequences = []
    for record in linesBatch:
        text = record[0]
        sequence = text_to_sequence(text, ['english_cleaners'])
        text = torch.LongTensor(sequence)
        text_sequences.append(text)
    text_sequences = pad_sequence(text_sequences, batch_first=True).to(fastpitch.device)

    with torch.no_grad():
        pace = torch.tensor([record[3] for record in linesBatch]).unsqueeze(1).to(fastpitch.device)
        pitch_data = None # Maybe in the future
        mel, mel_lens, dur_pred, pitch_pred = fastpitch.infer_advanced(logger, text_sequences, speaker_i=speaker_i, pace=pace, pitch_data=pitch_data, old_sequence=None)

        if "waveglow" in vocoder:

            models_manager.init_model(vocoder)
            audios = models_manager.models[vocoder].model.infer(mel, sigma=sigma_infer)
            audios = models_manager.models[vocoder].denoiser(audios.float(), strength=denoising_strength).squeeze(1)

            for i, audio in enumerate(audios):
                audio = audio[:mel_lens[i].item() * stft_hop_length]
                audio = audio/torch.max(torch.abs(audio))
                output = linesBatch[i][4]
                write(output, sampling_rate, audio.cpu().numpy())
            del audios
        else:
            models_manager.load_model("hifigan", f'{"./resources/app" if PROD else "."}/python/hifigan/hifi.pt' if vocoder=="qnd" else fastpitch.ckpt_path+".hg.pt")

            y_g_hat = models_manager.models["hifigan"].model(mel)
            audios = y_g_hat.view((y_g_hat.shape[0], y_g_hat.shape[2]))
            # audio = audio * 2.3026  # This brings it to the same volume, but makes it clip in places
            for i, audio in enumerate(audios):
                audio = audio[:mel_lens[i].item() * stft_hop_length]
                audio = audio.cpu().numpy()
                audio = audio * 32768.0
                audio = audio.astype('int16')
                output = linesBatch[i][4]
                write(output, sampling_rate, audio)

        del mel, mel_lens

    return ""



def infer(PROD, user_settings, models_manager, text, output, fastpitch, vocoder, speaker_i, pace=1.0, pitch_data=None, logger=None, old_sequence=None):

    print(f'Inferring: "{text}" ({len(text)})')

    sigma_infer = 0.9
    stft_hop_length = 256
    sampling_rate = 22050
    denoising_strength = 0.01

    sequence = text_to_sequence(text, ['english_cleaners'])
    cleaned_text = sequence_to_text(sequence)
    text = torch.LongTensor(sequence)
    text = pad_sequence([text], batch_first=True).to(fastpitch.device)

    with torch.no_grad():

        if old_sequence is not None:
            old_sequence = text_to_sequence(old_sequence, ['english_cleaners'])
            old_sequence = torch.LongTensor(old_sequence)
            old_sequence = pad_sequence([old_sequence], batch_first=True).to(fastpitch.device)

        mel, mel_lens, dur_pred, pitch_pred = fastpitch.infer_advanced(logger, text, speaker_i=speaker_i, pace=pace, pitch_data=pitch_data, old_sequence=old_sequence)

        if "waveglow" in vocoder:

            models_manager.init_model(vocoder)
            audios = models_manager.models[vocoder].model.infer(mel, sigma=sigma_infer)
            audios = models_manager.models[vocoder].denoiser(audios.float(), strength=denoising_strength).squeeze(1)

            for i, audio in enumerate(audios):
                audio = audio[:mel_lens[i].item() * stft_hop_length]
                audio = audio/torch.max(torch.abs(audio))
                write(output, sampling_rate, audio.cpu().numpy())
            del audios
        else:
            models_manager.load_model("hifigan", f'{"./resources/app" if PROD else "."}/python/hifigan/hifi.pt' if vocoder=="qnd" else fastpitch.ckpt_path+".hg.pt")

            y_g_hat = models_manager.models["hifigan"].model(mel)
            audio = y_g_hat.squeeze()
            audio = audio * 32768.0
            # audio = audio * 2.3026  # This brings it to the same volume, but makes it clip in places
            audio = audio.cpu().numpy().astype('int16')
            write(output, sampling_rate, audio)
            del audio

        del mel, mel_lens

    [pitch, durations] = [pitch_pred.cpu().detach().numpy()[0], dur_pred.cpu().detach().numpy()[0]]
    pitch_durations_text = ",".join([str(v) for v in pitch])+"\n"+",".join([str(v) for v in durations])


    del pitch_pred, dur_pred, text, sequence
    return pitch_durations_text +"\n"+cleaned_text
