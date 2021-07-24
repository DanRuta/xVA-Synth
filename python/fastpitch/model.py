import argparse

import torch
import torch.nn as nn
from python.fastpitch import models

from scipy.io.wavfile import write
from torch.nn.utils.rnn import pad_sequence
from python.common.text import text_to_sequence, sequence_to_text

class FastPitch(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(FastPitch, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.ckpt_path = None

        torch.backends.cudnn.benchmark = True
        parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference', allow_abbrev=False)

        model_parser = models.parse_model_args("FastPitch", parser, add_help=False)
        model_args, model_unk_args = model_parser.parse_known_args()
        model_config = models.get_model_config("FastPitch", model_args)

        self.model = models.get_model("FastPitch", model_config, self.device, self.logger, forward_is_infer=True, jitable=False)
        self.model.eval()
        self.model.device = self.device

        self.isReady = True


    def load_state_dict (self, ckpt_path, ckpt, n_speakers):

        self.ckpt_path = ckpt_path

        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        symbols_embedding_dim = 384
        self.model.speaker_emb = nn.Embedding(n_speakers, symbols_embedding_dim).to(self.device)

        self.model.load_state_dict(ckpt, strict=False)
        self.model = self.model.float()
        self.model.eval()


    def infer_batch(self, user_settings, linesBatch, vocoder, speaker_i, old_sequence=None):
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
        text_sequences = pad_sequence(text_sequences, batch_first=True).to(self.device)

        with torch.no_grad():
            pace = torch.tensor([record[3] for record in linesBatch]).unsqueeze(1).to(self.device)
            pitch_data = None # Maybe in the future
            mel, mel_lens, dur_pred, pitch_pred = self.model.infer_advanced(self.logger, text_sequences, speaker_i=speaker_i, pace=pace, pitch_data=pitch_data, old_sequence=None)

            if "waveglow" in vocoder:

                self.models_manager.init_model(vocoder)
                audios = self.models_manager.models(vocoder).model.infer(mel, sigma=sigma_infer)
                audios = self.models_manager.models(vocoder).denoiser(audios.float(), strength=denoising_strength).squeeze(1)

                for i, audio in enumerate(audios):
                    audio = audio[:mel_lens[i].item() * stft_hop_length]
                    audio = audio/torch.max(torch.abs(audio))
                    output = linesBatch[i][4]
                    write(output, sampling_rate, audio.cpu().numpy())
                del audios
            else:
                self.models_manager.load_model("hifigan", f'{"./resources/app" if self.PROD else "."}/python/hifigan/hifi.pt' if vocoder=="qnd" else self.ckpt_path.replace(".pt", ".hg.pt"))

                y_g_hat = self.models_manager.models("hifigan").model(mel)
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

    def infer(self, user_settings, text, output, vocoder, speaker_i, pace=1.0, pitch_data=None, old_sequence=None):

        print(f'Inferring: "{text}" ({len(text)})')

        sigma_infer = 0.9
        stft_hop_length = 256
        sampling_rate = 22050
        denoising_strength = 0.01

        sequence = text_to_sequence(text, ['english_cleaners'])
        cleaned_text = sequence_to_text(sequence)
        text = torch.LongTensor(sequence)
        text = pad_sequence([text], batch_first=True).to(self.models_manager.device)

        with torch.no_grad():

            if old_sequence is not None:
                old_sequence = text_to_sequence(old_sequence, ['english_cleaners'])
                old_sequence = torch.LongTensor(old_sequence)
                old_sequence = pad_sequence([old_sequence], batch_first=True).to(self.models_manager.device)

            mel, mel_lens, dur_pred, pitch_pred = self.model.infer_advanced(self.logger, text, speaker_i=speaker_i, pace=pace, pitch_data=pitch_data, old_sequence=old_sequence)

            if "waveglow" in vocoder:

                self.models_manager.init_model(vocoder)
                audios = self.models_manager.models(vocoder).model.infer(mel, sigma=sigma_infer)
                audios = self.models_manager.models(vocoder).denoiser(audios.float(), strength=denoising_strength).squeeze(1)

                for i, audio in enumerate(audios):
                    audio = audio[:mel_lens[i].item() * stft_hop_length]
                    audio = audio/torch.max(torch.abs(audio))
                    write(output, sampling_rate, audio.cpu().numpy())
                del audios
            else:
                self.models_manager.load_model("hifigan", f'{"./resources/app" if self.PROD else "."}/python/hifigan/hifi.pt' if vocoder=="qnd" else self.ckpt_path.replace(".pt", ".hg.pt"))

                y_g_hat = self.models_manager.models("hifigan").model(mel)
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

    def set_device (self, device):
        self.device = device
        self.model = self.model.to(device)
        self.model.device = device


