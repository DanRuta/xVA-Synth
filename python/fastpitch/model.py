import re
import os
import json
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

        self.init_model("english_basic")
        self.isReady = True


    def init_model (self, symbols_alphabet):

        parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference', allow_abbrev=False)
        self.symbols_alphabet = symbols_alphabet

        model_parser = models.parse_model_args("FastPitch", symbols_alphabet, parser, add_help=False)
        model_args, model_unk_args = model_parser.parse_known_args()
        model_config = models.get_model_config("FastPitch", model_args)

        self.model = models.get_model("FastPitch", model_config, self.device, self.logger, forward_is_infer=True, jitable=False)
        self.model.eval()
        self.model.device = self.device

    def load_state_dict (self, ckpt_path, ckpt, n_speakers, base_lang=None):

        self.ckpt_path = ckpt_path

        with open(ckpt_path.replace(".pt", ".json"), "r") as f:
            data = json.load(f)
            if "symbols_alphabet" in data.keys() and data["symbols_alphabet"]!=self.symbols_alphabet:
                self.logger.info(f'Changing symbols_alphabet from {self.symbols_alphabet} to {data["symbols_alphabet"]}')
                self.init_model(data["symbols_alphabet"])

        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        symbols_embedding_dim = 384
        self.model.speaker_emb = nn.Embedding(1 if n_speakers is None else n_speakers, symbols_embedding_dim).to(self.device)

        self.model.load_state_dict(ckpt, strict=False)
        self.model = self.model.float()
        self.model.eval()


    def infer_batch(self, plugin_manager, linesBatch, outputJSON, vocoder, speaker_i, old_sequence=None):
        print(f'Inferring batch of {len(linesBatch)} lines')

        sigma_infer = 0.9
        stft_hop_length = 256
        sampling_rate = 22050
        denoising_strength = 0.01

        text_sequences = []
        cleaned_text_sequences = []
        for record in linesBatch:
            text = record[0]
            text = re.sub(r'[^a-zA-Z\s\(\)\[\]0-9\?\.\,\!\'\{\}]+', '', text)
            sequence = text_to_sequence(text, "english_basic", ['english_cleaners'])
            cleaned_text_sequences.append(sequence_to_text("english_basic", sequence))
            text = torch.LongTensor(sequence)
            text_sequences.append(text)

        text_sequences = pad_sequence(text_sequences, batch_first=True).to(self.device)

        with torch.no_grad():
            pace = torch.tensor([record[3] for record in linesBatch]).unsqueeze(1).to(self.device)
            pitch_amp = torch.tensor([record[7] for record in linesBatch]).unsqueeze(1).to(self.device)
            pitch_data = None # Maybe in the future
            mel, mel_lens, dur_pred, pitch_pred, start_index, end_index = self.model.infer_advanced(self.logger, plugin_manager, cleaned_text_sequences, text_sequences, speaker_i=speaker_i, pace=pace, pitch_data=pitch_data, old_sequence=None, pitch_amp=pitch_amp)

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

            if outputJSON:
                for ri, record in enumerate(linesBatch):
                    # linesBatch: sequence, pitch, duration, pace, tempFileLocation, outPath, outFolder
                    output_fname = linesBatch[ri][5].replace(".wav", ".json")

                    containing_folder = "/".join(output_fname.split("/")[:-1])
                    os.makedirs(containing_folder, exist_ok=True)

                    with open(output_fname, "w+") as f:
                        data = {}
                        data["inputSequence"] = str(linesBatch[ri][0])
                        data["pacing"] = float(linesBatch[ri][3])
                        data["letters"] = [char.replace("{", "").replace("}", "") for char in list(cleaned_text_sequences[ri].split("|"))]
                        data["currentVoice"] = self.ckpt_path.split("/")[-1].replace(".pt", "")
                        data["resetEnergy"] = []
                        data["resetPitch"] = [float(val) for val in list(pitch_pred[ri].cpu().detach().numpy())]
                        data["resetDurs"] = [float(val) for val in list(dur_pred[ri].cpu().detach().numpy())]
                        data["ampFlatCounter"] = 0
                        data["pitchNew"] = data["resetPitch"]
                        data["energyNew"] = data["resetEnergy"]
                        data["dursNew"] = data["resetDurs"]

                        f.write(json.dumps(data, indent=4))

            del mel, mel_lens

        return ""

    def infer(self, plugin_manager, text, output, vocoder, speaker_i, pace=1.0, pitch_data=None, old_sequence=None, globalAmplitudeModifier=None, base_lang=None, base_emb=None, useSR=False):

        self.logger.info(f'Inferring: "{text}" ({len(text)})')

        sigma_infer = 0.9
        stft_hop_length = 256
        sampling_rate = 22050
        denoising_strength = 0.01

        text = re.sub(r'[^a-zA-Z\s\(\)\[\]0-9\?\.\,\!\'\{\}]+', '', text)
        sequence = text_to_sequence(text, "english_basic", ['english_cleaners'])
        cleaned_text = sequence_to_text("english_basic", sequence)
        text = torch.LongTensor(sequence)
        text = pad_sequence([text], batch_first=True).to(self.models_manager.device)

        with torch.no_grad():

            if old_sequence is not None:
                old_sequence = re.sub(r'[^a-zA-Z\s\(\)\[\]0-9\?\.\,\!\'\{\}]+', '', old_sequence)
                old_sequence = text_to_sequence(old_sequence, "english_basic", ['english_cleaners'])
                old_sequence = torch.LongTensor(old_sequence)
                old_sequence = pad_sequence([old_sequence], batch_first=True).to(self.models_manager.device)

            mel, mel_lens, dur_pred, pitch_pred, start_index, end_index = self.model.infer_advanced(self.logger, plugin_manager, [cleaned_text], text, speaker_i=speaker_i, pace=pace, pitch_data=pitch_data, old_sequence=old_sequence)

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
                write(output.replace(".wav", "_preSR.wav") if useSR else output, sampling_rate, audio)

                if useSR:
                    self.models_manager.init_model("nuwave2")
                    self.models_manager.models("nuwave2").sr_audio(output.replace(".wav", "_preSR.wav"), output)
                del audio

            del mel, mel_lens

        [pitch, durations] = [pitch_pred.cpu().detach().numpy()[0], dur_pred.cpu().detach().numpy()[0]]
        pitch_durations_text = ",".join([str(v) for v in pitch])+"\n"+",".join([str(v) for v in durations]) + "\n"


        del pitch_pred, dur_pred, text, sequence
        return pitch_durations_text +"\n"+cleaned_text+"\n" + f'{start_index}\n{end_index}'

    def set_device (self, device):
        self.device = device
        self.model = self.model.to(device)
        self.model.device = device


