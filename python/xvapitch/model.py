import os
import re
import json
import codecs
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

import scipy
import scipy.io.wavfile
import librosa
from scipy.io.wavfile import write
import numpy as np

# from python.xvapitch import models
# from python.xvapitch.fastpitch import FastPitch

# from python.common.text import text_to_sequence, sequence_to_text



try:
    import sys
    sys.path.append(".")
    from resources.app.python.xvapitch.text import ALL_SYMBOLS, get_text_preprocessor, lang_names
    from resources.app.python.xvapitch.xvapitch_model import xVAPitch as xVAPitchModel
except:
    try:
        from python.xvapitch.text import ALL_SYMBOLS, get_text_preprocessor, lang_names
        from python.xvapitch.xvapitch_model import xVAPitch as xVAPitchModel
    except:
        try:
            from xvapitch.text import ALL_SYMBOLS, get_text_preprocessor, lang_names
            from xvapitch.xvapitch_model import xVAPitch as xVAPitchModel
        except:
            from text import ALL_SYMBOLS, get_text_preprocessor, lang_names
            from xvapitch_model import xVAPitch as xVAPitchModel
            # try:
            # except:
            #     from text_processing import ALL_SYMBOLS, get_text_preprocessor



class xVAPitch(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(xVAPitch, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.ckpt_path = None

        self.arpabet_dict = {}

        # torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark = False

        self.base_dir = f'{"./resources/app" if self.PROD else "."}/python/xvapitch/text'
        self.lang_tp = {}
        self.lang_tp["en"] = get_text_preprocessor("en", self.base_dir, logger=self.logger)

        self.language_id_mapping = {name: i for i, name in enumerate(sorted(list(lang_names.keys())))}


        self.base_lang = "en"
        self.init_model("english_basic")
        self.isReady = True


    def init_model (self, symbols_alphabet):

        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        # Params from training
        args.pitch = 1
        args.pe_scaling = 0.1
        args.expanded_flow = 0
        args.ow_flow = 0
        args.energy = 0


        self.model = xVAPitchModel(args).to(self.device)
        self.model.eval()
        self.model.device = self.device

    def load_state_dict (self, ckpt_path, ckpt, n_speakers=1, base_lang="en"):

        self.logger.info(f'load_state_dict base_lang: {base_lang}')

        if base_lang not in self.lang_tp.keys():
            self.lang_tp[base_lang] = get_text_preprocessor(base_lang, self.base_dir, logger=self.logger)

        self.base_lang = base_lang
        self.ckpt_path = ckpt_path

        with open(ckpt_path.replace(".pt", ".json"), "r") as f:
            data = json.load(f)

            # TODO -  figure out a good way to handle variants, instead of just 0, the first
            self.base_emb = data["games"][0]["base_speaker_emb"]
            # self.base_emb = torch.tensor(self.base_emb)

            # if "symbols_alphabet" in data.keys() and data["symbols_alphabet"]!=self.symbols_alphabet:
            #     self.logger.info(f'Changing symbols_alphabet from {self.symbols_alphabet} to {data["symbols_alphabet"]}')
            #     self.init_model(data["symbols_alphabet"])

        if 'model' in ckpt:
            ckpt = ckpt['model']

        self.model.load_state_dict(ckpt, strict=False)
        self.model = self.model.float()
        self.model.eval()


    def init_arpabet_dicts (self):
        if len(list(self.arpabet_dict.keys()))==0:
            self.refresh_arpabet_dicts()

    def refresh_arpabet_dicts (self):
        self.arpabet_dict = {}
        json_files = sorted(os.listdir(f'{"./resources/app" if self.PROD else "."}/arpabet'))
        json_files = [fname for fname in json_files if fname.endswith(".json")]

        for fname in json_files:
            with codecs.open(f'{"./resources/app" if self.PROD else "."}/arpabet/{fname}', encoding="utf-8") as f:
                json_data = json.load(f)

                for word in list(json_data["data"].keys()):
                    if json_data["data"][word]["enabled"]==True:
                        self.arpabet_dict[word] = json_data["data"][word]["arpabet"]


    def run_speech_to_speech (self, audiopath, audio_out_path, style_emb, models_manager, plugin_manager, modelType, s2s_components):

        if ".wav" in style_emb:
            style_emb = models_manager.models("speaker_rep").compute_embedding(style_emb).squeeze()
            self.logger.info(f'Getting style emb from: {style_emb}')
        else:
            style_emb = torch.tensor(style_emb).squeeze()
            self.logger.info(f'Getting style emb from json')

        content_emb = models_manager.models("speaker_rep").compute_embedding(audiopath).squeeze()
        style_emb = F.normalize(style_emb.unsqueeze(0), dim=1).unsqueeze(-1).to(self.models_manager.device)
        content_emb = F.normalize(content_emb.unsqueeze(0), dim=1).unsqueeze(-1).to(self.models_manager.device)

        y, sr = librosa.load(audiopath, sr=22050)
        D = librosa.stft(
                    y=y,
                    n_fft=1024,
                    hop_length=256,
                    win_length=1024,
                    pad_mode="reflect",
                    window="hann",
                    center=True,
                )
        spec = np.abs(D).astype(np.float32)
        ref_spectrogram = torch.FloatTensor(spec).unsqueeze(0)

        y_lengths = torch.tensor([ref_spectrogram.size(-1)]).to(self.models_manager.device)

        y = ref_spectrogram.to(self.models_manager.device)
        wav = self.model.voice_conversion(y=y, y_lengths=y_lengths, spk1_emb=content_emb, spk2_emb=style_emb)
        wav = wav.squeeze().cpu().detach().numpy()

        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        scipy.io.wavfile.write(audio_out_path, 22050, wav_norm.astype(np.int16))
        return



    def infer_batch(self, plugin_manager, linesBatch, outputJSON, vocoder, speaker_i, old_sequence=None):
        TODO()
        pass
        # print(f'Inferring batch of {len(linesBatch)} lines')

        # sigma_infer = 0.9
        # stft_hop_length = 256
        # sampling_rate = 22050
        # denoising_strength = 0.01

        # text_sequences = []
        # cleaned_text_sequences = []
        # for record in linesBatch:
        #     text = record[0]
        #     text = re.sub(r'[^a-zA-ZäöüÄÖÜß_\s\(\)\[\]0-9\?\.\,\!\'\{\}]+', '', text)
        #     text = self.infer_arpabet_dict(text, plugin_manager)
        #     text = text.replace("(", "").replace(")", "")
        #     text = re.sub(r'[^a-zA-ZäöüÄÖÜß\s\(\)\[\]0-9\?\.\,\!\'\{\}]+', '', text)
        #     sequence = text_to_sequence(text, "english_basic", ['english_cleaners'])
        #     cleaned_text_sequences.append(sequence_to_text("english_basic", sequence))
        #     text = torch.LongTensor(sequence)
        #     text_sequences.append(text)

        # text_sequences = pad_sequence(text_sequences, batch_first=True).to(self.device)

        # with torch.no_grad():
        #     pace = torch.tensor([record[3] for record in linesBatch]).unsqueeze(1).to(self.device)
        #     pitch_amp = torch.tensor([record[7] for record in linesBatch]).unsqueeze(1).to(self.device)
        #     pitch_data = None # Maybe in the future
        #     mel, mel_lens, dur_pred, pitch_pred, energy_pred, start_index, end_index = self.model.infer_advanced(self.logger, plugin_manager, cleaned_text_sequences, text_sequences, speaker_i=speaker_i, pace=pace, pitch_data=pitch_data, old_sequence=None, pitch_amp=pitch_amp)

        #     if "waveglow" in vocoder:

        #         self.models_manager.init_model(vocoder)
        #         audios = self.models_manager.models(vocoder).model.infer(mel, sigma=sigma_infer)
        #         audios = self.models_manager.models(vocoder).denoiser(audios.float(), strength=denoising_strength).squeeze(1)

        #         for i, audio in enumerate(audios):
        #             audio = audio[:mel_lens[i].item() * stft_hop_length]
        #             audio = audio/torch.max(torch.abs(audio))
        #             output = linesBatch[i][4]
        #             write(output, sampling_rate, audio.cpu().numpy())
        #         del audios
        #     else:
        #         self.models_manager.load_model("hifigan", f'{"./resources/app" if self.PROD else "."}/python/hifigan/hifi.pt' if vocoder=="qnd" else self.ckpt_path.replace(".pt", ".hg.pt"))

        #         y_g_hat = self.models_manager.models("hifigan").model(mel)
        #         audios = y_g_hat.view((y_g_hat.shape[0], y_g_hat.shape[2]))
        #         # audio = audio * 2.3026  # This brings it to the same volume, but makes it clip in places
        #         for i, audio in enumerate(audios):
        #             audio = audio[:mel_lens[i].item() * stft_hop_length]
        #             audio = audio.cpu().numpy()
        #             audio = audio * 32768.0
        #             audio = audio.astype('int16')
        #             output = linesBatch[i][4]
        #             write(output, sampling_rate, audio)

        #     if outputJSON:
        #         for ri, record in enumerate(linesBatch):
        #             # linesBatch: sequence, pitch, duration, pace, tempFileLocation, outPath, outFolder
        #             output_fname = linesBatch[ri][5].replace(".wav", ".json")

        #             containing_folder = "/".join(output_fname.split("/")[:-1])
        #             os.makedirs(containing_folder, exist_ok=True)

        #             with open(output_fname, "w+") as f:
        #                 data = {}
        #                 data["inputSequence"] = str(linesBatch[ri][0])
        #                 data["pacing"] = float(linesBatch[ri][3])
        #                 data["letters"] = [char.replace("{", "").replace("}", "") for char in list(cleaned_text_sequences[ri].split("|"))]
        #                 data["currentVoice"] = self.ckpt_path.split("/")[-1].replace(".pt", "")
        #                 data["resetEnergy"] = [float(val) for val in list(energy_pred[ri].cpu().detach().numpy())]
        #                 data["resetPitch"] = [float(val) for val in list(pitch_pred[ri][0].cpu().detach().numpy())]
        #                 data["resetDurs"] = [float(val) for val in list(dur_pred[ri].cpu().detach().numpy())]
        #                 data["ampFlatCounter"] = 0
        #                 data["pitchNew"] = data["resetPitch"]
        #                 data["energyNew"] = data["resetEnergy"]
        #                 data["dursNew"] = data["resetDurs"]

        #                 f.write(json.dumps(data, indent=4))


        #     del mel, mel_lens

        # return ""

    def infer(self, plugin_manager, text, out_path, vocoder, speaker_i, pace=1.0, pitch_data=None, old_sequence=None, globalAmplitudeModifier=None, base_lang="en"):

        if base_lang not in self.lang_tp.keys():
            self.lang_tp[base_lang] = get_text_preprocessor(base_lang, self.base_dir, logger=self.logger)


        # sigma_infer = 0.9
        # stft_hop_length = 256
        sampling_rate = 22050
        # denoising_strength = 0.01

        # =============

        # sequence, cleaned_text = self.tp.text_to_sequence(text)
        sequence, cleaned_text = self.lang_tp[base_lang].text_to_sequence(text)
        # =============





        # text = re.sub(r'[^a-zA-ZäöüÄÖÜß_\s\(\)\[\]0-9\?\.\,\!\'\{\}]+', '', text)
        # text = self.infer_arpabet_dict(text, plugin_manager)
        # text = text.replace("(", "").replace(")", "")
        # text = re.sub(r'[^a-zA-ZäöüÄÖÜß\s\(\)\[\]0-9\?\.\,\!\'\{\}]+', '', text)
        # sequence = text_to_sequence(text, "english_basic", ['english_cleaners'])
        # cleaned_text = sequence_to_text("english_basic", sequence)

        text = torch.LongTensor(sequence)
        text = pad_sequence([text], batch_first=True).to(self.models_manager.device)

        with torch.no_grad():

            if old_sequence is not None:
                old_sequence = re.sub(r'[^a-zA-Z\s\(\)\[\]0-9\?\.\,\!\'\{\}]+', '', old_sequence)
                self.logger.info(f'old_sequence: {old_sequence}')
                # old_sequence = text_to_sequence(old_sequence, "english_basic", ['english_cleaners'])
                old_sequence, clean_old_sequence = self.lang_tp[base_lang].text_to_sequence(old_sequence)#, "english_basic", ['english_cleaners'])
                old_sequence = torch.LongTensor(old_sequence)
                old_sequence = pad_sequence([old_sequence], batch_first=True).to(self.models_manager.device)


            # ==== TODO, make editable
            language_id = self.language_id_mapping[base_lang]
            lang_ids = [language_id for _ in range(text.shape[1])]
            lang_ids = [language_id] # TODO, add per-symbol support
            lang_ids = torch.tensor(lang_ids).to(self.models_manager.device)
            # ====

            # ==== TODO, make editable
            # self.logger.info(f'self.base_emb: {self.base_emb}')
            speaker_embs = [F.normalize(torch.tensor(self.base_emb).unsqueeze(dim=0))[0].unsqueeze(-1) for _ in range(text.shape[1])]
            speaker_embs = [F.normalize(torch.tensor(self.base_emb).unsqueeze(dim=0))[0].unsqueeze(-1)] # TODO, add per-symbol support
            self.logger.info(f'speaker_embs: {len(speaker_embs)}, {len(speaker_embs[0])}')
            speaker_embs = torch.stack(speaker_embs, dim=0).to(self.models_manager.device)#.unsqueeze(-1)

            # g = F.normalize(speaker_embs[0])
            # if g.ndim == 2:
            #     g = g.unsqueeze_(0)


            # ====
            self.logger.info(f'speaker_embs: {speaker_embs.shape}')
            self.logger.info(f'lang_ids: {lang_ids.shape} {lang_ids}')

            lang_embs = lang_ids # TODO, use pre-extracted trained language embeddings, for interpolation



            editor_data = pitch_data # TODO, propagate rename
            output_wav, dur_pred, pitch_pred, energy_pred, start_index, end_index = self.model.infer_advanced(self.logger, plugin_manager, [cleaned_text], text, lang_embs=lang_embs, speaker_embs=speaker_embs, pace=pace, editor_data=editor_data, old_sequence=old_sequence)
            wav = output_wav.squeeze().cpu().detach().numpy()
            wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
            scipy.io.wavfile.write(out_path, sampling_rate, wav_norm.astype(np.int16))

            # if "waveglow" in vocoder:

            #     self.models_manager.init_model(vocoder)
            #     audios = self.models_manager.models(vocoder).model.infer(mel, sigma=sigma_infer)
            #     audios = self.models_manager.models(vocoder).denoiser(audios.float(), strength=denoising_strength).squeeze(1)

            #     for i, audio in enumerate(audios):
            #         audio = audio[:mel_lens[i].item() * stft_hop_length]
            #         audio = audio/torch.max(torch.abs(audio))
            #         write(output, sampling_rate, audio.cpu().numpy())
            #     del audios
            # else:
            #     self.models_manager.load_model("hifigan", f'{"./resources/app" if self.PROD else "."}/python/hifigan/hifi.pt' if vocoder=="qnd" else self.ckpt_path.replace(".pt", ".hg.pt"))

            #     y_g_hat = self.models_manager.models("hifigan").model(mel)
            #     audio = y_g_hat.squeeze()
            #     audio = audio * 32768.0
            #     # audio = audio * 2.3026  # This brings it to the same volume, but makes it clip in places
            #     audio = audio.cpu().numpy().astype('int16')
            #     write(output, sampling_rate, audio)
                # del audio

            # del mel, mel_lens

        [pitch, durations, energy] = [pitch_pred.squeeze().cpu().detach().numpy(), dur_pred.squeeze().cpu().detach().numpy(), energy_pred.cpu().detach().numpy() if energy_pred is not None else []]
        # self.logger.info(f'pitch: {pitch}')
        # self.logger.info(f'durations: {durations}')
        # self.logger.info(f'energy: {energy}')
        pitch_durations_energy_text = ",".join([str(v) for v in pitch]) + "\n" + ",".join([str(v) for v in durations]) + "\n" + ",".join([str(v) for v in energy])

        del pitch_pred, dur_pred, energy_pred, text, sequence
        return pitch_durations_energy_text +"\n"+cleaned_text +"\n"+ f'{start_index}\n{end_index}'

    def set_device (self, device):
        self.device = device
        self.model = self.model.to(device)
        self.model.enc = self.model.enc.to(device)
        self.model.post = self.model.post.to(device)
        self.model.device = device


    # def run_speech_to_speech (self, audiopath, models_manager, plugin_manager, modelType, s2s_components, text):
    #     return self.model.run_speech_to_speech(self.device, self.logger, models_manager, plugin_manager, modelType, s2s_components, audiopath, text, text_to_sequence, sequence_to_text, self)


