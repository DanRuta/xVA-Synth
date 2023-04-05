import os
import re
import json
import codecs
import ffmpeg
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


        self.pitch_emb_values = torch.tensor(np.load(f'{"./resources/app" if self.PROD else "."}/python/xvapitch/embs/pitch_emb.npy')).unsqueeze(0).unsqueeze(-1)
        self.angry_emb_values = torch.tensor(np.load(f'{"./resources/app" if self.PROD else "."}/python/xvapitch/embs/angry.npy')).unsqueeze(0).unsqueeze(-1)
        self.happy_emb_values = torch.tensor(np.load(f'{"./resources/app" if self.PROD else "."}/python/xvapitch/embs/happy.npy')).unsqueeze(0).unsqueeze(-1)
        self.sad_emb_values = torch.tensor(np.load(f'{"./resources/app" if self.PROD else "."}/python/xvapitch/embs/sad.npy')).unsqueeze(0).unsqueeze(-1)
        self.surprise_emb_values = torch.tensor(np.load(f'{"./resources/app" if self.PROD else "."}/python/xvapitch/embs/surprise.npy')).unsqueeze(0).unsqueeze(-1)

        self.base_lang = "en"
        self.init_model()
        self.model.pitch_emb_values = self.pitch_emb_values.to(self.models_manager.device)
        self.model.angry_emb_values = self.angry_emb_values.to(self.models_manager.device)
        self.model.happy_emb_values = self.happy_emb_values.to(self.models_manager.device)
        self.model.sad_emb_values = self.sad_emb_values.to(self.models_manager.device)
        self.model.surprise_emb_values = self.surprise_emb_values.to(self.models_manager.device)
        self.isReady = True


    def init_model (self):

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

        if os.path.exists(ckpt_path.replace(".pt", ".json")):
            with open(ckpt_path.replace(".pt", ".json"), "r") as f:
                data = json.load(f)
                self.base_emb = data["games"][0]["base_speaker_emb"]

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


    def run_speech_to_speech (self, audiopath, audio_out_path, style_emb, models_manager, plugin_manager, useSR=False, useCleanup=False):

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
        if useCleanup:
            ffmpeg_path = f'{"./resources/app" if self.PROD else "."}/python/ffmpeg.exe'

            if useSR:
                scipy.io.wavfile.write(audio_out_path.replace(".wav", "_preSR.wav"), 22050, wav_norm.astype(np.int16))
            else:
                scipy.io.wavfile.write(audio_out_path.replace(".wav", "_preCleanupPreFFmpeg.wav"), 22050, wav_norm.astype(np.int16))
                stream = ffmpeg.input(audio_out_path.replace(".wav", "_preCleanupPreFFmpeg.wav"))
                ffmpeg_options = {"ar": 48000}
                output_path = audio_out_path.replace(".wav", "_preCleanup.wav")
                stream = ffmpeg.output(stream, output_path, **ffmpeg_options)
                out, err = (ffmpeg.run(stream, cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True, overwrite_output=True))
                os.remove(audio_out_path.replace(".wav", "_preCleanupPreFFmpeg.wav"))
        else:
            scipy.io.wavfile.write(audio_out_path.replace(".wav", "_preSR.wav") if useSR else audio_out_path, 22050, wav_norm.astype(np.int16))

        if useSR:
            self.models_manager.init_model("nuwave2")
            self.models_manager.models("nuwave2").sr_audio(audio_out_path.replace(".wav", "_preSR.wav"), audio_out_path.replace(".wav", "_preCleanup.wav") if useCleanup else audio_out_path)

        if useCleanup:
            self.models_manager.init_model("deepfilternet2")
            self.models_manager.models("deepfilternet2").cleanup_audio(audio_out_path.replace(".wav", "_preCleanup.wav"), audio_out_path)

        return



    def infer_batch(self, plugin_manager, linesBatch, outputJSON, vocoder, speaker_i, old_sequence=None, useSR=False, useCleanup=False):
        print(f'Inferring batch of {len(linesBatch)} lines')

        text_sequences = []
        cleaned_text_sequences = []
        lang_embs = []
        speaker_embs = []
        # [sequence, pitch, duration, pace, tempFileLocation, outPath, outFolder, pitch_amp, base_lang, base_emb, vc_content, vc_style]
        vc_input = []
        tts_input = []
        for ri,record in enumerate(linesBatch):
            if record[-2]: # If a VC content file has been given, handle this as VC
                vc_input.append(record)
            else:
                tts_input.append(record)

        # =================
        # ======= Handle VC
        # =================
        if len(vc_input):
            for ri,record in enumerate(vc_input):
                content_emb = self.models_manager.models("speaker_rep").compute_embedding(record[-2]).squeeze()
                style_emb = self.models_manager.models("speaker_rep").compute_embedding(record[-1]).squeeze()
                # content_emb = F.normalize(content_emb.unsqueeze(0), dim=1).squeeze(0)
                # style_emb = F.normalize(style_emb.unsqueeze(0), dim=1).squeeze(0)
                content_emb = content_emb.unsqueeze(0).unsqueeze(-1).to(self.models_manager.device)
                style_emb = style_emb.unsqueeze(0).unsqueeze(-1).to(self.models_manager.device)

                y, sr = librosa.load(record[-2], sr=22050)
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

                # Run Voice Conversion
                self.model.logger = self.logger
                wav = self.model.voice_conversion(y=y, y_lengths=y_lengths, spk1_emb=content_emb, spk2_emb=style_emb)
                wav = wav.squeeze().cpu().detach().numpy()
                wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

                if useCleanup:
                    ffmpeg_path = f'{"./resources/app" if self.PROD else "."}/python/ffmpeg.exe'

                    if useSR:
                        scipy.io.wavfile.write(tts_input[ri][4].replace(".wav", "_preSR.wav"), 22050, wav_norm.astype(np.int16))
                    else:
                        scipy.io.wavfile.write(tts_input[ri][4].replace(".wav", "_preCleanupPreFFmpeg.wav"), 22050, wav_norm.astype(np.int16))
                        stream = ffmpeg.input(tts_input[ri][4].replace(".wav", "_preCleanupPreFFmpeg.wav"))
                        ffmpeg_options = {"ar": 48000}
                        output_path = tts_input[ri][4].replace(".wav", "_preCleanup.wav")
                        stream = ffmpeg.output(stream, output_path, **ffmpeg_options)
                        out, err = (ffmpeg.run(stream, cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True, overwrite_output=True))
                        os.remove(tts_input[ri][4].replace(".wav", "_preCleanupPreFFmpeg.wav"))
                else:
                    scipy.io.wavfile.write(vc_input[ri][4].replace(".wav", "_preSR.wav") if useSR else vc_input[ri][4], 22050, wav_norm.astype(np.int16))

                if useSR:
                    self.models_manager.init_model("nuwave2")
                    self.models_manager.models("nuwave2").sr_audio(vc_input[ri][4].replace(".wav", "_preSR.wav"), vc_input[ri][4].replace(".wav", "_preCleanup.wav") if useCleanup else vc_input[ri][4])
                    os.remove(vc_input[ri][4].replace(".wav", "_preSR.wav"))

                if useCleanup:
                    self.models_manager.init_model("deepfilternet2")
                    self.models_manager.models("deepfilternet2").cleanup_audio(vc_input[ri][4].replace(".wav", "_preCleanup.wav"), vc_input[ri][4])
                    os.remove(vc_input[ri][4].replace(".wav", "_preCleanup.wav"))



        # ==================
        # ======= Handle TTS
        # ==================
        if len(tts_input):
            for ri,record in enumerate(tts_input):
                # Language set-up
                base_lang = record[-4]
                if base_lang not in self.lang_tp.keys():
                    self.lang_tp[base_lang] = get_text_preprocessor(base_lang, self.base_dir, logger=self.logger)

                # Pre-process text
                text = record[0]
                sequence, cleaned_text = self.lang_tp[base_lang].text_to_sequence(text)
                cleaned_text_sequences.append(cleaned_text)
                text = torch.LongTensor(sequence)
                text_sequences.append(text)

                # Set the language ID per-symbol
                # TODO, add per-symbol support
                # lang_embs.append(torch.tensor([self.language_id_mapping[base_lang] for _ in range(text.shape[0])]))
                lang_embs.append(torch.tensor(self.language_id_mapping[base_lang]))

                # Set the speaker embedding per-symbol
                # TODO, add per-symbol support
                # speaker_embs.append(torch.stack([torch.tensor(tts_input[ri][-1]).unsqueeze(dim=0)[0].unsqueeze(-1) for _ in range(text.shape[0])], dim=0))
                # speaker_embs.append(torch.stack([torch.tensor(tts_input[ri][-1]).unsqueeze(-1)], dim=0))
                speaker_embs.append(torch.tensor(tts_input[ri][-3]).unsqueeze(-1))

            lang_embs = torch.stack(lang_embs, dim=0).to(self.models_manager.device)

            text_sequences = pad_sequence(text_sequences, batch_first=True).to(self.models_manager.device)
            speaker_embs = pad_sequence(speaker_embs, batch_first=True).to(self.models_manager.device)


            pace = torch.tensor([record[3] for record in tts_input]).unsqueeze(1).to(self.device)
            pitch_amp = torch.tensor([record[7] for record in tts_input]).unsqueeze(1).to(self.device)


            # Could pass indexes (and get them returned) to the tts inference fn
            # Do the same to the vc infer fn
            # Then marge them into their place in an output array?

            out = self.model.infer_advanced(self.logger, plugin_manager, [cleaned_text_sequences], text_sequences, lang_embs=lang_embs, speaker_embs=speaker_embs, pace=pace, old_sequence=None, pitch_amp=pitch_amp)
            if isinstance(out, str):
                return out
            else:
                output_wav, dur_pred, pitch_pred, energy_pred, _, _, _ = out

                for i,wav in enumerate(output_wav):
                    wav = wav.squeeze().cpu().detach().numpy()
                    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
                    if useCleanup:
                        ffmpeg_path = f'{"./resources/app" if self.PROD else "."}/python/ffmpeg.exe'

                        if useSR:
                            scipy.io.wavfile.write(tts_input[i][4].replace(".wav", "_preSR.wav"), 22050, wav_norm.astype(np.int16))
                        else:
                            scipy.io.wavfile.write(tts_input[i][4].replace(".wav", "_preCleanupPreFFmpeg.wav"), 22050, wav_norm.astype(np.int16))
                            stream = ffmpeg.input(tts_input[i][4].replace(".wav", "_preCleanupPreFFmpeg.wav"))
                            ffmpeg_options = {"ar": 48000}
                            output_path = tts_input[i][4].replace(".wav", "_preCleanup.wav")
                            stream = ffmpeg.output(stream, output_path, **ffmpeg_options)
                            out, err = (ffmpeg.run(stream, cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True, overwrite_output=True))
                            os.remove(tts_input[i][4].replace(".wav", "_preCleanupPreFFmpeg.wav"))
                    else:
                        scipy.io.wavfile.write(tts_input[i][4].replace(".wav", "_preSR.wav") if useSR else tts_input[i][4], 22050, wav_norm.astype(np.int16))

                    if useSR:
                        self.models_manager.init_model("nuwave2")
                        self.models_manager.models("nuwave2").sr_audio(tts_input[i][4].replace(".wav", "_preSR.wav"), tts_input[i][4].replace(".wav", "_preCleanup.wav") if useCleanup else tts_input[i][4])
                        os.remove(tts_input[i][4].replace(".wav", "_preSR.wav"))

                    if useCleanup:
                        self.models_manager.init_model("deepfilternet2")
                        self.models_manager.models("deepfilternet2").cleanup_audio(tts_input[i][4].replace(".wav", "_preCleanup.wav"), tts_input[i][4])
                        os.remove(tts_input[i][4].replace(".wav", "_preCleanup.wav"))

                if outputJSON:
                    for ri, record in enumerate(tts_input):
                        # tts_input: sequence, pitch, duration, pace, tempFileLocation, outPath, outFolder
                        output_fname = tts_input[ri][5].replace(".wav", ".json")

                        containing_folder = "/".join(output_fname.split("/")[:-1])
                        os.makedirs(containing_folder, exist_ok=True)

                        with open(output_fname, "w+") as f:
                            data = {}
                            data["inputSequence"] = str(tts_input[ri][0])
                            data["pacing"] = float(tts_input[ri][3])
                            data["letters"] = [char.replace("{", "").replace("}", "") for char in list(cleaned_text_sequences[ri].split("|"))]
                            data["currentVoice"] = self.ckpt_path.split("/")[-1].replace(".pt", "")
                            data["resetEnergy"] = [float(val) for val in list(energy_pred[ri].cpu().detach().numpy())]
                            data["resetPitch"] = [float(val) for val in list(pitch_pred[ri][0].cpu().detach().numpy())]
                            data["resetDurs"] = [float(val) for val in list(dur_pred[ri].cpu().detach().numpy())]
                            data["ampFlatCounter"] = 0
                            data["pitchNew"] = data["resetPitch"]
                            data["energyNew"] = data["resetEnergy"]
                            data["dursNew"] = data["resetDurs"]

                            f.write(json.dumps(data, indent=4))



        return ""



    # Split words by space, while also breaking out the \land[code][text] formatting
    def splitWords (self, sequence, addSpace=False):
        words = []
        for word in sequence:
            if word.startswith("\\lang["):
                words.append(word.split("][")[0]+"][")
                word = word.split("][")[1]
            for char in ["}","]","[","{"]:
                if word.startswith(char):
                    words.append(char)
                    word = word[1:]
            end_extras = []
            for char in ["}","]","[","{"]:
                if word.endswith(char):
                    end_extras.append(char)
                    word = word[:-1]

            words.append(word)
            end_extras.reverse()
            for extra in end_extras:
                words.append(extra)

            if addSpace:
                words.append(" ")
        return words

    def preprocess_prompt_language (self, sequence, base_lang):

        # Prepare the input sequence for processing. Do a few times to catch edge cases
        sequence = self.splitWords(sequence.split(" "), True)
        sequence = self.splitWords(sequence)
        sequence = self.splitWords(sequence)
        sequence = self.splitWords(sequence)

        subSequences = []

        openedLangs = 0
        langs_stack = [base_lang]
        for word in sequence:
            skip_word = False
            if word.startswith("\\lang["):
                openedLangs += 1
                langs_stack.append(word.split("lang[")[1].split("]")[0])
                skip_word = True

            if word.endswith("]"):
                openedLangs -= 1
                langs_stack.pop()
                skip_word = True

            # Add the word to the list if not skipping it, if it's not empty, or it's not a second space in a row
            if not skip_word and len(word) and (word!=" " or len(subSequences)==0 or subSequences[-1][list(subSequences[-1].keys())[0]]!=" "):
                subSequences.append({langs_stack[-1]: word})


        subSequences_collapsed = []
        current_open_arpabet = []
        last_lang = None
        is_in_arpabet = False
        # Collapse groups of inlined ARPABet symbols, to have them treated as such
        for subSequence in subSequences:
            ss_lang = list(subSequence.keys())[0]
            ss_val = subSequence[ss_lang]
            if ss_lang is not last_lang:
                if len(current_open_arpabet):
                    subSequences_collapsed.append({ss_lang: "{"+" ".join(current_open_arpabet).replace("  "," ")+"}"})
                    current_open_arpabet = []
                last_lang = ss_lang

            if ss_val.strip()=="{":
                is_in_arpabet = True
            elif ss_val.strip()=="}":
                subSequences_collapsed.append({ss_lang: "{"+" ".join(current_open_arpabet).replace("  "," ")+"}"})
                current_open_arpabet = []
                is_in_arpabet = False
            else:
                if is_in_arpabet:
                    current_open_arpabet.append(ss_val)
                else:
                    subSequences_collapsed.append({ss_lang: ss_val})

        return subSequences_collapsed


    def getG2P (self, text, base_lang):

        sequenceSplitByLanguage = self.preprocess_prompt_language(text, base_lang)

        # Make sure all languages' text processors are initialized
        for subSequence in sequenceSplitByLanguage:
            langCode = list(subSequence.keys())[0]
            if langCode not in self.lang_tp.keys():
                self.lang_tp[langCode] = get_text_preprocessor(langCode, self.base_dir, logger=self.logger)

        returnString = "{"
        langs_stack = [base_lang]

        last_lang = base_lang
        for subSequence in sequenceSplitByLanguage:
            langCode = list(subSequence.keys())[0]
            subSeq = subSequence[langCode]
            sequence, cleaned_text = self.lang_tp[langCode].text_to_sequence(subSeq)

            if langCode != last_lang:
                last_lang = langCode

                if len(langs_stack)>1 and langs_stack[-2]==langCode:
                    langs_stack.pop()
                    if returnString[-1]=="}":
                        returnString = returnString[:-1]
                    returnString += "]}"
                else:
                    langs_stack.append(langCode)

                    if returnString[-1]=="{":
                        returnString = returnString[:-1]
                    returnString += f'\\lang[{langCode}][' + "{"

            returnString += " ".join([symb for symb in cleaned_text.split("|") if symb != "<PAD>"]).replace("_", "} {")


        if returnString[-1]=="{":
            returnString = returnString[:-1]
        else:
            returnString = returnString+"}"


        returnString = returnString.replace(".}", "}.")
        returnString = returnString.replace(",}", "},")
        returnString = returnString.replace("!}", "}!")
        returnString = returnString.replace("?}", "}?")
        returnString = returnString.replace("]}", "}]")
        returnString = returnString.replace("}]}", "}]")
        returnString = returnString.replace("{"+"}", "")
        returnString = returnString.replace("}"+"}", "}")
        returnString = returnString.replace("{"+"{", "{")

        return returnString


    def infer(self, plugin_manager, text, out_path, vocoder, speaker_i, pace=1.0, editor_data=None, old_sequence=None, globalAmplitudeModifier=None, base_lang="en", base_emb=None, useSR=False, useCleanup=False):

        sequenceSplitByLanguage = self.preprocess_prompt_language(text, base_lang)

        # Make sure all languages' text processors are initialized
        for subSequence in sequenceSplitByLanguage:
            langCode = list(subSequence.keys())[0]
            if langCode not in self.lang_tp.keys():
                self.lang_tp[langCode] = get_text_preprocessor(langCode, self.base_dir, logger=self.logger)

        try:
            pad_symb = len(ALL_SYMBOLS)-2
            all_sequence = []
            all_cleaned_text = []
            all_text = []
            all_lang_ids = []

            for ssi, subSequence in enumerate(sequenceSplitByLanguage):
                langCode = list(subSequence.keys())[0]
                subSeq = subSequence[langCode]
                sequence, cleaned_text = self.lang_tp[langCode].text_to_sequence(subSeq)

                if ssi<len(sequenceSplitByLanguage)-1:
                    sequence = sequence + [pad_symb]

                all_sequence.append(sequence)
                all_cleaned_text += ("|"+cleaned_text) if len(all_cleaned_text) else cleaned_text
                if ssi<len(sequenceSplitByLanguage)-1:
                    all_cleaned_text = all_cleaned_text + ["|<PAD>"]
                all_text.append(torch.LongTensor(sequence))

                language_id = self.language_id_mapping[langCode]
                all_lang_ids += [language_id for _ in range(len(sequence))]

        except ValueError as e:
            self.logger.info("====")
            self.logger.info(str(e))
            self.logger.info("====--")
            if "not in list" in str(e):
                symbol_not_in_list = str(e).split("is not in list")[0].split("ValueError:")[-1].replace("'", "").strip()
                return f'ERR: ARPABET_NOT_IN_LIST: {symbol_not_in_list}'

        all_cleaned_text = "".join(all_cleaned_text)

        text = torch.cat(all_text, dim=0)
        text = pad_sequence([text], batch_first=True).to(self.models_manager.device)

        with torch.no_grad():

            if old_sequence is not None:
                old_sequence = re.sub(r'[^a-zA-Z\s\(\)\[\]0-9\?\.\,\!\'\{\}\_\@]+', '', old_sequence)
                old_sequence, clean_old_sequence = self.lang_tp[base_lang].text_to_sequence(old_sequence)#, "english_basic", ['english_cleaners'])
                old_sequence = torch.LongTensor(old_sequence)
                old_sequence = pad_sequence([old_sequence], batch_first=True).to(self.models_manager.device)

            lang_ids = torch.tensor(all_lang_ids).to(self.models_manager.device)

            num_embs = text.shape[1]
            base_emb = [float(val) for val in base_emb.split(",")] if "," in base_emb else self.base_emb
            speaker_embs = [torch.tensor(base_emb).unsqueeze(dim=0)[0].unsqueeze(-1)]
            speaker_embs = torch.stack(speaker_embs, dim=0).to(self.models_manager.device)#.unsqueeze(-1)
            speaker_embs = speaker_embs.repeat(1,1,num_embs)

            editorStyles = editor_data[-1]
            # Do interpolations of speaker style embeddings
            if editorStyles is not None:
                # normalizing_scales = np.array([float(1) for _ in range(num_embs)])
                style_keys = list(editorStyles.keys())
                for style_key in style_keys:
                    emb = editorStyles[style_key]["embedding"]
                    sliders_vals = editorStyles[style_key]["sliders"]
                    # normalizing_scales += np.array(sliders_vals)

                    style_embs = [torch.tensor(emb).unsqueeze(dim=0)[0].unsqueeze(-1)]
                    style_embs = torch.stack(style_embs, dim=0).to(self.models_manager.device)#.unsqueeze(-1)
                    style_embs = style_embs.repeat(1,1,num_embs)
                    sliders_vals = torch.tensor(sliders_vals).to(self.models_manager.device)
                    speaker_embs = speaker_embs*(1-sliders_vals) + sliders_vals*style_embs

                # speaker_embs = speaker_embs / torch.tensor(normalizing_scales).to(self.models_manager.device)

            speaker_embs = speaker_embs.float()




            lang_embs = lang_ids # TODO, use pre-extracted trained language embeddings, for interpolation



            out = self.model.infer_advanced(self.logger, plugin_manager, [all_cleaned_text], text, lang_embs=lang_embs, speaker_embs=speaker_embs, pace=pace, editor_data=editor_data, old_sequence=old_sequence)
            if isinstance(out, str):
                return f'ERR:{out}'
            else:
                output_wav, dur_pred, pitch_pred, energy_pred, em_pred, start_index, end_index, wav_mult = out

                [em_angry_pred, em_happy_pred, em_sad_pred, em_surprise_pred] = em_pred

                wav = output_wav.squeeze().cpu().detach().numpy()
                wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
                if wav_mult is not None:
                    wav_norm = wav_norm * wav_mult
                if useCleanup:
                    ffmpeg_path = f'{"./resources/app" if self.PROD else "."}/python/ffmpeg.exe'

                    if useSR:
                        scipy.io.wavfile.write(out_path.replace(".wav", "_preSR.wav"), 22050, wav_norm.astype(np.int16))
                    else:
                        scipy.io.wavfile.write(out_path.replace(".wav", "_preCleanupPreFFmpeg.wav"), 22050, wav_norm.astype(np.int16))
                        stream = ffmpeg.input(out_path.replace(".wav", "_preCleanupPreFFmpeg.wav"))
                        ffmpeg_options = {"ar": 48000}
                        output_path = out_path.replace(".wav", "_preCleanup.wav")
                        stream = ffmpeg.output(stream, output_path, **ffmpeg_options)
                        out, err = (ffmpeg.run(stream, cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True, overwrite_output=True))
                        os.remove(out_path.replace(".wav", "_preCleanupPreFFmpeg.wav"))

                else:
                    scipy.io.wavfile.write(out_path.replace(".wav", "_preSR.wav") if useSR else out_path, 22050, wav_norm.astype(np.int16))

                if useSR:
                    self.models_manager.init_model("nuwave2")
                    self.models_manager.models("nuwave2").sr_audio(out_path.replace(".wav", "_preSR.wav"), out_path.replace(".wav", "_preCleanup.wav") if useCleanup else out_path)

                if useCleanup:
                    self.models_manager.init_model("deepfilternet2")
                    self.models_manager.models("deepfilternet2").cleanup_audio(out_path.replace(".wav", "_preCleanup.wav"), out_path)



        [pitch, durations, energy, em_angry, em_happy, em_sad, em_surprise] = [
            pitch_pred.squeeze().cpu().detach().numpy(),
            dur_pred.squeeze().cpu().detach().numpy(),
            energy_pred.cpu().detach().numpy() if energy_pred is not None else [],
            em_angry_pred.squeeze().cpu().detach().numpy() if em_angry_pred is not None else [],
            em_happy_pred.squeeze().cpu().detach().numpy() if em_happy_pred is not None else [],
            em_sad_pred.squeeze().cpu().detach().numpy() if em_sad_pred is not None else [],
            em_surprise_pred.squeeze().cpu().detach().numpy() if em_surprise_pred is not None else [],
        ]
        editor_values_text = ",".join([str(v) for v in pitch]) + "\n" + \
                             ",".join([str(v) for v in durations]) + "\n" + \
                             ",".join([str(v) for v in energy]) + "\n" + \
                             ",".join([str(v) for v in em_angry]) + "\n" + \
                             ",".join([str(v) for v in em_happy]) + "\n" + \
                             ",".join([str(v) for v in em_sad]) + "\n" + \
                             ",".join([str(v) for v in em_surprise]) + "\n" + \
                             json.dumps(editorStyles)

        del pitch_pred, dur_pred, energy_pred, em_angry, em_happy, em_sad, em_surprise, text, sequence
        return editor_values_text +"\n"+all_cleaned_text +"\n"+ f'{start_index}\n{end_index}'

    def set_device (self, device):
        self.device = device
        self.model = self.model.to(device)
        self.model.pitch_emb_values = self.model.pitch_emb_values.to(device)
        self.model.device = device


