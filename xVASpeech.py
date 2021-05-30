
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import OrderedDict
from python.common.text.text_processing import TextProcessing
from python.fastpitch.transformer import FFTransformer


class xVASpeech(nn.Module):
    def __init__(self, logger, n_mel_channels=80, max_seq_len=2048, n_symbols=148, padding_idx=0,
                 symbols_embedding_dim=384, in_fft_n_layers=6, in_fft_n_heads=1,
                 in_fft_d_head=64,
                 in_fft_conv1d_kernel_size=3, in_fft_conv1d_filter_size=1536,
                 in_fft_output_size=384,
                 p_in_fft_dropout=0.1, p_in_fft_dropatt=0.1, p_in_fft_dropemb=0.0,
                 out_fft_n_layers=6, out_fft_n_heads=1, out_fft_d_head=64,
                 out_fft_conv1d_kernel_size=3, out_fft_conv1d_filter_size=1536,
                 out_fft_output_size=384,
                 p_out_fft_dropout=0.1, p_out_fft_dropatt=0.1, p_out_fft_dropemb=0.0,
                 dur_predictor_kernel_size=3, dur_predictor_filter_size=256,
                 p_dur_predictor_dropout=0.1, dur_predictor_n_layers=2,
                 pitch_predictor_kernel_size=3, pitch_predictor_filter_size=256,
                 p_pitch_predictor_dropout=0.1, pitch_predictor_n_layers=2,
                 pitch_embedding_kernel_size=3, n_speakers=2, speaker_emb_weight=1.0):
        super(xVASpeech, self).__init__()

        self.rev_proj = nn.Linear(n_mel_channels, out_fft_output_size, bias=True)
        self.device = None
        self.voiceId = None
        self.logger = logger

        self.rev_decoder = FFTransformer(
            n_layer=out_fft_n_layers, n_head=out_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=out_fft_d_head,
            d_inner=out_fft_conv1d_filter_size,
            kernel_size=out_fft_conv1d_kernel_size,
            dropout=p_out_fft_dropout,
            dropatt=p_out_fft_dropatt,
            dropemb=p_out_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim
        )

        self.symbol_set = "english_basic"
        self.text_cleaners = ["english_cleaners"]

        self.register_buffer('pitch_mean', torch.tensor([113.7768555512196]))
        self.register_buffer('pitch_std', torch.tensor([0]))
        self.register_buffer('model_version', torch.tensor([1]))


        self.tp = TextProcessing(self.symbol_set, self.text_cleaners)
        self.num_symbols = len(list(self.tp.symbol_to_id.keys()))
        self.letters = list(self.tp.symbol_to_id.keys())

        self.USE_FFT_CHAR_PRED = True
        self.USE_FFT_CHAR_PRED_PROJ = True
        self.USE_FFT_CHAR_SEQ = False
        self.DO_PITCH_LOSS = True
        self.DO_REV_COMP_LOSS = True
        self.DO_SMALL_FFT_CHAR_PRED = True
        self.DO_FFT2 = False
        self.DO_FP = True

        self.char_pred_fft_proj = nn.Linear(symbols_embedding_dim, self.num_symbols, bias=True)


    def infer (self, mel):
        mel_max_len = mel.size(2)
        dec_out = self.rev_proj(mel.transpose(1, 2))

        # Assign to this the size of each mel in the batch. Max when just one
        output_lengths = [mel_max_len]
        output_lengths = torch.tensor(output_lengths).to(self.device)

        len_regulated, dec_lens = self.rev_decoder(dec_out, output_lengths)

        padded_char_pred = []

        if self.USE_FFT_CHAR_PRED:
            if self.DO_FFT2:
                padded_char_pred, cpfft_lens = self.char_pred_fft(len_regulated, output_lengths)
            else:
                padded_char_pred = len_regulated

            if self.USE_FFT_CHAR_PRED_PROJ:
                padded_char_pred = self.char_pred_fft_proj(padded_char_pred)
        else:
            char_pred = self.char_pred(len_regulated)
            padded_char_pred = char_pred

        char_seq2, _ = padded_char_pred, output_lengths
        char_seq2 = torch.argmax(char_seq2, dim=2)
        text = "".join([self.letters[i] for i in char_seq2.cpu().detach().numpy()[0]])
        seq_numpy = len_regulated.cpu().detach().numpy()

        # Build output
        text_out = []
        pitch = []
        durs = []

        last_c = None
        dur_counter = 0
        curr_pitch_vals = []

        last_c = text[0]
        for ci, char in enumerate(text):
            if last_c != char or ci==len(text)-1:
                text_out.append(last_c)
                durs.append(dur_counter)
                dur_counter = 1
                pitch.append(np.mean(curr_pitch_vals))
                last_c = char
            else:
                curr_pitch_vals.append(seq_numpy[0][ci])
                dur_counter += 1

        text_out = "".join(text_out)

        return padded_char_pred, pitch, durs, text_out



def init (PROD, use_gpu, logger):
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if use_gpu else 'cpu')

    xVASpeechModel = xVASpeech(logger)
    xVASpeechModel.device = device
    xVASpeechModel = xVASpeechModel.to(device)

    return xVASpeechModel


def loadModel (xVASpeechModel, voiceId, ckpt):
    print(f'Loading xVASpeech model: {ckpt}')

    checkpoint_data = torch.load(ckpt, map_location="cpu")
    sd = {k.replace('module.', ''): v for k, v in checkpoint_data.items()}

    getattr(xVASpeechModel, 'module', xVASpeechModel).load_state_dict(sd, strict=False)
    xVASpeechModel.eval()
    xVASpeechModel.voiceId = voiceId
    del checkpoint_data

    return xVASpeechModel


from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from python.common.stft import STFT
from python.common.audio_processing import dynamic_range_compression, dynamic_range_decompression


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output
stft = TacotronSTFT(1024, 256, 1024, 80, 22050, 0, 8000)



def maybe_pad(vec, l):
    assert np.abs(vec.shape[0] - l) <= 3
    vec = vec[:l]
    if vec.shape[0] < l:
        vec = np.pad(vec, pad_width=(0, l - vec.shape[0]))
    return vec



import parselmouth
def calculate_pitch (fname, durs):
    mel_len = durs.sum()
    durs_cum = np.cumsum(np.pad(durs, (1, 0), mode="constant"))
    snd = parselmouth.Sound(fname)

    pitch = snd.to_pitch(time_step=snd.duration / (mel_len + 3)).selected_array['frequency']

    assert np.abs(mel_len - pitch.shape[0]) <= 1.0

    # Average pitch over characters
    pitch_char = np.zeros((durs.shape[0],), dtype=np.float)
    for idx, a, b in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = pitch[a:b][np.where(pitch[a:b] != 0.0)[0]]
        pitch_char[idx] = np.mean(values) if len(values) > 0 else 0.0

    pitch_char = maybe_pad(pitch_char, len(durs))

    return pitch_char

def normalize_pitch_vectors(logger, pitch_vecs):
    nonzeros = [v for v in pitch_vecs if v!=0.0]
    mean, std = np.mean(nonzeros), np.std(nonzeros)

    logger.info(f'mean {mean}')
    logger.info(f'std {std}')

    for vi, v in enumerate(pitch_vecs):
        v -= mean
        v /= std
        pitch_vecs[vi] = v

    return pitch_vecs


from parselmouth.praat import call
def change_pitch(logger, xVASpeechModel, fname, factor):

    model_mean = xVASpeechModel.pitch_mean.item()

    sound = parselmouth.Sound(fname)
    pitch = sound.to_pitch().selected_array['frequency']
    pitch = [p for p in pitch if p!=0]
    median = np.median(pitch)
    factor = model_mean / median

    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)

    pitch_tier = call(manipulation, "Extract pitch tier")

    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)

    call([pitch_tier, manipulation], "Replace pitch tier")
    sound_changed_pitch = call(manipulation, "Get resynthesis (overlap-add)")

    out_file_name = f'{fname.split(".wav")[0]}_praat.wav'
    sound_changed_pitch.save(out_file_name, "WAV")
    return out_file_name





def infer (logger, xVASpeechModel, audio_file, use_gpu, doPitchShift=False):

    device = torch.device('cuda' if use_gpu else 'cpu')
    max_wav_value = 32768.0

    # Change the median pitch of the input to match the average pitch of the xVASpeech voice
    if doPitchShift:
        logger.info(f'change_pitch: {audio_file}')
        audio_file = change_pitch(logger, xVASpeechModel, audio_file, 2)

    # Read the audio file
    logger.info(f'Reading in file: {audio_file}')
    _, data = read(audio_file)

    audio = torch.FloatTensor(data.astype(np.float32))
    audio_norm = audio / max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)

    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)

    char_pred, pitch, durs, text = xVASpeechModel.infer(melspec.unsqueeze(0).to(device))

    pitch = calculate_pitch(audio_file, np.array(durs))
    pitch = normalize_pitch_vectors(logger, pitch)

    # Some "temporary" hacks to manually improve the quality a bit
    pitch_final = []
    for p in pitch:
        if p>2 or p<-2:
            pitch_final.append(p/10)
        else:
            pitch_final.append(p)
    pitch = pitch_final


    return text, pitch, durs