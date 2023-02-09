import math
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.conv import Conv1d

from python.xvapitch.glow_tts import RelativePositionTransformer
from python.xvapitch.wavenet import WN
from python.xvapitch.hifigan import HifiganGenerator
from python.xvapitch.sdp import StochasticDurationPredictor#, StochasticPredictor

from python.xvapitch.util import maximum_path, rand_segments, segment, sequence_mask, generate_path
from python.xvapitch.text import get_text_preprocessor, ALL_SYMBOLS, lang_names


class xVAPitch(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.args.init_discriminator = True
        self.args.speaker_embedding_channels = 512
        self.args.use_spectral_norm_disriminator = False
        self.args.d_vector_dim = 512
        self.args.use_language_embedding = True
        self.args.detach_dp_input = True

        self.END2END = True

        self.embedded_language_dim = 12
        self.latent_size = 256

        num_languages = len(list(lang_names.keys()))
        self.emb_l = nn.Embedding(num_languages, self.embedded_language_dim)



        self.length_scale = 1.0
        self.noise_scale = 1.0

        self.inference_noise_scale = 0.333

        self.inference_noise_scale_dp = 0.333
        self.noise_scale_dp = 1.0
        self.max_inference_len = None
        self.spec_segment_size = 32


        self.text_encoder = TextEncoder(
            # 165,
            len(ALL_SYMBOLS),
            self.latent_size,#192,
            self.latent_size,#192,
            768,
            2,
            10,
            3,
            0.1,
            # language_emb_dim=4,
            language_emb_dim=self.embedded_language_dim,
        )

        self.posterior_encoder = PosteriorEncoder(
            513,
            self.latent_size,#+self.embedded_language_dim if self.args.flc else self.latent_size,#192,
            self.latent_size,#+self.embedded_language_dim if self.args.flc else self.latent_size,#192,
            kernel_size=5,
            dilation_rate=1,
            num_layers=16,
            cond_channels=self.args.d_vector_dim,
        )

        self.flow = ResidualCouplingBlocks(
            self.latent_size,#192,
            self.latent_size,#192,
            kernel_size=5,
            dilation_rate=1,
            num_layers=4,
            cond_channels=self.args.d_vector_dim,
            args=self.args
        )

        self.duration_predictor = StochasticDurationPredictor(
            self.latent_size,#192,
            self.latent_size,#192,
            3,
            0.5,
            4,
            cond_channels=self.args.d_vector_dim,
            language_emb_dim=self.embedded_language_dim,
        )

        self.waveform_decoder = HifiganGenerator(
            self.latent_size,#192,
            1,
            "1",
            [[1,3,5],[1,3,5],[1,3,5]],
            [3,7,11],
            [16,16,4,4],
            512,
            [8,8,2,2],
            inference_padding=0,
            # cond_channels=self.args.d_vector_dim+self.embedded_language_dim if self.args.flc else self.args.d_vector_dim,
            cond_channels=self.args.d_vector_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

        self.USE_PITCH_COND = False
        # self.USE_PITCH_COND = True
        if self.USE_PITCH_COND:
            self.pitch_predictor = RelativePositioningPitchEnergyEncoder(
                # 165,
                # len(ALL_SYMBOLS),
                out_channels=1,
                hidden_channels=self.latent_size+self.embedded_language_dim,#196,
                hidden_channels_ffn=768,
                num_heads=2,
                # num_layers=10,
                num_layers=3,
                kernel_size=3,
                dropout_p=0.1,
                # language_emb_dim=4,
                conditioning_emb_dim=self.args.d_vector_dim,
            )

            self.pitch_emb = nn.Conv1d(
                # 1, 384,
                # 1, 196,
                1,
                self.args.expanded_flow_dim if args.expanded_flow else self.latent_size,
                # pitch_conditioning_formants, symbols_embedding_dim,
                kernel_size=3,
                padding=int((3 - 1) / 2))


        self.TEMP_timing = []




    def infer_get_lang_emb (self, language_id):

        aux_input = {
            # "d_vectors": embedding.unsqueeze(dim=0),
            "language_ids": language_id
        }

        sid, g, lid = self._set_cond_input(aux_input)
        lang_emb = self.emb_l(lid).unsqueeze(-1)
        return lang_emb


    def infer_advanced (self, logger, plugin_manager, cleaned_text, text, lang_embs, speaker_embs, pace=1.0, editor_data=None, old_sequence=None, pitch_amp=None):

        if (editor_data is not None) and ((editor_data[0] is not None and len(editor_data[0])) or (editor_data[1] is not None and len(editor_data[1]))):
            pitch_pred, dur_pred, energy_pred = editor_data
            # TODO, use energy_pred
            dur_pred = torch.tensor(dur_pred)
            dur_pred = dur_pred.view((1, dur_pred.shape[0])).float().to(self.device)
            pitch_pred = torch.tensor(pitch_pred)
            pitch_pred = pitch_pred.view((1, pitch_pred.shape[0])).float().to(self.device)
            energy_pred = torch.tensor(energy_pred)
            energy_pred = energy_pred.view((1, energy_pred.shape[0])).float().to(self.device)

            if not self.USE_PITCH_COND and pitch_pred.shape[1]==speaker_embs.shape[2]:
                pitch_delta = self.pitch_emb_values * pitch_pred
                speaker_embs = speaker_embs + pitch_delta.float()

            try:
                wav, dur_pred, pitch_pred_out, energy_pred, start_index, end_index, wav_mult = self.infer_using_vals(logger, plugin_manager, cleaned_text, text, lang_embs, speaker_embs, pace, dur_pred_existing=dur_pred, pitch_pred_existing=pitch_pred, energy_pred_existing=energy_pred, old_sequence=old_sequence, new_sequence=text, pitch_amp=pitch_amp)
                if not self.USE_PITCH_COND and pitch_pred.shape[1]==speaker_embs.shape[2]:
                    pitch_pred_out = pitch_pred
                return wav, dur_pred, pitch_pred_out, energy_pred, start_index, end_index, wav_mult
            except:
                print(traceback.format_exc())
                logger.info(traceback.format_exc())
                return traceback.format_exc()
                return self.infer_using_vals(logger, plugin_manager, cleaned_text, text, lang_embs, speaker_embs, pace, None, None, None, None, None, pitch_amp=pitch_amp)

        else:
            return self.infer_using_vals(logger, plugin_manager, cleaned_text, text, lang_embs, speaker_embs, pace, None, None, None, None, None, pitch_amp=pitch_amp)




    def infer_using_vals (self, logger, plugin_manager, cleaned_text, sequence, lang_embs, speaker_embs, pace, dur_pred_existing, pitch_pred_existing, energy_pred_existing, old_sequence, new_sequence, pitch_amp=None):

        start_index = None
        end_index = None

        # Calculate text splicing bounds, if needed
        if old_sequence is not None:

            old_sequence_np = old_sequence.cpu().detach().numpy()
            old_sequence_np = list(old_sequence_np[0])
            new_sequence_np = new_sequence.cpu().detach().numpy()
            new_sequence_np = list(new_sequence_np[0])


            # Get the index of the first changed value
            if old_sequence_np[0]==new_sequence_np[0]: # If the start of both sequences is the same, then the change is not at the start
                for i in range(len(old_sequence_np)):
                    if i<len(new_sequence_np):
                        if old_sequence_np[i]!=new_sequence_np[i]:
                            start_index = i-1
                            break
                    else:
                        start_index = i-1
                        break
                if start_index is None:
                    start_index = len(old_sequence_np)-1


            # Get the index of the last changed value
            old_sequence_np.reverse()
            new_sequence_np.reverse()
            if old_sequence_np[0]==new_sequence_np[0]: # If the end of both reversed sequences is the same, then the change is not at the end
                for i in range(len(old_sequence_np)):
                    if i<len(new_sequence_np):
                        if old_sequence_np[i]!=new_sequence_np[i]:
                            end_index = len(old_sequence_np)-1-i+1
                            break
                    else:
                        end_index = len(old_sequence_np)-1-i+1
                        break

            old_sequence_np.reverse()
            new_sequence_np.reverse()


        # cleaned_text is the actual text phonemes
        input_symbols = sequence
        x_lengths = torch.where(input_symbols > 0, torch.ones_like(input_symbols), torch.zeros_like(input_symbols)).sum(dim=1)



        lang_emb_full = None # TODO
        self.text_encoder.logger = logger

        # TODO, store a bank of trained 31 language embeds, to use for interpolating
        lang_emb = self.emb_l(lang_embs).unsqueeze(-1)

        x, x_emb, x_mask = self.text_encoder(input_symbols, x_lengths, lang_emb=lang_emb, stats=False, lang_emb_full=lang_emb_full)
        m_p, logs_p = self.text_encoder(x, x_lengths, lang_emb=lang_emb, stats=True, x_mask=x_mask)


        self.inference_noise_scale_dp = 0 # TEMP DEBUGGING. REMOVE - or should I? It seems to make it worse, the higher it is

        # Calculate its own pitch, and duration vals if these were not already provided
        if (dur_pred_existing is None or dur_pred_existing.shape[1]==0) or old_sequence is not None:
            # Predict durations
            self.duration_predictor.logger = logger
            logw = self.duration_predictor(x, x_mask, g=speaker_embs, reverse=True, noise_scale=self.inference_noise_scale_dp, lang_emb=lang_emb)

            w = torch.exp(logw) * x_mask * self.length_scale
            w = w * 1.3 # The model seems to generate quite fast speech, so I'm gonna just globally adjust that
            w = w * (pace.unsqueeze(2) if torch.is_tensor(pace) else 1)
            w_ceil = w
            w_ceil = torch.ceil(w)
            dur_pred = w_ceil
        else:
            dur_pred = dur_pred_existing.unsqueeze(dim=0)
            dur_pred = dur_pred * pace

        y_lengths = torch.clamp_min(torch.sum(torch.round(dur_pred), [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype)

        # if False and dur_pred.shape[0]>1:
        if dur_pred.shape[0]>1:
            attn_all = []
            m_p_all = []
            logs_p_all = []
            pitch_pred_all = []
            for b in range(dur_pred.shape[0]):
                attn_mask = torch.unsqueeze(x_mask[b,:].unsqueeze(0), 2) * torch.unsqueeze(y_mask[b,:].unsqueeze(0), -1)
                attn_all.append(generate_path(dur_pred.squeeze(1)[b,:].unsqueeze(0), attn_mask.squeeze(0).transpose(1, 2)))
                m_p_all.append(torch.matmul(attn_all[-1].transpose(1, 2), m_p[b,:].unsqueeze(0).transpose(1, 2)).transpose(1, 2))
                logs_p_all.append(torch.matmul(attn_all[-1].transpose(1, 2), logs_p[b,:].unsqueeze(0).transpose(1, 2)).transpose(1, 2))
                if self.USE_PITCH_COND:
                    pitch_pred_all.append(self.pitch_predictor(x[b,:].unsqueeze(0).permute(0, 2, 1), x_lengths[b].unsqueeze(0), speaker_emb=speaker_embs[b,:].unsqueeze(0), stats=False))
            del attn_all
            m_p = torch.stack(m_p_all, dim=1).squeeze(dim=0)
            logs_p = torch.stack(logs_p_all, dim=1).squeeze(dim=0)
            if self.USE_PITCH_COND:
                pitch_pred = torch.stack(pitch_pred_all, dim=1).squeeze(dim=0)
            else:
                pitch_pred = torch.zeros((x.shape[0], x.shape[0], x.shape[2])).to(x)
        else:
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = generate_path(dur_pred.squeeze(1), attn_mask.squeeze(0).transpose(1, 2))
            m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
            logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)


            if self.USE_PITCH_COND:
                if (pitch_pred_existing is None or pitch_pred_existing.shape[1]==0) or old_sequence is not None:
                    # Pitch over chars
                    pitch_pred = self.pitch_predictor(x.permute(0, 2, 1), x_lengths, speaker_emb=speaker_embs, stats=False)
                else:
                    pitch_pred = pitch_pred_existing.unsqueeze(1)#.unsqueeze(1)
            else:
                pitch_pred = torch.zeros((x.shape[0], x.shape[0], x.shape[2])).to(x)


        # Splice/replace pitch/duration values from the old input if simulating only a partial re-generation
        if start_index is not None or end_index is not None:
            dur_pred_np = list(dur_pred.cpu().detach().numpy())[0]
            pitch_pred_np = list(pitch_pred.cpu().detach().numpy())[0][0]
            dur_pred_existing_np = list(dur_pred_existing.cpu().detach().numpy())[0]
            pitch_pred_existing_np = list(pitch_pred_existing.cpu().detach().numpy())[0]

            if start_index is not None: # Replace starting values

                for i in range(start_index+1):
                    dur_pred_np[i] = dur_pred_existing_np[i]
                    pitch_pred_np[i] = pitch_pred_existing_np[i]

            if end_index is not None: # Replace end values

                for i in range(len(old_sequence_np)-end_index):
                    dur_pred_np[-i-1] = dur_pred_existing_np[-i-1]
                    pitch_pred_np[-i-1] = pitch_pred_existing_np[-i-1]

            dur_pred = torch.tensor(dur_pred_np).to(self.device).unsqueeze(0)
            pitch_pred = torch.tensor(pitch_pred_np).to(self.device).unsqueeze(0).unsqueeze(0)


        if pitch_amp is not None:
            pitch_pred = pitch_pred * pitch_amp.unsqueeze(dim=-1)


        if plugin_manager is not None and len(plugin_manager.plugins["synth-line"]["mid"]):
            pitch_pred = pitch_pred.cpu().detach().numpy()
            plugin_data = {
                "duration": dur_pred.cpu().detach().numpy(),
                "pitch": pitch_pred.reshape((pitch_pred.shape[0],pitch_pred.shape[2])),
                "text": [val.split("|") for val in cleaned_text],
                "is_fresh_synth": pitch_pred_existing is None and dur_pred_existing is None,

                "pluginsContext": plugin_manager.context
            }
            plugin_manager.run_plugins(plist=plugin_manager.plugins["synth-line"]["mid"], event="mid synth-line", data=plugin_data)

            dur_pred = torch.tensor(plugin_data["duration"]).to(self.device)
            pitch_pred = torch.tensor(plugin_data["pitch"]).unsqueeze(1).to(self.device)

        if self.USE_PITCH_COND:
            pitch_emb = self.pitch_emb(self.expand_pitch_energy(pitch_pred, dur_pred, logger=logger))
            m_p += pitch_emb * self.args.pe_scaling


        # TODO, incorporate some sort of control for this
        self.inference_noise_scale = 0
        for flow in self.flow.flows:
            flow.logger = logger
            flow.enc.logger = logger

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.inference_noise_scale
        z = self.flow(z_p, y_mask, g=speaker_embs, reverse=True)
        self.waveform_decoder.logger = logger
        wav = self.waveform_decoder((z * y_mask.unsqueeze(1))[:, :, : self.max_inference_len], g=speaker_embs)

        # In batch mode, trim the shorter audio waves in the batch. The masking doesn't seem to work, so have to do it manually
        if dur_pred.shape[0]>1:
            wav_all = []
            for b in range(dur_pred.shape[0]):
                percent_to_mask = torch.sum(y_mask[b])/y_mask.shape[1]
                wav_all.append(wav[b,0,0:int((wav.shape[2]*percent_to_mask).item())])
            wav = wav_all


        start_index = -1 if start_index is None else start_index
        end_index = -1 if end_index is None else end_index


        # Apply volume adjustments
        stretched_energy_mult = None
        if energy_pred_existing is not None:
            energy_mult = self.expand_pitch_energy(energy_pred_existing.unsqueeze(0), dur_pred, logger=logger)
            stretched_energy_mult = torch.nn.functional.interpolate(energy_mult.unsqueeze(0).unsqueeze(0), (1,1,wav.shape[2])).squeeze()
            stretched_energy_mult = stretched_energy_mult.cpu().detach().numpy()
            energy_pred = energy_pred_existing.squeeze()
        else:
            energy_pred = [1.0 for _ in range(pitch_pred.shape[-1])]
            energy_pred = torch.tensor(energy_pred)

        return wav, dur_pred, pitch_pred, energy_pred, start_index, end_index, stretched_energy_mult




    def voice_conversion(self, y, y_lengths=None, spk1_emb=None, spk2_emb=None):

        if y_lengths is None:
            y_lengths = self.y_lengths_default

        z, _, _, y_mask = self.posterior_encoder(y, y_lengths, g=spk1_emb)
        # z_hat = z
        y_mask = y_mask.squeeze(0)
        z_p = self.flow(z, y_mask, g=spk1_emb)
        z_hat = self.flow(z_p, y_mask, g=spk2_emb, reverse=True)

        o_hat = self.waveform_decoder(z_hat * y_mask, g=spk2_emb)
        return o_hat



    def _set_cond_input (self, aux_input):
        """Set the speaker conditioning input based on the multi-speaker mode."""
        sid, g, lid = None, None, None
        # if "speaker_ids" in aux_input and aux_input["speaker_ids"] is not None:
        #     sid = aux_input["speaker_ids"]
        #     if sid.ndim == 0:
        #         sid = sid.unsqueeze_(0)
        if "d_vectors" in aux_input and aux_input["d_vectors"] is not None:
            g = F.normalize(aux_input["d_vectors"]).unsqueeze(-1)
            if g.ndim == 2:
                g = g.unsqueeze_(0)

        if "language_ids" in aux_input and aux_input["language_ids"] is not None:
            lid = aux_input["language_ids"]
            if lid.ndim == 0:
                lid = lid.unsqueeze_(0)

        return sid, g, lid


    # Opposite of average_pitch; Repeat per-symbol values by durations, to get sequence-wide values
    def expand_pitch_energy (self, vals, durations, logger=None):

        vals = vals.view((vals.shape[0], vals.shape[2]))

        if len(durations.shape)>2:
            durations = durations.view((durations.shape[0], durations.shape[2]))

        max_dur = int(torch.round(durations).sum().item())
        max_dur = int(torch.max(torch.sum(torch.round(durations), dim=1)).item())

        expanded = torch.zeros((vals.shape[0], 1, max_dur)).to(vals)

        for b in range(vals.shape[0]):
            b_vals = vals[b]
            b_durs = durations[b]
            expanded_vals = []

            for vi in range(b_vals.shape[0]):
                for dur_i in range(round(b_durs[vi].item())):
                    if len(durations.shape)>2:
                        expanded_vals.append(b_vals[vi])
                    else:
                        expanded_vals.append(b_vals[vi].unsqueeze(dim=0))

            expanded_vals = torch.tensor(expanded_vals).to(expanded)
            expanded[b,:,0:expanded_vals.shape[0]] += expanded_vals
        return expanded







class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int, # len(ALL_SYMBOLS)
        out_channels: int, # 192
        hidden_channels: int, # 192
        hidden_channels_ffn: int, # 768
        num_heads: int, # 2
        num_layers: int, # 10
        kernel_size: int, # 3
        dropout_p: float, # 0.1
        language_emb_dim: int = None,
    ):
        """Text Encoder for VITS model.

        Args:
            n_vocab (int): Number of characters for the embedding layer.
            out_channels (int): Number of channels for the output.
            hidden_channels (int): Number of channels for the hidden layers.
            hidden_channels_ffn (int): Number of channels for the convolutional layers.
            num_heads (int): Number of attention heads for the Transformer layers.
            num_layers (int): Number of Transformer layers.
            kernel_size (int): Kernel size for the FFN layers in Transformer network.
            dropout_p (float): Dropout rate for the Transformer layers.
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)

        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        if language_emb_dim:
            hidden_channels += language_emb_dim

        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout_p=dropout_p,
            layer_norm_type="2",
            rel_attn_window_size=4,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, lang_emb=None, stats=False, x_mask=None, lang_emb_full=None):
        """
        Shapes:
            - x: :math:`[B, T]`
            - x_length: :math:`[B]`
        """

        if stats:
            stats = self.proj(x) * x_mask
            m, logs = torch.split(stats, self.out_channels, dim=1)
            return m, logs
        else:
            x_emb = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]

            # concat the lang emb in embedding chars
            if lang_emb is not None or lang_emb_full is not None:
                # x = torch.cat((x_emb, lang_emb.transpose(2, 1).expand(x_emb.size(0), x_emb.size(1), -1)), dim=-1)
                if lang_emb_full is None:
                    lang_emb_full = lang_emb.transpose(2, 1).expand(x_emb.size(0), x_emb.size(1), -1)
                x = torch.cat((x_emb, lang_emb_full), dim=-1)

            x = torch.transpose(x, 1, -1)  # [b, h, t]
            x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

            x = self.encoder(x * x_mask, x_mask)
            # stats = self.proj(x) * x_mask

            # m, logs = torch.split(stats, self.out_channels, dim=1)
            return x, x_emb, x_mask

class RelativePositioningPitchEnergyEncoder(nn.Module):
    def __init__(
        self,
        # n_vocab: int, # len(ALL_SYMBOLS)
        out_channels: int, # 192
        hidden_channels: int, # 192
        hidden_channels_ffn: int, # 768
        num_heads: int, # 2
        num_layers: int, # 10
        kernel_size: int, # 3
        dropout_p: float, # 0.1
        conditioning_emb_dim: int = None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # self.emb = nn.Embedding(n_vocab, hidden_channels)

        # nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        if conditioning_emb_dim:
            hidden_channels += conditioning_emb_dim

        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            # out_channels=hidden_channels,
            out_channels=1,
            # out_channels=196,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout_p=dropout_p,
            layer_norm_type="2",
            rel_attn_window_size=4,
        )

        # self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        # self.proj = nn.Conv1d(196, out_channels * 2, 1)

    def forward(self, x, x_lengths=None, speaker_emb=None, stats=False, x_mask=None):
        """
        Shapes:
            - x: :math:`[B, T]`
            - x_length: :math:`[B]`
        """

        # concat the lang emb in embedding chars
        if speaker_emb is not None:
            x = torch.cat((x, speaker_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1)

        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        return x#, x_mask


class ResidualCouplingBlocks(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        num_flows=4,
        cond_channels=0,
        args=None
    ):
        """Redisual Coupling blocks for VITS flow layers.

        Args:
            channels (int): Number of input and output tensor channels.
            hidden_channels (int): Number of hidden network channels.
            kernel_size (int): Kernel size of the WaveNet layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            num_flows (int, optional): Number of Residual Coupling blocks. Defaults to 4.
            cond_channels (int, optional): Number of channels of the conditioning tensor. Defaults to 0.
        """
        super().__init__()
        self.args = args
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.num_flows = num_flows
        self.cond_channels = cond_channels

        self.flows = nn.ModuleList()
        for flow_i in range(num_flows):
            self.flows.append(
                ResidualCouplingBlock(
                    (192+self.args.expanded_flow_dim+self.args.expanded_flow_dim) if flow_i==(num_flows-1) and self.args.expanded_flow else channels,
                    (192+self.args.expanded_flow_dim+self.args.expanded_flow_dim) if flow_i==(num_flows-1) and self.args.expanded_flow else hidden_channels,
                    kernel_size,
                    dilation_rate,
                    num_layers,
                    cond_channels=cond_channels,
                    out_channels_override=(192+self.args.expanded_flow_dim+self.args.expanded_flow_dim) if flow_i==(num_flows-1) and self.args.expanded_flow else None,
                    mean_only=True,
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        if not reverse:
            for fi, flow in enumerate(self.flows):
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
                x = torch.flip(x, [1])
        else:
            for flow in reversed(self.flows):
                x = torch.flip(x, [1])
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x
class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        cond_channels=0,
    ):
        """Posterior Encoder of VITS model.

        ::
            x -> conv1x1() -> WaveNet() (non-causal) -> conv1x1() -> split() -> [m, s] -> sample(m, s) -> z

        Args:
            in_channels (int): Number of input tensor channels.
            out_channels (int): Number of output tensor channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size of the WaveNet convolution layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            cond_channels (int, optional): Number of conditioning tensor channels. Defaults to 0.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.cond_channels = cond_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels, hidden_channels, kernel_size, dilation_rate, num_layers, c_in_channels=cond_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_lengths: :math:`[B, 1]`
            - g: :math:`[B, C, 1]`
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        mean, log_scale = torch.split(stats, self.out_channels, dim=1)
        z = (mean + torch.randn_like(mean) * torch.exp(log_scale)) * x_mask
        return z, mean, log_scale, x_mask
class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        dropout_p=0,
        cond_channels=0,
        out_channels_override=None,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        # input layer
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        # coupling layers
        self.enc = WN(
            hidden_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_layers,
            dropout_p=dropout_p,
            c_in_channels=cond_channels,
        )
        # output layer
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)


        self.conv1d_projector = None
        if out_channels_override:
            self.conv1d_projector = nn.Conv1d(192, out_channels_override, 1)


    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        if self.conv1d_projector is not None and not reverse:
            x = self.conv1d_projector(x)

        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask.unsqueeze(1)
        h = self.enc(h, x_mask.unsqueeze(1), g=g)
        stats = self.post(h) * x_mask.unsqueeze(1)
        if not self.mean_only:
            m, log_scale = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            log_scale = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(log_scale) * x_mask.unsqueeze(1)
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(log_scale, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-log_scale) * x_mask.unsqueeze(1)
            x = torch.cat([x0, x1], 1)
            return x




def mask_from_lens(lens, max_len= None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask
class TemporalPredictor(nn.Module):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, filter_size, kernel_size, dropout,
                 n_layers=2, n_predictions=1):
        super(TemporalPredictor, self).__init__()

        self.layers = nn.Sequential(*[
            ConvReLUNorm(input_size if i == 0 else filter_size, filter_size,
                         kernel_size=kernel_size, dropout=dropout)
            for i in range(n_layers)]
        )
        self.n_predictions = n_predictions
        self.fc = nn.Linear(filter_size, self.n_predictions, bias=True)

    def forward(self, enc_out, enc_out_mask):
        out = enc_out * enc_out_mask
        out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc(out) * enc_out_mask
        return out
class ConvReLUNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0):
        super(ConvReLUNorm, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    padding=(kernel_size // 2))
        self.norm = torch.nn.LayerNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, signal):
        out = F.relu(self.conv(signal))
        out = self.norm(out.transpose(1, 2)).transpose(1, 2).to(signal.dtype)
        return self.dropout(out)
