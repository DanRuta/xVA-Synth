import os

import torch
import traceback

class ModelsManager(object):

    def __init__(self, logger, PROD, device="cpu"):
        super(ModelsManager, self).__init__()

        self.models_bank = {}
        self.logger = logger
        self.PROD = PROD
        self.device_label = device
        self.device = torch.device(device)

    def init_model (self, model_key):
        model_key = model_key.lower()
        try:
            if model_key in list(self.models_bank.keys()) and self.models_bank[model_key].isReady:
                return
            self.logger.info(f'ModelsManager: Initializing model: {model_key}')

            if model_key=="resemblyzer":
                from python.resemblyzer.model import ResemblyzerModel
                self.models_bank[model_key] = ResemblyzerModel(self.logger, self.PROD, self.device, self)

            elif model_key=="hifigan":
                from python.hifigan.model import HiFi_GAN
                self.models_bank[model_key] = HiFi_GAN(self.logger, self.PROD, self.device, self)

            elif model_key=="big_waveglow":
                from python.big_waveglow.model import BIG_WaveGlow
                self.models_bank[model_key] = BIG_WaveGlow(self.logger, self.PROD, self.device, self)

            elif model_key=="256_waveglow":
                from python.waveglow.model import WaveGlow
                self.models_bank[model_key] = WaveGlow(self.logger, self.PROD, self.device, self)

            elif model_key=="fastpitch":
                from python.fastpitch.model import FastPitch
                self.models_bank[model_key] = FastPitch(self.logger, self.PROD, self.device, self)

            elif model_key=="fastpitch1_1":
                from python.fastpitch1_1.model import FastPitch1_1
                self.models_bank[model_key] = FastPitch1_1(self.logger, self.PROD, self.device, self)

            elif model_key=="xvapitch":
                from python.xvapitch.model import xVAPitch
                self.models_bank[model_key] = xVAPitch(self.logger, self.PROD, self.device, self)

            elif model_key=="s2s_fastpitch1_1":
                from python.fastpitch1_1.model import FastPitch1_1 as S2S_FastPitch1_1
                self.models_bank[model_key] = S2S_FastPitch1_1(self.logger, self.PROD, self.device, self)

            elif model_key=="wav2vec2":
                from python.wav2vec2.model import Wav2Vec2
                self.models_bank[model_key] = Wav2Vec2(self.logger, self.PROD, self.device, self)

            self.models_bank[model_key].model = self.models_bank[model_key].model.to(self.device)
        except:
            self.logger.info(traceback.format_exc())

    def load_model (self, model_key, ckpt_path, **kwargs):

        if model_key not in self.models_bank.keys():
            self.init_model(model_key)

        if not os.path.exists(ckpt_path):
            return "ENOENT"

        if self.models_bank[model_key].ckpt_path != ckpt_path:
            self.logger.info(f'ModelsManager: Loading model checkpoint: {model_key}, {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.models_bank[model_key].load_state_dict(ckpt_path, ckpt, **kwargs)

    def set_device (self, device):
        if device=="gpu":
            device = "cuda"
        if self.device_label==device:
            return
        self.device_label = device
        self.device = torch.device(device)
        self.logger.info(f'ModelsManager: Changing device to: {device}')
        for model_key in list(self.models_bank.keys()):
            self.models_bank[model_key].set_device(self.device)

    def models (self, key):
        return self.models_bank[key.lower()]
