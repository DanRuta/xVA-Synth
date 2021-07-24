import sys
import argparse
from python.waveglow import models
from python.waveglow.denoiser import Denoiser

class WaveGlow(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(WaveGlow, self).__init__()

        import python.waveglow.waveglow as glow
        sys.modules['glow'] = glow

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.ckpt_path = None

        model_name = "waveglow"
        parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference', allow_abbrev=False)
        model_parser = models.parse_model_args(model_name, parser, add_help=False)
        model_args, model_unk_args = model_parser.parse_known_args()

        model_config = models.get_model_config(model_name, model_args)

        self.model = models.get_model(model_name, model_config, self.device, self.logger, forward_is_infer=True, jitable=False)


        self.model.device = self.device
        self.model.eval()
        self.model.to(self.device)

        self.isReady = True


    def forward (self, data, sigma):
        return self.model.infer(data, sigma)


    def load_state_dict (self, ckpt_path, ckpt):
        self.ckpt_path = ckpt_path

        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']

            if any(key.startswith('module.') for key in sd):
                sd = {k.replace('module.', ''): v for k,v in sd.items()}
            self.model.load_state_dict(sd, strict=True)
        else:
            self.model = ckpt['model']

        self.model = self.model.remove_weightnorm(self.model).to(self.device)
        self.model.device = self.device
        self.model.eval()
        self.denoiser = Denoiser(self.model, self.device).to(self.device)

    def set_device (self, device):
        self.device = device
        self.model = self.model.to(device)
        self.model.device = device
        self.model.set_device(device)
        self.denoiser.set_device(device)