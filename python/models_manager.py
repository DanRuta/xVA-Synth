import torch
import traceback

class ModelsManager(object):

    def __init__(self, logger, PROD, device):
        super(ModelsManager, self).__init__()

        self.models = {}
        self.logger = logger
        self.PROD = PROD
        self.device = device

    def init_model (self, model_key):
        try:
            if model_key=="xVARep":
                if model_key in self.models.keys() and self.models[model_key].isReady:
                    pass
                else:
                    self.logger.info(f'ModelsManager: Initializing model: {model_key}')
                    from python.xVARep.model import xVARep
                    self.models[model_key] = xVARep(self.logger, self.PROD, self.device)

                    self.load_model(model_key, ("./resources/app" if self.PROD else ".")+"/python/xVARep/xVARep.pt")

        except:
            self.logger.info(traceback.format_exc())



    def load_model (self, model_key, ckpt_path):
        self.logger.info(f'ModelsManager: Loading model checkpoint: {model_key}, {ckpt_path}')
        ckpt = torch.load(ckpt_path)
        self.models[model_key].load_state_dict(ckpt)

    def change_device (self, model_key, use_cuda):
        device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
        self.models[model_key] = self.models[model_key].to(device)

