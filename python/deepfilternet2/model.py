from df.enhance import enhance, init_df, load_audio, save_audio

class DeepFilter2Model(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(DeepFilter2Model, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.path = "./resources/app" if PROD else "."

        model, df_state, _ = init_df(config_allow_defaults=True)
        self.model = model
        self.df_state = df_state

        self.isReady = True


    def load_state_dict (self, ckpt_path, sd):
        self.ckpt_path = ckpt_path


    def cleanup_audio (self, in_path, out_path):
        audio, _ = load_audio(in_path, sr=self.df_state.sr())
        enhanced = enhance(self.model, self.models_manager.device, self.df_state, audio)
        save_audio(out_path, enhanced, self.df_state.sr())

    def set_device (self, device):
        self.device = device
        self.model = self.model.to(self.device)
