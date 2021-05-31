
window.i18n = {}

window.i18n.setEnglish = () => {
    window.i18n.SELECT_GAME = "Select Game"
    window.i18n.SEARCH_VOICES = "Select voices..."
    window.i18n.SELECT_VOICE = "Select voice"
    window.i18n.SELECT_VOICE_TYPE = "Select Voice Type"
    window.i18n.KEEP_SAMPLE = "Keep Sample"
    window.i18n.GENERATE_VOICE = "Generate Voice"
    window.i18n.RENAME_THE_FILE = "Rename the file"
    window.i18n.DELETE_FILE = "Delete file"

    window.i18n.LENGTH = "Length:"
    window.i18n.RESET_LETTER = "Reset Letter"
    window.i18n.AUTO_REGEN = "Auto regenerate:"
    window.i18n.VOCODER = "Vocoder:"

    window.i18n.SEARCH_N_VOICES = "Search _ voices..."
    window.i18n.SEARCH_N_GAMES_WITH_N2_VOICES = "Search _1 games with _2 voices..."
    window.i18n.RESET = "Reset"
    window.i18n.AMPLIFY = "Amplify"
    window.i18n.FLATTEN = "Flatten"
    window.i18n.RAISE = "Raise"
    window.i18n.LOWER = "Lower"
    window.i18n.PACING = "Pacing"

    window.i18n.SETTINGS = "Settings"
    window.i18n.SETTINGS_GPU = "Use GPU (requires CUDA)"
    window.i18n.SETTINGS_AUTOPLAY = "Autoplay generated audio"
    window.i18n.SETTINGS_DEFAULT_HIFI = "Default to loading the HiFi vocoder on voice change, if available"
    window.i18n.SETTINGS_KEEP_PACING = "Keep the same pacing value on new text generations"
    window.i18n.SETTINGS_TOOLTIP = "Show the sliders tooltip"

    window.i18n.SETTINGS_DARKMODE = "Dark mode text prompt"
    window.i18n.SETTINGS_PROMPTSIZE = "Text prompt font size"
    window.i18n.SETTINGS_BG_FADE = "Background image fade opacity"
    window.i18n.SETTINGS_AUTORELOADVOICES = "Auto-reload voices on files changes"
    window.i18n.SETTINGS_KEEPEDITORSTATE = "Keep editor state on voice change"
    window.i18n.SETTINGS_OUTPUTJSON = "Output .json (needed for editing)"
    window.i18n.SETTINGS_SEQNUMBERING = "Use sequential numbering for file names"
    window.i18n.SETTINGS_EXTERNALEDIT = "External program for editing audio"
    window.i18n.SETTINGS_FFMPEG = "Use ffmpeg post-processing (requires ffmpeg >=4.3)"
    window.i18n.SETTINGS_FFMPEG_FORMAT = "Audio format (wav, mp3, etc)"
    window.i18n.SETTINGS_FFMPEG_HZ = "Audio sample rate (Hz)"
    window.i18n.SETTINGS_FFMPEG_PADSTART = "Silence padding start (ms)"
    window.i18n.SETTINGS_FFMPEG_PADEND = "Silence padding end (ms)"
    window.i18n.SETTINGS_FFMPEG_BITDEPTH = "Audio bit depth"
    window.i18n.SETTINGS_FFMPEG_AMPLITUDE = "Amplitude multiplier"
    window.i18n.SETTINGS_BATCH_FASTMODE = "Use fast mode for Batch synth"
    window.i18n.SETTINGS_MICROPHONE = "Microphone"
    window.i18n.SETTINGS_S2S_VOICE = "Speech-to-Speech voice"
    window.i18n.SETTINGS_AUTOGENERATEVOICE = "Automatically generate voice"
    window.i18n.SETTINGS_S2S_PREADJUST_PITCH = "Pre-adjust the input audio average pitch to match the xVASpeech model voice's"
    window.i18n.SETTINGS_S2S_BGNOISE = "Remove background noise from microphone. You need to record a background noise clip first. (requires sox >= v14.4.2) "
    window.i18n.SETTINGS_S2S_RECNOISE = "Record noise"
    window.i18n.SETTINGS_S2S_BGNOISE_STRENGTH = "Noise removal strength (0.2-0.3 recommended)"
    window.i18n.SETTINGS_MODELS_PATH = "models path"
    window.i18n.SETTINGS_OUTPUT_PATH = "output path"
    window.i18n.SETTINGS_RESET_SETTINGS = "Reset Settings"
    window.i18n.SETTINGS_RESET_PATHS = "Reset Paths"

    window.i18n.UPDATES_VERSION = "This app version: 1.0.0"
    window.i18n.THIS_APP_VERSION = "This app version"
    window.i18n.CHECK_FOR_UPDATES = "Check for updates now"
    window.i18n.CANT_REACH_SERVER = "Can't reach server"
    window.i18n.CHECKING_FOR_UPDATES = "Checking for updates..."
    window.i18n.UPDATE_AVAILABLE = "Update available"
    window.i18n.UPTODATE = "Up-to-date."
    window.i18n.UPDATES_LOG = "Updates log:"
    window.i18n.UPDATES_CHECK = "Check for updates now"

    window.i18n.AVAILABLE = "Available"
    window.i18n.PLUGINS = "Plugins"
    window.i18n.PLUGINS_TRUSTED = "Download plugins only from trusted sources"
    window.i18n.PLUGINSH_ENABLED = "Enabled"
    window.i18n.PLUGINSH_ORDER = "Order"
    window.i18n.PLUGINSH_NAME = "Plugin Name"
    window.i18n.PLUGINSH_AUTHOR = "Author"
    window.i18n.PLUGINSH_VERSION = "Plugin Version"
    window.i18n.PLUGINSH_TYPE = "Type"
    window.i18n.PLUGINSH_MINV = "Min App Version"
    window.i18n.PLUGINSH_MAXV = "Max App Version"
    window.i18n.PLUGINSH_DESCRIPTION = "Description"
    window.i18n.PLUGINSH_PLUGINID = "Plugin Id"
    window.i18n.PLUGINS_MOVEUP = "Move Up"
    window.i18n.PLUGINS_MOVEDOWN = "Move Down"
    window.i18n.PLUGINS_APPLY = "Apply"

    window.i18n.APP_INFO = "App info"
    window.i18n.APP_INFO_INSTR_1 = "For instructions on how to use the app, please watch"
    window.i18n.APP_INFO_INSTR_2 = "this short video"
    window.i18n.APP_INFO_INSTR_3 = "showcase on YouTube."
    window.i18n.APP_INFO_INSTR_4 = "You can also view and/or contribute to the community guide on GitHub "
    window.i18n.APP_INFO_INSTR_5 = "here"

    window.i18n.KEYBOARD_REFERENCE = "Keyboard shortcuts reference"
    window.i18n.KEYBOARD_ENTER = "Enter"
    window.i18n.KEYBOARD_ENTER_DO = "Generate the audio"
    window.i18n.KEYBOARD_ESCAPE = "Escape"
    window.i18n.KEYBOARD_ESCAPE_DO = "Close modals and menus"
    window.i18n.KEYBOARD_SPACE = "Space"
    window.i18n.KEYBOARD_SPACE_DO = "Bring focus to the input textarea"
    window.i18n.KEYBOARD_CTRLS = "Ctrl+S"
    window.i18n.KEYBOARD_CTRLS_DO = "Keep sample"
    window.i18n.KEYBOARD_CTRLSHIFTS = "Ctrl+Shift-S"
    window.i18n.KEYBOARD_CTRLSHIFTS_DO = "Keep sample (but with naming prompt)"
    window.i18n.KEYBOARD_YN = "Y/N"
    window.i18n.KEYBOARD_YN_DO = "Yes/No options in prompt modals"
    window.i18n.KEYBOARD_LR = "Left/Right arrows"
    window.i18n.KEYBOARD_LR_DO = "Move left/right along which letter is focused"
    window.i18n.KEYBOARD_SHIFT_LR = "Shift-Left/Right arrows"
    window.i18n.KEYBOARD_SHIFT_LR_DO = "Create multi-letter selection range"
    window.i18n.KEYBOARD_UD = "Up/Down arrows"
    window.i18n.KEYBOARD_UD_DO = "Move pitch up/down for the letter(s) selected"
    window.i18n.KEYBOARD_CTRL_LR = "Ctrl+Left/Right arrows"
    window.i18n.KEYBOARD_CTRL_LR_DO = "Move the sequence-wide pacing slider"
    window.i18n.KEYBOARD_CTRL_UD = "Ctrl+Up/Down arrows"
    window.i18n.KEYBOARD_CTRL_UD_DO = "Pitch increase/decrease buttons"
    window.i18n.KEYBOARD_CTRLSHIFTUD = "Ctrl+Shift+Up/Down arrows"
    window.i18n.KEYBOARD_CTRLSHIFTUD_DO = "Pitch amplify/flatten buttons"

    window.i18n.SUPPORT = "Support"
    window.i18n.SUPPORT_LINK = "You can support development on patreon at this link:"
    window.i18n.SUPPORT_THANKS = "Special thanks:"

    window.i18n.SUPPORT_GAMES = "Search games..."

    window.i18n.EULA_ACCEPT = "I accept the EULA"
    window.i18n.EULA_CLOSE = "Close"

    window.i18n.BATCH_SYNTHESIS = "Batch Synthesis"
    window.i18n.BATCH_SIZE = "Batch Size"
    window.i18n.BATCH_INSTR1 = `Place the .csv batch file(s) into the box below. Click the "Generate sample" button to generate an example .csv file. The mandatory columns are "game_id", "voice_id", and "text". Watch `
    window.i18n.BATCH_INSTR2 = "this short video"
    window.i18n.BATCH_INSTR3 = "for a demo and more instructions."
    window.i18n.BATCH_GEN_SAMPLE = "Generate Sample"
    window.i18n.BATCH_DROPZONE = "Drag and drop .csv files here"

    window.i18n.BATCHH_NUM = "#"
    window.i18n.BATCHH_STATUS = "Status"
    window.i18n.BATCHH_GAME = "Game"
    window.i18n.BATCHH_VOICE = "Voice"
    window.i18n.BATCHH_TEXT = "Text"
    window.i18n.BATCHH_VOCODER = "Vocoder"
    window.i18n.BATCHH_OUTPATH = "Out Path"
    window.i18n.BATCHH_PACING = "Pacing"
    window.i18n.BATCH_ABS_DIR_PLACEHOLDER = "Complete absolute directory path to output"

    window.i18n.BATCH_CLEAR_DIR = "Clear out the directory first"
    window.i18n.BATCH_SKIP = "Skip existing output"
    window.i18n.BATCH_CURRENTLYDOING = "currently doing..."
    window.i18n.BATCH_SYNTHESIZE = "Synthesize Batch"
    window.i18n.BATCH_PAUSE = "Pause"
    window.i18n.BATCH_STOP = "Stop"
    window.i18n.BATCH_CLEAR = "Clear"
    window.i18n.BATCH_OPENOUT = "Open Output"

    window.i18n.S2S_SELECTVOICE = "Select Speech-to-Speech voice"
    window.i18n.S2S_EXPERIMENTAL = "Experimental"
    window.i18n.S2S_EXPERIMENTALNOTE = "The speech to speech feature is in active research/development. All is subject to change, and will be improved over time. This notice will be removed when the research is complete."
    window.i18n.S2S_RECORD_NOTE = "You can record a few seconds of silence to capture background noise to remove from your future recordings."
    window.i18n.S2S_RECORD_NOISE = "Record noise"
    window.i18n.S2S_INSTRUCTIONS = "Pick from the available xVASpeech models which voice sounds most like yours. If you have an xVASpeech model trained for your own voice, make sure you use that one. You can record a short sample here, for comparison. Ensure the volume is also similar. For this to work well, your voice/microphone should sound as similar as possible to the audio of the voice you select."
    window.i18n.S2S_RECORD_SAMPLE = "Record sample"
    window.i18n.FEMALE = "Female"
    window.i18n.MALE = "Male"
    window.i18n.S2S_OTHER = "Other"



    // Dynamic
    window.i18n.SOMETHING_WENT_WRONG = "Something went wrong"
    window.i18n.THERE_WAS_A_PROBLEM = "There was a problem"
    window.i18n.ENTER_DIR_PATH = "Please enter a directory path"
    window.i18n.SURE_RESET_SETTINGS = `Are you sure you'd like to reset your settings?`
    window.i18n.SURE_RESET_PATHS = `Are you sure you'd like to reset your paths? This includes the paths for models, and output.`
    window.i18n.LOAD_MODEL = "Load model"
    window.i18n.LOAD_TARGET_MODEL = "Please load a target voice from the panel on the left, first."
    window.i18n.NO_XVASPEECH_MODELS = "No xVASpeech models are installed"
    window.i18n.ONLY_WAV_S2S = "Only .wav files are supported for speech-to-speech file input at the moment."
    window.i18n.NO_MODELS_IN = "No models in"
    window.i18n.NO_MODELS_FOUND = "No models found"
    window.i18n.MODEL_REQUIRES_VERSION = `This model requires app version`
    window.i18n.OPEN_CONTAINING_FOLDER = "Open containing folder"
    window.i18n.ADJUST_SAMPLE_IN_EDITOR = "Adjust sample in the editor"
    window.i18n.ENTER_NEW_FILENAME_UNCHANGED_CANCEL = "Enter new file name, or submit unchanged to cancel."
    window.i18n.EDIT_IN_EXTERNAL_PROGRAM = "Edit in external program"
    window.i18n.FOLLOWING_PATH_NOT_VALID = "The following program path is not valid"
    window.i18n.SPECIFY_EDIT_TOOL = "Specify your audio editing tool in the settings"
    window.i18n.SURE_DELETE = "Are you sure you'd like to delete this file?"
    window.i18n.LOADING_VOICE = "Loading voice"
    window.i18n.ERR_SERVER = "There was an issue connecting to the python server.<br><br>Try again in a few seconds. If the issue persists, make sure localhost port 8008 is free, or send the server.log file to me on GitHub or Nexus."
    window.i18n.ABOUT_TO_SAVE_FROM_N1_TO_N2_WITH_OPTIONS = `About to save file from _1 to _2 with options`
    window.i18n.SAVING_AUDIO_FILE = "Saving the audio file..."
    window.i18n.TEMP_FILE_NOT_EXIST = "The temporary file does not exist at this file path"
    window.i18n.OUT_DIR_NOT_EXIST = "The output directory does not exist at this file path"
    window.i18n.YOU_CAN_CHANGE_IN_SETTINGS = "You can change this in the settings."
    window.i18n.FILE_EXISTS_ADJUST = `File already exists. Adjust the file name here, or submit without changing to overwrite the old file.`
    window.i18n.ENTER_FILE_NAME = `Enter file name`
    window.i18n.WAVEGLOW_NOT_FOUND = "WaveGlow model not found. Download it also (separate download), and place the .pt file in the models folder."
    window.i18n.ERR_LOADING_MODELS_FOR_GAME = "ERROR loading models for game"
    window.i18n.ERR_LOADING_MODELS_FOR_GAME_WITH_FILENAME = "ERROR loading models for game _1 with filename:"

    window.i18n.CHANGING_MODELS = "Changing models..."
    window.i18n.CHANGING_DEVICE = "Changing device..."
    window.i18n.PROCESSING_DATA = "Procesing data..."
    window.i18n.DELETING_FILE = "Deleting file"
    window.i18n.DELETING_NEW_FILE = "Deleting new file"
    window.i18n.FAILED = "Failed"
    window.i18n.DONE = "Done"
    window.i18n.READY = "Ready"
    window.i18n.RUNNING = "Running"
    window.i18n.PAUSED = "Paused"
    window.i18n.PAUSE = "Pause"
    window.i18n.RESUME = "Resume"
    window.i18n.STOPPED = "Stopped"
    window.i18n.SYNTHESIZING = "Synthesizing"
    window.i18n.LINES = "lines"
    window.i18n.LINE = "Line"
    window.i18n.ERROR = "Error"
    window.i18n.MISSING = "Missing"
    window.i18n.INPUT = "Input"
    window.i18n.OUTPUT = "Output"
    window.i18n.OUTPUTTING = "Outputting"
    window.i18n.SUBMIT = "Submit"
    window.i18n.CLOSE = "Close"
    window.i18n.YES = "Yes"
    window.i18n.NO = "No"
    window.i18n.VOICE = "voice"
    window.i18n.VOICE_PLURAL = "voices"
    window.i18n.NEW = "new"
    window.i18n.LOADING = "Loading"
    window.i18n.MAY_TAKE_A_MINUTE = "May take a minute (but not much more)"
    window.i18n.BUILDING_FASTPITCH = "Building FastPitch model"
    window.i18n.LOADING_WAVEGLOW = "Loading WaveGlow model"
    window.i18n.STARTING_PYTHON = "Starting up the python backend"

    window.i18n.BATCH_CHANGING_MODEL_TO = "Changing voice model to"
    window.i18n.BATCH_CHANGING_VOCODER_TO = "Changing vocoder to"
    window.i18n.BATCH_OUTPUTTING_FFMPEG = `Outputting audio via ffmpeg...`

    window.i18n.BATCH_ERR_NO_VOICES = "No voice models available in the app. Load at least one."
    window.i18n.BATCH_ERR_GAMEID = "does not match any available games"
    window.i18n.BATCH_ERR_VOICEID = "does not match any in the game"
    window.i18n.BATCH_ERR_VOCODER1 = "does not exist. Available options"
    window.i18n.BATCH_ERR_VOCODER2 = "(or leaving it blank)"
    window.i18n.BATCH_ERR_CUDA_OOM = "CUDA OOM: There is not enough VRAM to run this. Try lowering the batch size, or shortening very long sentences."
    window.i18n.BATCH_ERR_IN_PROGRESS = "Batch synthesis is in progress. Loading a model in the main app now would break things."

    window.i18n.ERR_LOADING_PLUGIN = "Error loading plugin"
    window.i18n.SUCCESSFULLY_INITIALIZED = "Successfully initialized"
    window.i18n.FAILED_INIT_FOLLOWING = "Failed to initialize the following"
    window.i18n.CHECK_SERVERLOG = "Check the server.log file for detailed error traces"
    window.i18n.SUCC_NO_ACTIVE_PLUGINS = "Success. No plugins active."
    window.i18n.APP_RESTART_NEEDED = "App restart is required for at least one of the plugins to take effect."
    window.i18n.ERR_LOADING_CSS = "Error loading style file for plugin"
    window.i18n.PLUGIN = "Plugin"
    window.i18n.PLUGINS = "Plugins"
    window.i18n.CANT_IMPORT_FILE_FOR_HOOK_TASK_ENTRYPOINT = "Cannot import _1 file for _2 _3 entry-point"
    window.i18n.ONLY_JS = "Only JavaScript files are supported right now."
    window.i18n.PLUGIN_RUN_ERROR = "Plugin run error at event"
}


window.i18n.updateUI = () => {

    selectedGameDisplay.innerHTML = window.i18n.SELECT_GAME
    voiceSearchInput.placeholder = window.i18n.SEARCH_VOICES
    title.innerHTML = window.i18n.SELECT_VOICE_TYPE
    generateVoiceButton.innerHTML = window.i18n.GENERATE_VOICE
    keepSampleButton.innerHTML = window.i18n.KEEP_SAMPLE

    i18n_length.innerHTML = window.i18n.LENGTH
    resetLetter_btn.innerHTML = window.i18n.RESET_LETTER
    i18n_autoregen.innerHTML = window.i18n.AUTO_REGEN
    i18n_vocoder.innerHTML = window.i18n.VOCODER

    reset_btn.innerHTML = window.i18n.RESET
    amplify_btn.innerHTML = window.i18n.AMPLIFY
    flatten_btn.innerHTML = window.i18n.FLATTEN
    increase_btn.innerHTML = window.i18n.RAISE
    decrease_btn.innerHTML = window.i18n.LOWER
    i18n_pacing.innerHTML = window.i18n.PACING

    i18n_settings.innerHTML = window.i18n.SETTINGS
    i18n_setting_gpu.innerHTML = window.i18n.SETTINGS_GPU
    i18n_setting_autoplay.innerHTML = window.i18n.SETTINGS_AUTOPLAY
    i18n_setting_defaulthifi.innerHTML = window.i18n.SETTINGS_DEFAULT_HIFI
    i18n_setting_keeppacing.innerHTML = window.i18n.SETTINGS_KEEP_PACING
    i18n_setting_tooltip.innerHTML = window.i18n.SETTINGS_TOOLTIP

    i18n_setting_darkmode.innerHTML = window.i18n.SETTINGS_DARKMODE
    i18n_setting_promptfontsize.innerHTML = window.i18n.SETTINGS_PROMPTSIZE
    i18n_setting_bg_fade.innerHTML = window.i18n.SETTINGS_BG_FADE
    i18n_setting_autoreloadvoices.innerHTML = window.i18n.SETTINGS_AUTORELOADVOICES
    i18n_setting_keepeditorstate.innerHTML = window.i18n.SETTINGS_KEEPEDITORSTATE
    i18n_setting_outputjson.innerHTML = window.i18n.SETTINGS_OUTPUTJSON
    i18n_setting_seqnumbering.innerHTML = window.i18n.SETTINGS_SEQNUMBERING
    i18n_setting_external_edit.innerHTML = window.i18n.SETTINGS_EXTERNALEDIT
    i18n_setting_ffmpeg.innerHTML = window.i18n.SETTINGS_FFMPEG
    i18n_setting_ffmpeg_format.innerHTML = window.i18n.SETTINGS_FFMPEG_FORMAT
    i18n_setting_ffmpeg_hz.innerHTML = window.i18n.SETTINGS_FFMPEG_HZ
    i18n_setting_ffmpeg_padstart.innerHTML = window.i18n.SETTINGS_FFMPEG_PADSTART
    i18n_setting_ffmpeg_padend.innerHTML = window.i18n.SETTINGS_FFMPEG_PADEND
    i18n_setting_ffmpeg_bitdepth.innerHTML = window.i18n.SETTINGS_FFMPEG_BITDEPTH
    i18n_setting_ffmpeg_amplitude.innerHTML = window.i18n.SETTINGS_FFMPEG_AMPLITUDE
    i18n_setting_batch_fastmode.innerHTML = window.i18n.SETTINGS_BATCH_FASTMODE
    i18n_setting_microphone.innerHTML = window.i18n.SETTINGS_MICROPHONE
    i18n_setting_s2s_voice.innerHTML = window.i18n.SETTINGS_S2S_VOICE
    i18n_setting_autogeneratevoice.innerHTML = window.i18n.SETTINGS_AUTOGENERATEVOICE
    i18n_setting_s2s_preadjust_pitch.innerHTML = window.i18n.SETTINGS_S2S_PREADJUST_PITCH
    i18n_setting_s2s_bgnoise.innerHTML = window.i18n.SETTINGS_S2S_BGNOISE
    s2s_settingsRecNoiseBtn.innerHTML = window.i18n.SETTINGS_S2S_RECNOISE
    i18n_setting_s2s_bgnoise_strength.innerHTML = window.i18n.SETTINGS_S2S_BGNOISE_STRENGTH
    reset_settings_btn.innerHTML = window.i18n.SETTINGS_RESET_SETTINGS
    reset_paths_btn.innerHTML = window.i18n.SETTINGS_RESET_PATHS
    s2s_selectVoiceBtn.innerHTML = window.i18n.SELECT_VOICE

    updatesVersions.innerHTML = window.i18n.UPDATES_VERSION
    i18n_updateslog.innerHTML = window.i18n.UPDATES_LOG
    checkUpdates.innerHTML = window.i18n.UPDATES_CHECK

    i18n_plugins.innerHTML = window.i18n.PLUGINS
    i18n_plugins_trusted.innerHTML = window.i18n.PLUGINS_TRUSTED

    i18n_pluginsh_enabled.innerHTML = window.i18n.PLUGINSH_ENABLED
    i18n_pluginsh_order.innerHTML = window.i18n.PLUGINSH_ORDER
    i18n_pluginsh_name.innerHTML = window.i18n.PLUGINSH_NAME
    i18n_pluginsh_author.innerHTML = window.i18n.PLUGINSH_AUTHOR
    i18n_pluginsh_version.innerHTML = window.i18n.PLUGINSH_VERSION
    i18n_pluginsh_type.innerHTML = window.i18n.PLUGINSH_TYPE
    i18n_pluginsh_minv.innerHTML = window.i18n.PLUGINSH_MINV
    i18n_pluginsh_maxv.innerHTML = window.i18n.PLUGINSH_MAXV
    i18n_pluginsh_description.innerHTML = window.i18n.PLUGINSH_DESCRIPTION
    i18n_pluginsh_pluginid.innerHTML = window.i18n.PLUGINSH_PLUGINID
    plugins_moveUpBtn.innerHTML = window.i18n.PLUGINS_MOVEUP
    plugins_moveDownBtn.innerHTML = window.i18n.PLUGINS_MOVEDOWN
    plugins_applyBtn.innerHTML = window.i18n.PLUGINS_APPLY

    i18n_appinfo.innerHTML = window.i18n.APP_INFO
    i18n_appinfo_instr_1.innerHTML = window.i18n.APP_INFO_INSTR_1
    i18n_appinfo_instr_2.innerHTML = window.i18n.APP_INFO_INSTR_2
    i18n_appinfo_instr_3.innerHTML = window.i18n.APP_INFO_INSTR_3
    i18n_appinfo_instr_4.innerHTML = window.i18n.APP_INFO_INSTR_4
    i18n_appinfo_instr_5.innerHTML = window.i18n.APP_INFO_INSTR_5

    i18n_keyboard_reference.innerHTML = window.i18n.KEYBOARD_REFERENCE
    i18n_keyboard_enter.innerHTML = window.i18n.KEYBOARD_ENTER
    i18n_keyboard_enter_do.innerHTML = window.i18n.KEYBOARD_ENTER_DO
    i18n_keyboard_escape.innerHTML = window.i18n.KEYBOARD_ESCAPE
    i18n_keyboard_escape_do.innerHTML = window.i18n.KEYBOARD_ESCAPE_DO
    i18n_keyboard_space.innerHTML = window.i18n.KEYBOARD_SPACE
    i18n_keyboard_space_do.innerHTML = window.i18n.KEYBOARD_SPACE_DO
    i18n_keyboard_ctrls.innerHTML = window.i18n.KEYBOARD_CTRLS
    i18n_keyboard_ctrls_do.innerHTML = window.i18n.KEYBOARD_CTRLS_DO
    i18n_keyboard_ctrlshifts.innerHTML = window.i18n.KEYBOARD_CTRLSHIFTS
    i18n_keyboard_ctrlshifts_do.innerHTML = window.i18n.KEYBOARD_CTRLSHIFTS_DO
    i18n_keyboard_yn.innerHTML = window.i18n.KEYBOARD_YN
    i18n_keyboard_yn_do.innerHTML = window.i18n.KEYBOARD_YN_DO
    i18n_keyboard_lr.innerHTML = window.i18n.KEYBOARD_LR
    i18n_keyboard_lr_do.innerHTML = window.i18n.KEYBOARD_LR_DO
    i18n_keyboard_shift_lr.innerHTML = window.i18n.KEYBOARD_SHIFT_LR
    i18n_keyboard_shift_lr_do.innerHTML = window.i18n.KEYBOARD_SHIFT_LR_DO
    i18n_keyboard_ud.innerHTML = window.i18n.KEYBOARD_UD
    i18n_keyboard_ud_do.innerHTML = window.i18n.KEYBOARD_UD_DO
    i18n_keyboard_ctrl_lr.innerHTML = window.i18n.KEYBOARD_CTRL_LR
    i18n_keyboard_ctrl_lr_do.innerHTML = window.i18n.KEYBOARD_CTRL_LR_DO
    i18n_keyboard_ctrl_ud.innerHTML = window.i18n.KEYBOARD_CTRL_UD
    i18n_keyboard_ctrl_ud_do.innerHTML = window.i18n.KEYBOARD_CTRL_UD_DO
    i18n_keyboard_ctrlshiftud.innerHTML = window.i18n.KEYBOARD_CTRLSHIFTUD
    i18n_keyboard_ctrlshiftud_do.innerHTML = window.i18n.KEYBOARD_CTRLSHIFTUD_DO

    i18n_support.innerHTML = window.i18n.SUPPORT
    i18n_support_link.innerHTML = window.i18n.SUPPORT_LINK
    i18n_support_thanks.innerHTML = window.i18n.SUPPORT_THANKS

    searchGameInput.placeholder = window.i18n.SEARCH_GAMES

    i18n_eula_accept.innerHTML = window.i18n.EULA_ACCEPT
    EULA_closeButon.innerHTML = window.i18n.EULA_CLOSE

    i18n_batch_synthesis.innerHTML = window.i18n.BATCH_SYNTHESIS
    i18n_batchsize.innerHTML = window.i18n.BATCH_SIZE
    i18n_batch_instr1.innerHTML = window.i18n.BATCH_INSTR1
    i18n_batch_instr2.innerHTML = window.i18n.BATCH_INSTR2
    i18n_batch_instr3.innerHTML = window.i18n.BATCH_INSTR3
    batch_generateSample.innerHTML = window.i18n.BATCH_GEN_SAMPLE
    batchDropZoneNote.innerHTML = window.i18n.BATCH_DROPZONE

    i18n_batchh_num.innerHTML = window.i18n.BATCHH_NUM
    i18n_batchh_status.innerHTML = window.i18n.BATCHH_STATUS
    i18n_batchh_game.innerHTML = window.i18n.BATCHH_GAME
    i18n_batchh_voice.innerHTML = window.i18n.BATCHH_VOICE
    i18n_batchh_text.innerHTML = window.i18n.BATCHH_TEXT
    i18n_batchh_vocoder.innerHTML = window.i18n.BATCHH_VOCODER
    i18n_batchh_outpath.innerHTML = window.i18n.BATCHH_OUTPATH
    i18n_batchh_pacing.innerHTML = window.i18n.BATCHH_PACING
    batch_outputFolderInput.placeholder = window.i18n.BATCH_ABS_DIR_PLACEHOLDER

    i18n_batch_cleardir.innerHTML = window.i18n.BATCH_CLEAR_DIR
    i18n_batch_skip.innerHTML = window.i18n.BATCH_SKIP
    batch_progressNotes.innerHTML = window.i18n.BATCH_CURRENTLYDOING
    batch_synthesizeBtn.innerHTML = window.i18n.BATCH_SYNTHESIZE
    batch_pauseBtn.innerHTML = window.i18n.BATCH_PAUSE
    batch_stopBtn.innerHTML = window.i18n.BATCH_STOP
    batch_clearBtn.innerHTML = window.i18n.BATCH_CLEAR
    batch_openDirBtn.innerHTML = window.i18n.BATCH_OPENOUT

    i18n_s2s_selectvoice.innerHTML = window.i18n.S2S_SELECTVOICE
    i18n_s2s_experimental.innerHTML = window.i18n.S2S_EXPERIMENTAL
    i18n_s2s_experimental_note.innerHTML = window.i18n.S2S_EXPERIMENTALNOTE
    i18n_s2s_record_note.innerHTML = window.i18n.S2S_RECORD_NOTE
    s2sNoiseRecordSampleBtn.innerHTML = window.i18n.S2S_RECORD_NOISE
    i18n_s2s_instructions.innerHTML = window.i18n.S2S_INSTRUCTIONS
    s2sVLRecordSampleBtn.innerHTML = window.i18n.S2S_RECORD_SAMPLE
    i18n_s2s_female.innerHTML = window.i18n.FEMALE
    i18n_s2s_male.innerHTML = window.i18n.MALE
    i18n_s2s_other.innerHTML = window.i18n.S2S_OTHER
}


window.i18n.setEnglish()
window.i18n.updateUI()
