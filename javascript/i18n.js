
window.i18n = {}

window.i18n.setEnglish = () => {
    window.i18n.SELECT_GAME = "Select Game"
    window.i18n.SEARCH_VOICES = "Search voices..."
    window.i18n.SELECT_VOICE = "Select voice"
    window.i18n.SELECT_VOICE_TYPE = "Select Voice Type"
    window.i18n.KEEP_SAMPLE = "Keep Sample"
    window.i18n.GENERATE_VOICE = "Generate Voice"
    window.i18n.RENAME_THE_FILE = "Rename the file"
    window.i18n.DELETE_FILE = "Delete file"

    window.i18n.PITCH_AND_ENERGY = "Pitch+Energy"
    window.i18n.PITCH = "Pitch"
    window.i18n.ENERGY = "Energy"

    window.i18n.VIEW_IS = "View:"
    window.i18n.ENERGY_IS = "Energy:"
    window.i18n.LENGTH = "Length:"
    window.i18n.RESET_LETTER = "Reset Letter"
    window.i18n.AUTO_REGEN = "Auto regenerate"
    window.i18n.VOCODER = "Vocoder:"

    window.i18n.SEARCH_GAMES = "Search games..."
    window.i18n.SEARCH_SETTINGS = "Search settings..."
    window.i18n.SEARCH_N_VOICES = "Search _ voices..."
    window.i18n.SEARCH_N_GAMES_WITH_N2_VOICES = "Search _1 games with _2 voices..."
    window.i18n.RESET = "Reset"
    window.i18n.AMPLIFY = "Amplify"
    window.i18n.FLATTEN = "Flatten"
    window.i18n.RAISE = "Raise"
    window.i18n.LOWER = "Lower"
    window.i18n.PACING = "Pacing"
    window.i18n.OPEN = "Open"
    window.i18n.DOWNLOAD = "Download"
    window.i18n.VRAM_USAGE = "VRAM usage:"

    window.i18n.SETTINGS = "Settings"
    window.i18n.SETTINGS_GPU = "Use GPU (requires CUDA)"
    window.i18n.SETTINGS_AUTOPLAY = "Autoplay generated audio"
    window.i18n.SETTINGS_DEFAULT_HIFI = "Default to loading the HiFi vocoder on voice change, if available"
    window.i18n.SETTINGS_KEEP_PACING = "Keep the same pacing value on new text generations"
    window.i18n.SETTINGS_TOOLTIP = "Show the sliders tooltip"

    window.i18n.SETTINGS_SHOW_DISCORD = "Show Discord status"
    window.i18n.SETTINGS_DARKMODE = "Dark mode text prompt"
    window.i18n.SETTINGS_PROMPTSIZE = "Text prompt font size"
    window.i18n.SETTINGS_BG_FADE = "Background image fade opacity"
    window.i18n.SETTINGS_AUTORELOADVOICES = "Auto-reload voices on files changes"
    window.i18n.SETTINGS_KEEPEDITORSTATE = "Keep editor state on voice change"
    window.i18n.SETTINGS_OUTPUTJSON = "Output .json (needed for editing)"
    window.i18n.SETTINGS_SEQNUMBERING = "Use sequential numbering for file names"
    window.i18n.SETTINGS_BASE_SPEAKER = "Base app output device"
    window.i18n.SETTINGS_ALT_SPEAKER = "Alternate output device (ctrl+click play)"
    window.i18n.SETTINGS_EXTERNALEDIT = "External program for editing audio"
    window.i18n.SETTINGS_FFMPEG = "Use ffmpeg post-processing (requires ffmpeg >=4.3)"
    window.i18n.SETTINGS_FFMPEG_FORMAT = "Audio format (wav, mp3, etc)"
    window.i18n.SETTINGS_FFMPEG_HZ = "Audio sample rate (Hz)"
    window.i18n.SETTINGS_FFMPEG_PADSTART = "Silence padding start (ms)"
    window.i18n.SETTINGS_FFMPEG_PADEND = "Silence padding end (ms)"
    window.i18n.SETTINGS_FFMPEG_BITDEPTH = "Audio bit depth"
    window.i18n.SETTINGS_FFMPEG_AMPLITUDE = "Amplitude multiplier"
    window.i18n.SETTINGS_BATCH_FASTMODE = "Use fast mode for Batch synth"
    window.i18n.SETTINGS_BATCH_USEMULTIP = "Use multi-processing for batch mode ffmpeg output"
    window.i18n.SETTINGS_BATCH_MULTIPCOUNT = "Number of processes (0 for cpu threads count -1)"
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
    window.i18n.KEYBOARD_ALT_CTRL_LR = "Alt-Ctrl-Left/Right arrows"
    window.i18n.KEYBOARD_ALT_CTRL_LR_DO = "Adjust width of letter selection"
    window.i18n.KEYBOARD_UD = "Up/Down arrows"
    window.i18n.KEYBOARD_UD_DO = "Move pitch up/down for the letter(s) selected"
    window.i18n.KEYBOARD_CTRL_LR = "Ctrl+Left/Right arrows"
    window.i18n.KEYBOARD_CTRL_LR_DO = "Move the sequence-wide pacing slider"
    window.i18n.KEYBOARD_CTRL_UD = "Ctrl+Up/Down arrows"
    window.i18n.KEYBOARD_CTRL_UD_DO = "Pitch increase/decrease buttons"
    window.i18n.KEYBOARD_CTRLSHIFTUD = "Ctrl+Shift+Up/Down arrows"
    window.i18n.KEYBOARD_CTRLSHIFTUD_DO = "Pitch amplify/flatten buttons"
    window.i18n.KEYBOARD_CTRLA = "Ctrl+A"
    window.i18n.KEYBOARD_CTRLA_DO = "Select all editor sequence letters"

    window.i18n.SUPPORT = "Support"
    window.i18n.SUPPORT_LINK = "You can support 'xVASynth' development on patreon"
    window.i18n.SUPPORT_THANKS = "Special thanks:"

    window.i18n.SUPPORT_GAMES = "Search games..."

    window.i18n.EULA_ACCEPT = "I accept the EULA"
    window.i18n.EULA_CLOSE = "Close"

    window.i18n.BATCH_SYNTHESIS = "Batch Synthesis"
    window.i18n.BATCH_SIZE = "Batch Size"
    window.i18n.BATCH_INSTR1 = `Place the .csv batch file(s) into the box below. The mandatory columns are "game_id", "voice_id", and "text", but you can also specify output filename/filepath under "out_path", pacing under "pacing", and vocoder under "vocoder" (Available options: 'hifi', 'quickanddirty', 'waveglow', 'waveglowBIG'). Click the "Generate sample" button to generate an example .csv file if you need one. Watch`
    window.i18n.BATCH_INSTR2 = "this short video"
    window.i18n.BATCH_INSTR3 = "for a demo and more instructions."
    window.i18n.BATCH_GEN_SAMPLE = "Generate Sample"
    window.i18n.BATCH_DROPZONE = "Drag and drop .csv files here"

    window.i18n.BATCHH_NUM = "#"
    window.i18n.BATCHH_STATUS = "Status"
    window.i18n.BATCHH_ACTIONS = "Actions"
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
    window.i18n.S2S_RECORD_NOTE = "You can record a few seconds of silence to capture background noise to remove from your future recordings."
    window.i18n.S2S_RECORD_NOISE = "Record noise"
    window.i18n.S2S_INSTRUCTIONS = "Pick from the available FastPitch 1.1 models which voice sounds most like yours. If you have a model trained for your own voice, make sure you use that one. You can record a short sample here, for comparison. For this to work well, your voice/microphone/volume should sound as similar as possible to the audio of the voice you select."
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
    window.i18n.NO_XVASPEECH_MODELS = "No FastPitch1.1 models are installed"
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
    window.i18n.BATCH_MODEL_NOT_FOUND = "Model not found."
    window.i18n.BATCH_DOWNLOAD_WAVEGLOW = "Download WaveGlow files separately if you haven't, or check the path in the settings."
    window.i18n.ERR_LOADING_MODELS_FOR_GAME = "ERROR loading models for game"
    window.i18n.ERR_LOADING_MODELS_FOR_GAME_WITH_FILENAME = "ERROR loading models for game _1 with filename:"
    window.i18n.ERR_XVASPEECH_MODEL_VERSION = `This xVASpeech model needs minimum app version _1. Your app version:`

    window.i18n.CHANGING_MODELS = "Changing models..."
    window.i18n.CHANGING_DEVICE = "Changing device..."
    window.i18n.PROCESSING_DATA = "Processing data..."
    window.i18n.DELETING_FILE = "Deleting file"
    window.i18n.DELETING_NEW_FILE = "Deleting new file"
    window.i18n.FAILED = "Failed"
    window.i18n.DONE = "Done"
    window.i18n.READY = "Ready"
    window.i18n.RUNNING = "Running"
    window.i18n.PAUSED = "Paused"
    window.i18n.PAUSE = "Pause"
    window.i18n.PLAY = "Play"
    window.i18n.EDIT = "Edit"
    window.i18n.EDIT_IS = "Edit:"
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
    window.i18n.PAGE = "Page:"
    window.i18n.NEXT = "Next"
    window.i18n.PREVIOUS = "Previous"
    window.i18n.LOADING = "Loading"
    window.i18n.MAY_TAKE_A_MINUTE = "May take a minute (but not much more)"
    window.i18n.BUILDING_FASTPITCH = "Building FastPitch model"
    window.i18n.LOADING_WAVEGLOW = "Loading WaveGlow model"
    window.i18n.STARTING_PYTHON = "Starting up the python backend"
    window.i18n.NOT_USING_GPU = "Not using GPU"

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
    window.i18n.BATCH_ERR_EDIT = "Batch synthesis is in progress. Pause or stop it first to enable editor."
    window.i18n.BATCH_ERR_SKIPPEDALL = "No records imported, but _1 were skipped as they already exist."

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

    window.i18n.MONDAY = "Monday"
    window.i18n.TUESDAY = "Tuesday"
    window.i18n.WEDNESDAY = "Wednesday"
    window.i18n.THURSDAY = "Thursday"
    window.i18n.FRIDAY = "Friday"
    window.i18n.SATURDAY = "Saturday"
    window.i18n.SUNDAY = "Sunday"



    window.i18n.TOTD_1 = "You can right-click a voice on the left to hear a preview of the voice"
    window.i18n.TOTD_2 = "You can right-click the microphone icon after a recording, to hear back the audio you recorded/inserted"
    window.i18n.TOTD_3 = "There are a number of keyboard shortcuts you can use. Check the info tab for a reference"
    window.i18n.TOTD_4 = "Check the community guide for tips for how to get the best quality out of the tool. This is linked in the info (i) menu"
    window.i18n.TOTD_5 = "You can create a multi-letter selection in the editor by Ctrl+clicking several letters"
    window.i18n.TOTD_6 = "You can shift-click the 'Keep Sample' button (or Ctrl+Shift+S) to first give your file a custom name before saving"
    window.i18n.TOTD_7 = "You can alt+click editor letters to make a multi-letter selection for the entire word you click on"
    window.i18n.TOTD_8 = "You can drag+drop multiple .csv or .txt files into batch mode"
    window.i18n.TOTD_9 = "You can use .txt files in batch mode instead of .csv files, if you first click a voice in the main app to assign the lines to"
    window.i18n.TOTD_10 = "If you have a compatible NVIDIA GPU, and CUDA installed, you can switch to the CPU+GPU installation. Using the GPU is much faster, especially for batch mode."
    window.i18n.TOTD_11 = "The HiFi-GAN vocoder is normally the best quality, but you can also download and use WaveGlow vocoders, if you'd like."
    window.i18n.TOTD_12 = "If the 'Keep editor state on voice changes' option is ticked on, you can generate a line using one voice, then switch to a different voice, and click the 'Generate Voice' button again to generate a line using the new voice, but using a similar speaking style to the first voice."
    window.i18n.TOTD_13 = "If you set the 'Alternative Output device' to something other than the default device, you can Ctrl-click when playing audio, to have it play on a different speaker. You can couple this with something like Voicemeeter Banana split, to have the app speak for you over the microphone, for voice chat, or other audio recording."
    window.i18n.TOTD_14 = "If you add the path to an audio editing program to the 'External Program for Editing audio' setting, you can open generated audio straight in that program in one click, from the output records on the main page"
    window.i18n.TOTD_15 = "If you install ffmpeg (at least version 4.3), you can automatically directly apply a few different audio post processing tasks on the generated audio. This can include Hz resampling, silence padding to the start and/or end of the audio, bit depth, loudness, and different audio formats. You can also tick on the option to pre-apply these to the temporary preview audio sample."
    window.i18n.TOTD_16 = "You can tick on the 'Fast mode' for batch mode to parallelize the audio generation and the audio output (via ffmpeg for example)"
    window.i18n.TOTD_17 = "You can enable multiprocessing for ffmpeg file output in batch mode, to speed up the output process. This is especially useful if you use a large batch size, and your CPU has plenty of threads. This can be used together with Fast Mode."
    window.i18n.TOTD_18 = "If you're having trouble formatting a .csv file for batch mode, you can change the delimiter in the settings to something else (for example a pipe symbol '|')"
    window.i18n.TOTD_19 = "You can change the folder location of your output files, as well as the models. I'd recommend keeping your model files on an SSD, to reduce the loading time."
    window.i18n.TOTD_20 = "Use the voice embeddings search menu to get a 3D visualisation of all the voices in the app (including some 'officially' trained voices not downloaded yet). You can use this as a reference for voice similarly search, to see what other voices there are, which sound similar to a particular voice."
    window.i18n.TOTD_21 = "You can right click on the points in the 3D voice embeddings visualisation, to hear a preview of that voice. This will only work for the voices you have installed, locally."
    window.i18n.TOTD_22 = "The app is customisable via third-party plugins. Plugins can be managed from the plugins menu, and they can change, or add to the front end app functionality/looks (the UI), as well as the python back-end (the machine learning code). If you're interested in developing such a plugin, there is a full developer reference on the GitHub wiki, here: https://github.com/DanRuta/xvasynth-community-guide"
    window.i18n.TOTD_23 = "If you log into nexusmods.com from within the app, you can check for new and updated voice models on your chosen Nexus pages. You can also endorse these, as well as any plugins configured with a nexus link. If you have a premium membership for the Nexus, you can also download (or batch download) all available voices, and have them installed automatically."
    window.i18n.TOTD_24 = "You can manage the list of Nexus pages to check for voice models by clicking the 'Manage Repos' button in the Nexus menu, or by editing the repositories.txt file"
    window.i18n.TOTD_25 = "You can enable/disable error sounds in the settings. You can also pick a different sound, if you'd prefer something else"
    window.i18n.TOTD_26 = "You can resize the window by dragging one of the bottom corners"
    window.i18n.TOTD_27 = "You can right-click game buttons in the nexus window 'Games' list and voice embeddings 'Games' list, to de-select all other games apart from the one you right-clicked"
    window.i18n.TOTD_28 = "You can right-click the speech-to-speech microphone icon to play back the recorded (or drag+dropped) input audio"
    window.i18n.TOTD_29 = "You can ctrl+click the speech-to-speech microphone icon to quickly bring up the speech-to-speech menu"
    window.i18n.TOTD_30 = "You can either enter the text you want for the speech-to-speech mode, or leave the input text field blank to let the app automatically generate it for you"

    window.i18n.TOTD_NO_UNSEEN = "There are no unseen tips left to show. Untick the 'Only show unseen tips' setting to show all tips."


    window.i18n.LINES_PER_SECOND = "lines per second"
    window.i18n.ETA_FINISHED = "Estimated time until finished:"
    window.i18n.LOGGED_IN_AS = "Logged in as: "
    window.i18n.GAMES = "Games"
    window.i18n.MODELS = "Models"
    window.i18n.SHOW_NEW_UPDATED = "Show only new/updated"
    window.i18n.CHECK_NOW = "Check now"
    window.i18n.MANAGE_REPOS = "Manage repos"
    window.i18n.LOG_IN = "Log in"
    window.i18n.LOG_OUT = "Log out"
    window.i18n.NAME = "Name"
    window.i18n.AUTHOR = "Author"
    window.i18n.VERSION = "Version"
    window.i18n.DATE = "Date"
    window.i18n.TYPE = "Type"
    window.i18n.NOTES = "Notes"
    window.i18n.DOWNLOADING = "Downloading:"
    window.i18n.INSTALLING = "Installing:"
    window.i18n.FINISHED = "Finished:"
    window.i18n.DOWNLOAD_ALL = "Download All"
    window.i18n.REPOSITORIES = "Repositories"
    window.i18n.ADD = "Add"
    window.i18n.REMOVE = "Remove"
    window.i18n.V_EMB_VIS = "Voice embeddings visualiser"
    window.i18n.VOICES = "Voices"
    window.i18n.SHOW = "Show"
    window.i18n.GAME = "Game"
    window.i18n.GENDER = "Gender"
    window.i18n.GENDER_IS = "Gender:"
    window.i18n.GAME_IS = "Game:"
    window.i18n.PREVIEW = "Preview"
    window.i18n.LOAD = "Load"

    window.i18n.VOICE_NAME_IS = "Voice Name:"

    window.i18n.VEMB_INSTR_1 = "Left click drag to rotate"
    window.i18n.VEMB_INSTR_2 = "Right click drag to pan"
    window.i18n.VEMB_INSTR_3 = "Mouse wheel scroll to zoom"
    window.i18n.VEMB_INSTR_4 = "Left click on voice to select"
    window.i18n.VEMB_INSTR_5 = "Right click on voice to play sample"

    window.i18n.MALES = "Males"
    window.i18n.FEMALES = "Females"
    window.i18n.OTHER = "Other"

    window.i18n.SHOW_ONLY_INSTALED = "Show only installed voices"
    window.i18n.KEY_IS = "Key:"
    window.i18n.ALGORITHM = "Algorithm"

    window.i18n.TOTD = "Tip of the day"
    window.i18n.TOTD_SHOW = "Show tip of the day"
    window.i18n.TOTD_SHOW_UNSEEN = "Only show unseen tips"
    window.i18n.TOTD_PREV_TIP = "Previous tip"
    window.i18n.TOTD_NEXT_TIP = "Next tip"

    window.i18n.ENDORSE = "Endorse"
    window.i18n.GET_MORE_VOICES = "Get more voices"

    window.i18n.CURR_INSTALL = "Current installation:"
    window.i18n.CHANGE_TO_GPU = "Change to CPU+GPU"
    window.i18n.CHANGE_TO_CPU = "Change to CPU"
    window.i18n.USE_SOUND_ERR = "Use sound for errors"
    window.i18n.ERR_SOUNDFILE = "Error sound file"
    window.i18n.SHOW_NOW = "Show now"
    window.i18n.SETTINGS_PLAYCHANGEDAUDIO = "Play only changed audio, when regenerating"
    window.i18n.SETTINGS_PREAPPLY_FFMPEG = "(recommended) Pre-apply ffmpeg effects to the preview sample"
    window.i18n.SETTINGS_DOUBLE_AMP_DISPLAY = "Also display amplitude setting in the editor"
    window.i18n.SETTINGS_CSV_DELIMITER = "CSV delimiter"
    window.i18n.SETTINGS_PAGINATION_SIZE_BATCH = "Batch pagination size"
    window.i18n.SETTINGS_PAGINATION_SIZE_ARPABET = "ARPAbet pagination size"
    window.i18n.SETTINGS_GROUP_VOICEID = "Group voices by voiceId and vocoder in preprocessing to minimize model switching"
    window.i18n.SETTINGS_GROUP_VOCODER = "Also do a secondary group by the vocoder - can take long to do with big files (100k+ lines)"

    window.i18n.SEARCH_OUTPUT = "Search output file names..."
    window.i18n.DELETE = "Delete"
    window.i18n.DELETE_ALL = "Delete all"
    window.i18n.DELETE_ALL_FILES_CONFIRM = "Are you sure you'd like to delete all files for this voice? This will delete all _1 files in the following output directory:<br>_2"
    window.i18n.DELETE_ALL_FILES_ERR_NO_FILES = "There are no files in the following output directory:<br>_1"
    window.i18n.SORT_BY = "Sort by"
    window.i18n.ASCENDING = "Ascending"
    window.i18n.DESCENDING = "Descending"
    window.i18n.TIME = "Time"

    window.i18n.ERR_LOGGING_INTO_NEXUS = "Error attempting to log into nexusmods"
    window.i18n.LOGGING_INTO_NEXUS = "Logging into nexusmods (check your browser)..."
    window.i18n.NEXUS_PREMIUM = "Nexus requires premium membership for using their API for file downloads"
    window.i18n.NEXUS_ORIG_ERR = "Original error message"
    window.i18n.FAILED_DOWNLOAD = "Failed to download"
    window.i18n.DONE_INSTALLING = "Done installing"
    window.i18n.CHECKING_NEXUS = "Checking nexusmods.com..."
    window.i18n.NEXUS_NOT_DOWNLOADED_MOD = "You need to first download something from this repo to be able to endorse it."
    window.i18n.NEXUS_TOO_SOON_AFTER_DOWNLOAD = "Nexus requires you to wait at least 15 mins (at the time of writing) before you can endorse."
    window.i18n.NEXUS_IS_OWN_MOD = "Nexus does not allow you to rate your own content."
    window.i18n.YOURS = "Yours"
    window.i18n.NEXUS_ENTER_LINK = "Enter the nexusmods.com link to use as a repository"
    window.i18n.NEXUS_LINK_EXISTS = "This link already exists."

    window.i18n.VEMB_VOICE_NOT_ENABLED = "This voice is not enabled"
    window.i18n.VEMB_NO_PREVIEW = "No preview audio file available"
    window.i18n.VEMB_SELECT_VOICE_FIRST = "Select a voice from the scene below first."
    window.i18n.VEMB_NO_MODEL = "No model file available. Download it if you haven't already."
    window.i18n.VEMB_RECOMPUTING = "Re-computing embeddings and dimensionality reduction on voices. May take a minute the first time, subsequent runs should be instant."

    window.i18n.SETTINGS_FOR_PLUGIN = "Settings for plugin: <i>_1</i>"
    window.i18n.EMBEDDINGS_NEED_AT_LEAST_3 = "You need at least 3 voices to run dimensionality reduction for the plot"



    window.i18n.ARPABET_ERROR_BAD_SYMBOLS = "Found non-ARPAbet symbols: _1"
    window.i18n.ARPABET_ERROR_EMPTY_INPUT = "Words or ARPAbet symbols can't be left empty"
    window.i18n.PAGINATION_X_OF_Y = "_1 of _2"
    window.i18n.ARPABET_CONFIRM_ENABLE_ALL = "Are you sure you'd like to enable ALL words for the following dictionary?<br><br><i>_1</i>"
    window.i18n.ARPABET_CONFIRM_DISABLE_ALL = "Are you sure you'd like to disable ALL words for the following dictionary?<br><br><i>_1</i>"
    window.i18n.ARPABET_CONFIRM_DELETE_WORD = "Are you sure you'd like to delete the following word?<br><br><i>_1</i>"
    window.i18n.ARPABET_CONFIRM_SAME_WORD = "The word '_1' already exists in the following dictionaries:<br><br><i>_2</i><br><br>Are you sure you'd like to add it?"

    window.i18n.ONLY_ENABLED = "Only enabled"

    window.i18n.DICTIONARIES = "Dictionaries"
    window.i18n.SAVE = "Save"
    window.i18n.WORDS = "Words"
    window.i18n.WORD_IS = "Word:"
    window.i18n.WORD = "Word"
    window.i18n.REFERENCE = "Reference"
    window.i18n.SEARCH_WORDS = "Search words..."
    window.i18n.ENABLE_ALL = "Enable All"
    window.i18n.DISABLE_ALL = "Disable All"
    window.i18n.PREV = "Prev"

    window.i18n.ALL = "All"
    window.i18n.MOD_NAME = "Mod name"
    window.i18n.MOD_TITLE = "Mod title"
    window.i18n.SEARCH_NEXUS = "Search Nexus"
    window.i18n.MOD_REPOS_USED = "Mod repos used"
    window.i18n.LINK = "Link"
    window.i18n.ENDORSEMENTS = "Endorsements"
    window.i18n.DOWNLOADS = "Downloads"

    window.i18n.VOICE_ID_IS = "Voice ID:"
    window.i18n.APP_VERSION_IS = "App version:"
    window.i18n.MODEL_VERSION_IS = "Model version:"
    window.i18n.MODEL_TYPE_IS = "Model type:"
    window.i18n.LANGUAGE_IS = "Language:"
    window.i18n.TRAINED_BY_IS = "Trained by:"

    window.i18n.X_WORKSHOP_VOICES_INSTALLED = "_1 workshop voices installed"
    window.i18n.WORKSHOP_GAMES_NOT_RECOGNISED = "The following workshop games were not recognised. Do you have the asset file installed?<i>_1</i>"

    window.i18n.YOU_MUST_BE_LOGGED_IN = "You must be logged in to check what voices there are available on the nexus."


    // Useful during developing, to see if there are any strings left un-i18n-ed
    // Object.keys(window.i18n).forEach(key => {
    //     if (!["setEnglish", "updateUI"].includes(key)) {
    //         window.i18n[key] = ""
    //     }
    // })
}


window.i18n.updateUI = () => {



    i18n_voiceInfo_name.innerHTML = window.i18n.VOICE_NAME_IS
    i18n_voiceInfo_id.innerHTML = window.i18n.VOICE_ID_IS
    i18n_voiceInfo_gender.innerHTML = window.i18n.GENDER_IS
    i18n_voiceInfo_appVersion.innerHTML = window.i18n.APP_VERSION_IS
    i18n_voiceInfo_modelVersion.innerHTML = window.i18n.MODEL_VERSION_IS
    i18n_voiceInfo_modelType.innerHTML = window.i18n.MODEL_TYPE_IS
    i18n_voiceInfo_lang.innerHTML = window.i18n.LANGUAGE_IS
    i18n_voiceInfo_author.innerHTML = window.i18n.TRAINED_BY_IS


    i18n_nexusRepos_mod_name.innerHTML = window.i18n.MOD_NAME
    nexusReposSearchBar.placeholder = window.i18n.MOD_TITLE
    i18n_nexusRepos_all.innerHTML = window.i18n.ALL
    searchNexusButton.innerHTML = window.i18n.SEARCH_NEXUS
    i18n_nexusRepos_game.innerHTML = window.i18n.GAME_IS
    i18n_nexusRepos_modReposUsed.innerHTML = window.i18n.MOD_REPOS_USED

    i18n_nexus_searchh_add.innerHTML = window.i18n.ADD
    i18n_nexus_searchh_link.innerHTML = window.i18n.LINK
    i18n_nexus_searchh_game.innerHTML = window.i18n.GAME
    i18n_nexus_searchh_name.innerHTML = window.i18n.NAME
    i18n_nexus_searchh_author.innerHTML = window.i18n.AUTHOR
    i18n_nexus_searchh_endorsements.innerHTML = window.i18n.ENDORSEMENTS
    i18n_nexus_searchh_downloads.innerHTML = window.i18n.DOWNLOADS

    i18n_nexus_reposUsedh_link.innerHTML = window.i18n.LINK
    i18n_nexus_reposUsedh_game.innerHTML = window.i18n.GAME
    i18n_nexus_reposUsedh_name.innerHTML = window.i18n.NAME
    i18n_nexus_reposUsedh_author.innerHTML = window.i18n.AUTHOR
    i18n_nexus_reposUsedh_endorsements.innerHTML = window.i18n.ENDORSEMENTS
    i18n_nexus_reposUsedh_downloads.innerHTML = window.i18n.DOWNLOADS
    i18n_nexus_reposUsedh_remove.innerHTML = window.i18n.REMOVE



    i18n_arpabet_dictionaries.innerHTML = window.i18n.DICTIONARIES
    i18n_arpabet_words.innerHTML = window.i18n.WORDS
    i18n_arpabet_reference.innerHTML = window.i18n.REFERENCE
    arpabet_word_search_input.placeholder = window.i18n.SEARCH_WORDS
    i18n_arpabet_ckbx_only_enabled.placeholder = window.i18n.ONLY_ENABLED
    i18n_arpabet_word_is.innerHTML = window.i18n.WORD_IS
    arpabet_save.innerHTML = window.i18n.SAVE
    i18n_arpabetWordsListh_word.innerHTML = window.i18n.WORD
    i18n_arpabetWordsListh_delete.innerHTML = window.i18n.DELETE
    arpabet_enableall_button.innerHTML = window.i18n.ENABLE_ALL
    arpabet_disableall_button.innerHTML = window.i18n.DISABLE_ALL
    arpabet_prev_btn.innerHTML = window.i18n.PREV
    arpabet_next_btn.innerHTML = window.i18n.NEXT


    selectedGameDisplay.innerHTML = window.i18n.SELECT_GAME
    voiceSearchInput.placeholder = window.i18n.SEARCH_VOICES
    titleName.innerHTML = window.i18n.SELECT_VOICE_TYPE
    generateVoiceButton.innerHTML = window.i18n.GENERATE_VOICE
    keepSampleButton.innerHTML = window.i18n.KEEP_SAMPLE

    i18n_seq_edit_edit.innerHTML = window.i18n.EDIT_IS
    i18n_seq_edit_view.innerHTML = window.i18n.VIEW_IS
    i18n_energy.innerHTML = window.i18n.ENERGY_IS
    seq_edit_view_pitch_energy.innerHTML = window.i18n.PITCH_AND_ENERGY
    seq_edit_view_pitch.innerHTML = window.i18n.PITCH
    seq_edit_view_energy.innerHTML = window.i18n.ENERGY
    seq_edit_edit_pitch.innerHTML = window.i18n.PITCH
    seq_edit_edit_energy.innerHTML = window.i18n.ENERGY

    i18n_vramUsage.innerHTML = window.i18n.VRAM_USAGE
    i18n_length.innerHTML = window.i18n.LENGTH
    resetLetter_btn.innerHTML = window.i18n.RESET_LETTER
    i18n_autoregen.innerHTML = window.i18n.AUTO_REGEN
    i18n_vocoder.innerHTML = window.i18n.VOCODER

    batch_paginationPrev.innerHTML = window.i18n.PREVIOUS
    batch_paginationNext.innerHTML = window.i18n.NEXT
    i18n_page.innerHTML = window.i18n.PAGE
    i18n_batchLPS.innerHTML = window.i18n.LINES_PER_SECOND
    i18n_etaFinished.innerHTML = window.i18n.ETA_FINISHED
    nexusNameDisplay.innerHTML = window.i18n.LOGGED_IN_AS
    i18n_games.innerHTML = window.i18n.GAMES
    i18n_models.innerHTML = window.i18n.MODELS
    i18n_showNewUpdated.innerHTML = window.i18n.SHOW_NEW_UPDATED
    nexusCheckNow.innerHTML = window.i18n.CHECK_NOW
    nexusManageReposButton.innerHTML = window.i18n.MANAGE_REPOS
    nexusLogInButton.innerHTML = window.i18n.LOG_IN
    i18n_nexush_name.innerHTML = window.i18n.NAME
    i18n_nexush_author.innerHTML = window.i18n.AUTHOR
    i18n_nexush_version.innerHTML = window.i18n.VERSION
    i18n_nexush_date.innerHTML = window.i18n.DATE
    i18n_nexush_type.innerHTML = window.i18n.TYPE
    i18n_nexush_notes.innerHTML = window.i18n.NOTES
    i18n_nexusDownloading.innerHTML = window.i18n.DOWNLOADING
    i18n_nexusInstalling.innerHTML = window.i18n.INSTALLING
    i18n_nexusFinished.innerHTML = window.i18n.FINISHED
    nexusDownloadAllBtn.innerHTML = window.i18n.DOWNLOAD_ALL
    i18n_repositories.innerHTML = window.i18n.REPOSITORIES



    i18n_settings_curr_install.innerHTML = window.i18n.CURR_INSTALL
    setting_change_installation.innerHTML = window.i18n.CHANGE_TO_GPU
    i18n_settings_useSound.innerHTML = window.i18n.USE_SOUND_ERR
    i18n_settings_err_soundfile.innerHTML = window.i18n.ERR_SOUNDFILE
    i18n_settings_showTOTD.innerHTML = window.i18n.TOTD_SHOW
    setting_btnShowTOTD.innerHTML = window.i18n.SHOW_NOW
    i18n_settings_unseenTOTD.innerHTML = window.i18n.TOTD_SHOW_UNSEEN
    i18n_settings_playChangedAudio.innerHTML = window.i18n.SETTINGS_PLAYCHANGEDAUDIO
    i18n_setting_ffmpeg_preapply.innerHTML = window.i18n.SETTINGS_PREAPPLY_FFMPEG
    i18n_settings_doubleAmpDisplay.innerHTML = window.i18n.SETTINGS_DOUBLE_AMP_DISPLAY
    i18n_settings_csv_delimiter.innerHTML = window.i18n.SETTINGS_CSV_DELIMITER
    i18n_settings_paginationSize.innerHTML = window.i18n.SETTINGS_PAGINATION_SIZE_BATCH
    i18n_settings_arpabetPagination.innerHTML = window.i18n.SETTINGS_PAGINATION_SIZE_ARPABET
    i18n_settings_groupVoiceID.innerHTML = window.i18n.SETTINGS_GROUP_VOICEID
    i18n_settings_groupVocoder.innerHTML = window.i18n.SETTINGS_GROUP_VOCODER


    voiceSamplesSearch.placeholder = window.i18n.SEARCH_OUTPUT
    i18n_sortByOutput.innerHTML = window.i18n.SORT_BY
    voiceRecordsOrderByButton.innerHTML = window.i18n.NAME
    voiceRecordsOrderByOrderButton.innerHTML = window.i18n.ASCENDING
    voiceRecordsDeleteAllButton.innerHTML = window.i18n.DELETE_ALL

    i18n_pluginsh_endorse.innerHTML = window.i18n.ENDORSE

    i18n_vembVis.innerHTML = window.i18n.V_EMB_VIS
    i18n_games_vemb.innerHTML = window.i18n.GAMES
    i18n_voices.innerHTML = window.i18n.VOICES
    i18n_vembShow.innerHTML = window.i18n.SHOW
    i18n_vembName.innerHTML = window.i18n.NAME
    i18n_vembGame.innerHTML = window.i18n.GAME
    i18n_vembGender.innerHTML = window.i18n.GENDER
    embeddingsSearchBar.placeholder = window.i18n.SEARCH_VOICES
    nexusSearchBar.placeholder = window.i18n.SEARCH_VOICES
    i18n_voiceName.innerHTML = window.i18n.VOICE_NAME_IS
    i18n_genderIs.innerHTML = window.i18n.GENDER_IS
    i18n_vemb_game.innerHTML = window.i18n.GAME_IS
    embeddingsPreviewButton.innerHTML = window.i18n.PREVIEW
    embeddingsLoadButton.innerHTML = window.i18n.LOAD

    i18n_vemb_instr1.innerHTML = window.i18n.VEMB_INSTR_1
    i18n_vemb_instr2.innerHTML = window.i18n.VEMB_INSTR_2
    i18n_vemb_instr3.innerHTML = window.i18n.VEMB_INSTR_3
    i18n_vemb_instr4.innerHTML = window.i18n.VEMB_INSTR_4
    i18n_vemb_instr5.innerHTML = window.i18n.VEMB_INSTR_5

    i18n_vemb_males.innerHTML = window.i18n.MALES
    i18n_vemb_females.innerHTML = window.i18n.FEMALES
    i18n_vemb_other.innerHTML = window.i18n.OTHER

    i18n_showOnlyInstalled.innerHTML = window.i18n.SHOW_ONLY_INSTALED
    i18n_vemb_keyIs.innerHTML = window.i18n.KEY_IS
    i18n_vemb_game_option.innerHTML = window.i18n.GAME
    i18n_vemb_gender_option.innerHTML = window.i18n.GENDER
    i18n_algorithm.innerHTML = window.i18n.ALGORITHM

    i18n_totd.innerHTML = window.i18n.TOTD
    i18n_totd_show.innerHTML = window.i18n.TOTD_SHOW
    i18n_totd_show_unseen.innerHTML = window.i18n.TOTD_SHOW_UNSEEN
    totdPrevTipBtn.innerHTML = window.i18n.TOTD_PREV_TIP
    totdNextTipBtn.innerHTML = window.i18n.TOTD_NEXT_TIP
    totd_close.innerHTML = window.i18n.CLOSE
    embeddingsCloseHelpUI.innerHTML = window.i18n.CLOSE
    nexusMenuButton.innerHTML = window.i18n.GET_MORE_VOICES



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
    // i18n_setting_tooltip.innerHTML = window.i18n.SETTINGS_TOOLTIP

    i18n_showDiscordStatus.innerHTML = window.i18n.SETTINGS_SHOW_DISCORD
    i18n_setting_darkmode.innerHTML = window.i18n.SETTINGS_DARKMODE
    i18n_setting_promptfontsize.innerHTML = window.i18n.SETTINGS_PROMPTSIZE
    i18n_setting_bg_fade.innerHTML = window.i18n.SETTINGS_BG_FADE
    i18n_setting_autoreloadvoices.innerHTML = window.i18n.SETTINGS_AUTORELOADVOICES
    i18n_setting_keepeditorstate.innerHTML = window.i18n.SETTINGS_KEEPEDITORSTATE
    i18n_setting_outputjson.innerHTML = window.i18n.SETTINGS_OUTPUTJSON
    i18n_setting_seqnumbering.innerHTML = window.i18n.SETTINGS_SEQNUMBERING
    i18n_setting_base_speaker.innerHTML = window.i18n.SETTINGS_BASE_SPEAKER
    i18n_setting_alt_speaker.innerHTML = window.i18n.SETTINGS_ALT_SPEAKER
    i18n_setting_external_edit.innerHTML = window.i18n.SETTINGS_EXTERNALEDIT
    i18n_setting_ffmpeg.innerHTML = window.i18n.SETTINGS_FFMPEG
    i18n_setting_ffmpeg_format.innerHTML = window.i18n.SETTINGS_FFMPEG_FORMAT
    i18n_setting_ffmpeg_hz.innerHTML = window.i18n.SETTINGS_FFMPEG_HZ
    i18n_setting_ffmpeg_padstart.innerHTML = window.i18n.SETTINGS_FFMPEG_PADSTART
    i18n_setting_ffmpeg_padend.innerHTML = window.i18n.SETTINGS_FFMPEG_PADEND
    i18n_setting_ffmpeg_bitdepth.innerHTML = window.i18n.SETTINGS_FFMPEG_BITDEPTH
    i18n_setting_ffmpeg_amplitude.innerHTML = window.i18n.SETTINGS_FFMPEG_AMPLITUDE
    i18n_setting_batch_fastmode.innerHTML = window.i18n.SETTINGS_BATCH_FASTMODE
    i18n_setting_batch_multip.innerHTML = window.i18n.SETTINGS_BATCH_USEMULTIP
    i18n_setting_batch_multip_count.innerHTML = window.i18n.SETTINGS_BATCH_MULTIPCOUNT
    i18n_setting_microphone.innerHTML = window.i18n.SETTINGS_MICROPHONE
    i18n_setting_s2s_voice.innerHTML = window.i18n.SETTINGS_S2S_VOICE
    i18n_setting_autogeneratevoice.innerHTML = window.i18n.SETTINGS_AUTOGENERATEVOICE
    // i18n_setting_s2s_preadjust_pitch.innerHTML = window.i18n.SETTINGS_S2S_PREADJUST_PITCH
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

    i18n_keyboard_alt_ctrl_lr.innerHTML = window.i18n.KEYBOARD_ALT_CTRL_LR
    i18n_keyboard_alt_ctrl_lr_do.innerHTML = window.i18n.KEYBOARD_ALT_CTRL_LR_DO
    i18n_keyboard_ctrla.innerHTML = window.i18n.KEYBOARD_CTRLA
    i18n_keyboard_ctrla_do.innerHTML = window.i18n.KEYBOARD_CTRLA_DO

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
    searchSettingsInput.placeholder = window.i18n.SEARCH_SETTINGS

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
    i18n_batchh_actions.innerHTML = window.i18n.BATCHH_ACTIONS
    i18n_nexush_actions.innerHTML = window.i18n.BATCHH_ACTIONS
    i18n_batchh_game.innerHTML = window.i18n.BATCHH_GAME
    i18n_nexush_game.innerHTML = window.i18n.BATCHH_GAME
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
