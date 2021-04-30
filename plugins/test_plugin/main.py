logger = setup["logger"]
isCPUonly = setup["isCPUonly"]

def output_audio_pre(data=None):
    global logger, isCPUonly
    logger.log(f'Output audio pre: {data}')

    hz = data["audio_options"]["hz"]
    padStart = data["audio_options"]["padStart"]
    padEnd = data["audio_options"]["padEnd"]
    bit_depth = data["audio_options"]["bit_depth"]
    amplitude = data["audio_options"]["amplitude"]
    input_path = data["input_path"]
    output_path = data["output_path"]
    game = data["game"]
    voiceId = data["voiceId"]
    voiceName = data["voiceName"]
    inputSequence = data["inputSequence"]
    letters = data["letters"]
    pitch = data["pitch"]
    durations = data["durations"]
    vocoder = data["vocoder"]
    # TODO: amplitude data, when it gets implemented

    logger.log("hz:   "+ hz)
    logger.log("padStart:   "+ str(padStart))
    logger.log("padEnd:   "+ str(padEnd))
    logger.log("bit_depth:   "+ bit_depth)
    logger.log("amplitude:   "+ amplitude)
    logger.log("input_path:   "+ input_path)
    logger.log("output_path:   "+ output_path)
    logger.log("game:   "+ game)
    logger.log("voiceId:   "+ voiceId)
    logger.log("voiceName:   "+ voiceName)
    logger.log("vocoder:   "+ vocoder)
    logger.log("inputSequence:   "+ inputSequence)
    logger.log("letters:   "+ ",".join(letters))
    logger.log("pitch:   "+ ",".join([str(item) for item in pitch]))
    logger.log("durations:   "+ ",".join([str(item) for item in  durations]))





# OPTIONAL
# ========
def setup(data=None):
    logger.log(f'Setting up plugin.')
# ========



register_function(output_audio_pre)
register_function(setup)