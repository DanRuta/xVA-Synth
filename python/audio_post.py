import os
import ffmpeg
import traceback

def run_audio_post(logger, input, output, options=None):

    logger.info("input: " + input)
    logger.info("current dir: " + ", ".join(os.listdir("./")))

    try:
        stream = ffmpeg.input(input)
        stream = ffmpeg.output(stream, output)

        ffmpeg.run(stream)
    except:
        logger.info(traceback.format_exc())
        return traceback.format_exc()

    return ""