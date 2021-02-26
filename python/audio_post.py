import os
import ffmpeg
import traceback

def run_audio_post(logger, input, output, options=None):

    try:
        stream = ffmpeg.input(input)

        ffmpeg_options = {"ar": options["hz"]}

        if options["padStart"] or options["padEnd"]:
            ffmpeg_options["af"] = []
            if options["padStart"]:
                ffmpeg_options["af"].append(f'adelay={options["padStart"]}')
            if options["padEnd"]:
                ffmpeg_options["af"].append(f'apad=pad_dur={options["padEnd"]}ms')

            ffmpeg_options["af"] = ",".join(ffmpeg_options["af"])


        if options["bit_depth"]:
            ffmpeg_options["acodec"] = options["bit_depth"]

        if "mp3" in output:
            ffmpeg_options["c:a"] = "libmp3lame"

        ffmpeg_options["filter:a"] = f'volume={options["amplitude"]}'

        stream = ffmpeg.output(stream, output, **ffmpeg_options)

        logger.info("audio options: "+str(options))
        logger.info("ffmpeg command: "+ " ".join(stream.compile()))

        out, err = (ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True))

    except ffmpeg.Error as e:
        logger.info("ffmpeg err: "+ e.stderr.decode('utf8'))
        return e.stderr.decode('utf8')
    except:
        logger.info(traceback.format_exc())
        return traceback.format_exc()

    return ""