import os
import shutil
import ffmpeg
import traceback
import subprocess
from pydub import AudioSegment
from lib.ffmpeg_normalize._ffmpeg_normalize import FFmpegNormalize
import platform

import multiprocessing as mp

def mp_ffmpeg_output (PROD, logger, processes, input_paths, output_paths, options):

    workItems = []
    for ip, path in enumerate(input_paths):
        workItems.append([PROD, None, path, output_paths[ip], options])

    workers = processes if processes>0 else max(1, mp.cpu_count()-1)
    workers = min(len(workItems), workers)

    # logger.info("[mp ffmpeg] workers: "+str(workers))

    pool = mp.Pool(workers)
    results = pool.map(processingTask, workItems)
    pool.close()
    pool.join()

    return "\n".join(results)

def processingTask(data):
    return run_audio_post(data[0], data[1], data[2], data[3], data[4]).replace("\n", "<br>")



def run_audio_post(PROD, logger, input, output, options=None):

    ffmpeg_path = 'ffmpeg' if platform.system() == 'Linux' else '{"./resources/app" if PROD else "."}/python/ffmpeg.exe'

    try:
        stream = ffmpeg.input(input)

        ffmpeg_options = {"ar": options["hz"]}

        ffmpeg_options["af"] = []
        if options["padStart"]:
            ffmpeg_options["af"].append(f'adelay={options["padStart"]}')
        if options["padEnd"]:
            ffmpeg_options["af"].append(f'apad=pad_dur={options["padEnd"]}ms')


        # Pitch
        hz = 48000 if ("useSR" in options.keys() and options["useSR"] or "useCleanup" in options.keys() and options["useCleanup"]) else 22050
        ffmpeg_options["af"].append(f'asetrate={hz*(options["pitchMult"])},atempo=1/{options["pitchMult"]}')
        # Tempo
        ffmpeg_options["af"].append(f'atempo={options["tempo"]}')

        ffmpeg_options["af"].append(f'volume={options["amplitude"]}')

        ffmpeg_options["af"].append("adeclip,adeclick")

        if "useNR" in options.keys() and options["useNR"]:
            ffmpeg_options["af"].append(f'afftdn=nr={options["nr"]}:nf={options["nf"]}:tn=0')

        ffmpeg_options["af"] = ",".join(ffmpeg_options["af"])



        if options["bit_depth"]:
            ffmpeg_options["acodec"] = options["bit_depth"]

        if "mp3" in output:
            ffmpeg_options["c:a"] = "libmp3lame"


        if os.path.exists(output):
            try:
                os.remove(output)
            except:
                pass

        output_path = output.replace(".wav", "_temp.wav") if "deessing" in options and options["deessing"]>0 else output
        stream = ffmpeg.output(stream, output_path, **ffmpeg_options)
        out, err = (ffmpeg.run(stream, cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True, overwrite_output=True))

        # The "filter_complex" option can't be used in the same stream as the normal "filter", so have to do two ffmpeg runs
        if "deessing" in options and options["deessing"]>0:
            stream = ffmpeg.input(output_path)
            ffmpeg_options = {}
            ffmpeg_options["filter_complex"] = f'deesser=i={options["deessing"]}:m=0.5:f=0.5:s=o'
            stream = ffmpeg.output(stream, output, **ffmpeg_options)
            out, err = (ffmpeg.run(stream, cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True, overwrite_output=True))
            try:
                os.remove(output_path)
            except:
                pass


    except ffmpeg.Error as e:
        if logger!=None:
            logger.info("ffmpeg err: "+ e.stderr.decode('utf8'))
        return e.stderr.decode('utf8')
    except:
        if logger!=None:
            logger.info(traceback.format_exc())
        return traceback.format_exc().replace("\n", " ")

    return "-"


def prepare (PROD, logger, inputPath, outputPath, removeNoise, removeNoiseStrength):

    ffmpeg_path = 'ffmpeg' if platform.system() == 'Linux' else '{"./resources/app" if PROD else "."}/python/ffmpeg.exe'

    try:
        stream = ffmpeg.input(inputPath)
        ffmpeg_options = {"ar": 22050, "ac": 1}

        stream = ffmpeg.output(stream, outputPath, **ffmpeg_options)
        out, err = (ffmpeg.run(stream, cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True, overwrite_output=True))

        # Remove silence if a silence clip has been provided
        if removeNoise and os.path.exists(f'{"./resources/app" if PROD else "."}/output/silence.wav'):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # Create a silence noise profile if one does not yet exist
            if not os.path.exists(f'{"./resources/app" if PROD else "."}/output/noise_profile_file'):
                command = f'sox {"./resources/app" if PROD else "."}/output/silence.wav -n noiseprof {"./resources/app" if PROD else "."}/output/noise_profile_file'
                sox = subprocess.Popen(command, startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # sox.stdout.close()
                stdout, stderr = sox.communicate()
                stderr = stderr.decode("utf-8")
                if len(stderr):
                    logger.info(f'SOX Command: {command}')
                    logger.info(f'SOX ERROR: {stderr}')
                    return outputPath

            # Remove the background noise
            command = f'sox {outputPath} {outputPath.split(".wav")[0]}_sil.wav noisered {"./resources/app" if PROD else "."}/output/noise_profile_file {removeNoiseStrength}'
            sox = subprocess.Popen(command, startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = sox.communicate()
            stderr = stderr.decode("utf-8")
            if len(stderr):
                logger.info(f'SOX Command: {command}')
                logger.info(f'SOX ERROR: {stderr}')
            else:
                outputPath = f'{outputPath.split(".wav")[0]}_sil.wav'

    except ffmpeg.Error as e:
        logger.info("[prepare] ffmpeg err: "+ e.stderr.decode('utf8'))

    return outputPath

def prepare_input_audio(PROD, logger, path, removeNoise, removeNoiseStrength):

    existing_files_dir = "/".join(path.split("/")[:-1])
    logger.info("existing_files_dir")
    logger.info(existing_files_dir)
    existing_files = [fname for fname in os.listdir("/".join(path.split("/")[:-1])) if fname.startswith("recorded_file_")]

    logger.info("existing_files")
    logger.info(",".join(existing_files))
    for file in existing_files:
        os.remove(f'{existing_files_dir}/{file}')



    output = f'{path.split(".wav")[0]}_prepared.wav'
    logger.info(f'output pre prepare: {output}')
    output = prepare(PROD, logger, path, output, removeNoise, removeNoiseStrength)
    logger.info(f'output post prepare: {output}')


    threshold = -40
    interval = 1

    audio = AudioSegment.from_wav(output)

    # break into chunks
    chunks = [audio[i:i+interval] for i in range(0, len(audio), interval)]
    trimmed_audio = []

    for ci, c in enumerate(chunks):
        if (c.dBFS == float('-inf') or c.dBFS < threshold):
            pass
        else:
            trimmed_audio = chunks[ci:]
            break

    combined_sound = sum(trimmed_audio, AudioSegment.empty())
    combined_sound = combined_sound.set_frame_rate(22050)
    final_path = f'{path.split(".wav")[0]}_post.wav'
    combined_sound.export(final_path, format="wav", bitrate=22050) # parameters=["-ac", "1"]


    final_path = f'{path.split(".wav")[0]}_post.wav'
    # final_path = f'{path.split(".wav")[0]}_prepared.wav'
    # logger.info(f'final_path: {final_path}')

    return final_path

def normalize_audio (input_path, output_path):
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    sp = subprocess.Popen(f'ffmpeg-normalize -ar 22050 "{input_path}" -o "{output_path}"', startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = sp.communicate()
    stderr = stderr.decode("utf-8")

    if len(stderr) and "duration of less than 3 seconds" not in stderr:
        print("stderr", stderr)
        return "stderr: "+ stderr

    return ""




# Python based microphone recording (js libs are too buggy)
# https://github.com/egorsmkv/microphone-recorder/blob/master/record.py
import pyaudio
import wave

def start_microphone_recording (logger, models_manager, root_folder):
    logger.info(f'start_microphone_recording')

    CHUNK = 1024
    FORMAT = pyaudio.paInt16 #paInt8
    CHANNELS = 1
    RATE = 44100 #sample rate
    RECORD_SECONDS = 15
    WAVE_OUTPUT_FILENAME = f'{root_folder}/output/recorded_file.wav'

    if os.path.exists(WAVE_OUTPUT_FILENAME):
        os.remove(WAVE_OUTPUT_FILENAME)

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK) #buffer

    frames = []

    logger.info(f'Starting recording...')


    if os.path.exists(f'{root_folder}/python/temp_stop_recording'):
        os.remove(f'{root_folder}/python/temp_stop_recording')

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):

        if os.path.exists(f'{root_folder}/python/temp_stop_recording'):
            logger.info(f'Detected stop request. Ending recording...')
            os.remove(f'{root_folder}/python/temp_stop_recording')
            break

        data = stream.read(CHUNK)
        frames.append(data) # 2 bytes(16 bits) per channel

    stream.stop_stream()
    stream.close()
    p.terminate()

    logger.info(f'Dumping recording audio to file: {WAVE_OUTPUT_FILENAME}')
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()




def move_recorded_file(PROD, logger, models_manager, root_folder, file_path):
    if not os.path.exists(f'{root_folder}/output/recorded_file.wav'):
        logger.info("Not found audio file")
        import time
        time.sleep(5)
    try:

        models_manager.init_model("deepfilternet2")
        models_manager.models("deepfilternet2").cleanup_audio(f'{root_folder}/output/recorded_file.wav', f'{root_folder}/output/recorded_file_preCleanup.wav')

        # Do audio normalization also
        ffmpeg_path = 'ffmpeg' if platform.system() == 'Linux' else '{"./resources/app" if PROD else "."}/python/ffmpeg.exe'
        ffmpeg_normalize = FFmpegNormalize(
                normalization_type="ebu",
                target_level=-23.0,
                print_stats=False,
                loudness_range_target=7.0,
                true_peak=-2.0,
                offset=0.0,
                dual_mono=False,
                audio_codec=None,
                audio_bitrate=None,
                sample_rate=22050,
                keep_original_audio=False,
                pre_filter=None,
                post_filter=None,
                video_codec="copy",
                video_disable=False,
                subtitle_disable=False,
                metadata_disable=False,
                chapters_disable=False,
                extra_input_options=[],
                extra_output_options=[],
                output_format=None,
                dry_run=False,
                progress=False,
                ffmpeg_exe=ffmpeg_path
            )
        ffmpeg_normalize.ffmpeg_exe = ffmpeg_path
        ffmpeg_normalize.add_media_file(f'{root_folder}/output/recorded_file_preCleanup.wav', file_path)
        ffmpeg_normalize.run_normalization()

    except shutil.SameFileError:
        pass
    except:
        logger.info(traceback.format_exc())