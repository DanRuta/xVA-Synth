import os
import sys
import traceback
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    APP_VERSION = "2.0.0"

    PROD = False
    PROD = True
    CPU_ONLY = False
    CPU_ONLY = True



    # Imports and logger setup
    # ========================
    try:
        import numpy
        import pyinstaller_imports

        import logging
        from logging.handlers import RotatingFileHandler
        import json
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from python.audio_post import run_audio_post, prepare_input_audio, mp_ffmpeg_output
        import ffmpeg
    except:
        print(traceback.format_exc())
        with open("./DEBUG_err_imports.txt", "w+") as f:
            f.write(traceback.format_exc())

    # Pyinstaller hack
    # ================
    try:
        def script_method(fn, _rcb=None):
            return fn
        def script(obj, optimize=True, _frames_up=0, _rcb=None):
            return obj
        import torch.jit
        torch.jit.script_method = script_method
        torch.jit.script = script
        import torch
    except:
        with open("./DEBUG_err_import_torch.txt", "w+") as f:
            f.write(traceback.format_exc())
    # ================

    try:
        logger = logging.getLogger('serverLog')
        logger.setLevel(logging.DEBUG)
        fh = RotatingFileHandler('{}\server.log'.format(os.path.dirname(os.path.realpath(__file__))), maxBytes=5*1024*1024, backupCount=2)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.info(f'New session. Version: {APP_VERSION}. Installation: {"CPU" if CPU_ONLY else "CPU+GPU"}')

        logger.orig_info = logger.info

        def prefixed_log (msg):
            logger.info(f'{logger.logging_prefix}{msg}')


        def set_logger_prefix (prefix=""):
            if len(prefix):
                logger.logging_prefix = f'[{prefix}]: '
                logger.log = prefixed_log
            else:
                logger.log = logger.orig_info

        logger.set_logger_prefix = set_logger_prefix
        logger.set_logger_prefix("")

    except:
        with open("./DEBUG_err_logger.txt", "w+") as f:
            f.write(traceback.format_exc())
        try:
            logger.info(traceback.format_exc())
        except:
            pass
    # ========================


    try:
        from plugins_manager import PluginManager
        plugin_manager = PluginManager(APP_VERSION, PROD, CPU_ONLY, logger)
        logger.info("Plugin manager loaded.")
    except:
        logger.info("Plugin manager FAILED.")
        logger.info(traceback.format_exc())

    plugin_manager.run_plugins(plist=plugin_manager.plugins["start"]["pre"], event="pre start", data=None)



    # User settings
    # =============
    user_settings = {"use_gpu": not CPU_ONLY, "vocoder": "256_waveglow"}
    try:
        with open(f'{"./resources/app" if PROD else "."}/usersettings.csv', "r") as f:
            data = f.read().split("\n")
            head = data[0].split(",")
            values = data[1].split(",")
            for h, hv in enumerate(head):
                user_settings[hv] = values[h]=="True" if values[h] in ["True", "False"] else values[h]
        if CPU_ONLY:
            user_settings["use_gpu"] = False
        logger.info(str(user_settings))
    except:
        logger.info(traceback.format_exc())
        pass

    def write_settings ():
        with open(f'{"./resources/app" if PROD else "."}/usersettings.csv', "w+") as f:
            head = list(user_settings.keys())
            vals = ",".join([str(user_settings[h]) for h in head])
            f.write("\n".join([",".join(head), vals]))

    use_gpu = user_settings["use_gpu"]
    print(f'user_settings, {user_settings}')
    # =============


    # ======================== Models manager
    try:
        from python.models_manager import ModelsManager
        models_manager = ModelsManager(logger, PROD, torch.device("cuda:0") if use_gpu else torch.device("cpu"))
    except:
        logger.info("Models manager failed to initialize")
        logger.info(traceback.format_exc())
    # ========================


    # xVASpeech setup
    # ===============
    xVASpeechModel = 0
    try:
        import xVASpeech
    except:
        print(traceback.format_exc())
        logger.info(traceback.format_exc())
    try:
        print(xVASpeech)
        xVASpeechModel = xVASpeech.init(PROD, use_gpu, logger)
    except:
        print(traceback.format_exc())
        logger.info(traceback.format_exc())
    # ===============



    print("Models ready")
    logger.info("Models ready")


    def setDevice (use_gpu):
        try:
            models_manager.set_device(torch.device('cuda' if use_gpu else 'cpu'))
        except:
            logger.info("MODELS MANAGER FAILED TO LOAD")
            logger.info(traceback.format_exc())
    setDevice(user_settings["use_gpu"])


    # Server
    class Handler(BaseHTTPRequestHandler):
        def _set_response(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()

        def do_GET(self):
            returnString = "[DEBUG] Get request for {}".format(self.path).encode("utf-8")
            logger.info(returnString)
            self._set_response()
            self.wfile.write(returnString)

        def do_POST(self):
            post_data = ""
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = json.loads(self.rfile.read(content_length).decode('utf-8'))
                req_response = "POST request for {}".format(self.path)

                print("POST")
                print(self.path)


                if self.path == "/setVocoder":
                    logger.info("POST {}".format(self.path))
                    logger.info(post_data)
                    vocoder = post_data["vocoder"]
                    modelPath = post_data["modelPath"]
                    user_settings["vocoder"] = vocoder
                    hifi_gan = "waveglow" not in vocoder
                    write_settings()

                    if vocoder=="qnd":
                        req_response = models_manager.load_model("hifigan", f'{"./resources/app" if PROD else "."}/python/hifigan/hifi.pt')
                    elif not hifi_gan:
                        req_response = models_manager.load_model(vocoder, modelPath)

                    req_response = "" if req_response is None else req_response


                if self.path == "/stopServer":
                    logger.info("POST {}".format(self.path))
                    logger.info("STOPPING SERVER")
                    sys.exit()

                if self.path == "/customEvent":
                    logger.info("POST {}".format(self.path))
                    plugin_manager.run_plugins(plist=plugin_manager.plugins["custom-event"], event="custom-event", data=post_data)

                if self.path == "/setDevice":
                    logger.info("POST {}".format(self.path))
                    logger.info(post_data)
                    use_gpu = post_data["device"]=="gpu"
                    setDevice(use_gpu)

                    user_settings["use_gpu"] = use_gpu
                    write_settings()

                if self.path == "/loadModel":
                    logger.info("POST {}".format(self.path))
                    logger.info(post_data)
                    ckpt = post_data["model"]
                    n_speakers = post_data["model_speakers"] if "model_speakers" in post_data else None

                    plugin_manager.run_plugins(plist=plugin_manager.plugins["load-model"]["pre"], event="pre load-model", data=ckpt)
                    models_manager.load_model("fastpitch", ckpt+".pt", n_speakers=n_speakers)
                    plugin_manager.run_plugins(plist=plugin_manager.plugins["load-model"]["post"], event="post load-model", data=ckpt)

                if self.path == "/synthesize":
                    logger.info("POST {}".format(self.path))
                    text = post_data["sequence"]
                    pace = float(post_data["pace"])
                    out_path = post_data["outfile"]
                    pitch = post_data["pitch"] if "pitch" in post_data else None
                    duration = post_data["duration"] if "duration" in post_data else None
                    speaker_i = post_data["speaker_i"] if "speaker_i" in post_data else None
                    vocoder = post_data["vocoder"]
                    pitch_data = [pitch, duration]
                    old_sequence = post_data["old_sequence"] if "old_sequence" in post_data else None

                    # Handle the case where the vocoder remains selected on app start-up, with auto-HiFi turned off, but no setVocoder call is made before synth
                    continue_synth = True
                    if "waveglow" in vocoder:
                        waveglowPath = post_data["waveglowPath"]
                        req_response = models_manager.load_model(vocoder, waveglowPath)
                        if req_response=="ENOENT":
                            continue_synth = False

                    if continue_synth:
                        plugin_manager.run_plugins(plist=plugin_manager.plugins["synth-line"]["pre"], event="pre synth-line", data=post_data)
                        req_response = models_manager.models("fastpitch").infer(user_settings, text, out_path, vocoder=vocoder, \
                            speaker_i=speaker_i, pitch_data=pitch_data, pace=pace, old_sequence=old_sequence)
                        plugin_manager.run_plugins(plist=plugin_manager.plugins["synth-line"]["post"], event="post synth-line", data=post_data)



                if self.path == "/synthesize_batch":
                    linesBatch = post_data["linesBatch"]
                    speaker_i = post_data["speaker_i"]
                    vocoder = post_data["vocoder"]
                    plugin_manager.run_plugins(plist=plugin_manager.plugins["batch-synth-line"]["pre"], event="pre batch-synth-line", data=post_data)
                    try:
                        req_response = models_manager.models("fastpitch").infer_batch(user_settings, linesBatch, vocoder=vocoder, speaker_i=speaker_i)
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            req_response = "CUDA OOM"
                        else:
                            req_response = str(e)
                    post_data["req_response"] = req_response
                    plugin_manager.run_plugins(plist=plugin_manager.plugins["batch-synth-line"]["post"], event="post batch-synth-line", data=post_data)


                if self.path == "/runSpeechToSpeech":
                    logger.info("POST {}".format(self.path))
                    logger.info("post_data")
                    logger.info(post_data)
                    input_path = post_data["input_path"]
                    voiceId = post_data["voiceId"]
                    modelPath = post_data["modelPath"]
                    doPitchShift = post_data["doPitchShift"]
                    removeNoise = post_data["removeNoise"]
                    removeNoiseStrength = post_data["removeNoiseStrength"]

                    final_path = prepare_input_audio(PROD, logger, input_path, removeNoise, removeNoiseStrength)
                    print("final_path", final_path)
                    logger.info("final_path")
                    logger.info(final_path)

                    if xVASpeechModel.voiceId != voiceId:
                        error = xVASpeech.loadModel(logger, APP_VERSION, xVASpeechModel, voiceId, modelPath)
                        if error:
                            req_response = error
                            self._set_response()
                            self.wfile.write(req_response.encode("utf-8"))
                            return


                    text, pitch, durs = xVASpeech.infer(logger, xVASpeechModel, final_path, use_gpu=user_settings["use_gpu"], doPitchShift=doPitchShift)

                    pitch_durations_text = ""
                    pitch_durations_text += ",".join([str(v) for v in pitch])+"\n"
                    pitch_durations_text += ",".join([str(v) for v in durs])+"\n"
                    pitch_durations_text += f'{text.lower()}'

                    req_response = pitch_durations_text


                if self.path == "/batchOutputAudio":
                    logger.info("POST {}".format(self.path))
                    input_paths = post_data["input_paths"]
                    output_paths = post_data["output_paths"]
                    processes = post_data["processes"]
                    options = json.loads(post_data["options"])

                    req_response = mp_ffmpeg_output(logger, processes, input_paths, output_paths, options)


                if self.path == "/outputAudio":
                    isBatchMode = post_data["isBatchMode"]
                    if not isBatchMode:
                        logger.info("POST {}".format(self.path))

                    input_path = post_data["input_path"]
                    output_path = post_data["output_path"]
                    options = json.loads(post_data["options"])
                    # For plugins
                    extraInfo = {}
                    if "extraInfo" in post_data:
                        extraInfo = json.loads(post_data["extraInfo"])
                        extraInfo["audio_options"] = options
                        extraInfo["input_path"] = input_path
                        extraInfo["output_path"] = output_path
                        extraInfo["ffmpeg"] = ffmpeg

                    plugin_manager.run_plugins(plist=plugin_manager.plugins["output-audio"]["pre"], event="pre output-audio", data=extraInfo)
                    req_response = run_audio_post(None if isBatchMode else logger, input_path, output_path, options)
                    plugin_manager.run_plugins(plist=plugin_manager.plugins["output-audio"]["post"], event="post output-audio", data=extraInfo)

                if self.path == "/refreshPlugins":
                    logger.info("POST {}".format(self.path))
                    status = plugin_manager.refresh_active_plugins()
                    logger.info("status")
                    logger.info(status)
                    req_response = ",".join(status)


                if self.path == "/computeEmbsAndDimReduction":
                    logger.info("POST {}".format(self.path))
                    models_manager.init_model("xvarep")
                    embs = models_manager.models("xvarep").reduce_data_dimension(post_data["mappings"], post_data["includeAllVoices"], post_data["onlyInstalled"], post_data["algorithm"])
                    req_response = embs

                if self.path == "/checkReady":
                    req_response = "ready"


                self._set_response()
                self.wfile.write(req_response.encode("utf-8"))
            except Exception as e:
                with open("./DEBUG_request.txt", "w+") as f:
                    f.write(traceback.format_exc())
                    f.write(str(post_data))
                logger.info("Post Error:\n {}".format(repr(e)))
                print(traceback.format_exc())
                logger.info(traceback.format_exc())

    try:
        server = HTTPServer(("",8008), Handler)
    except:
        with open("./DEBUG_server_error.txt", "w+") as f:
            f.write(traceback.format_exc())
        logger.info(traceback.format_exc())
    try:
        plugin_manager.run_plugins(plist=plugin_manager.plugins["start"]["post"], event="post start", data=None)
        print("Server ready")
        logger.info("Server ready")
        server.serve_forever()


    except KeyboardInterrupt:
        pass
    server.server_close()