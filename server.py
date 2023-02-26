import os
import sys
import traceback
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    PROD = 'xVASynth.exe' in os.listdir(".")

    # Saves me having to do backend re-compilations for every little UI hotfix
    with open(f'{"./resources/app" if PROD else "."}/javascript/script.js', encoding="utf8") as f:
        lines = f.read().split("\n")
        APP_VERSION = lines[1].split('"v')[1].split('"')[0]


    # Imports and logger setup
    # ========================
    try:
        import python.pyinstaller_imports
        import numpy

        import logging
        from logging.handlers import RotatingFileHandler
        import json
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from socketserver     import ThreadingMixIn
        from python.audio_post import run_audio_post, prepare_input_audio, mp_ffmpeg_output, normalize_audio, start_microphone_recording, move_recorded_file
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
        import tqdm
        import regex
    except:
        with open("./DEBUG_err_import_torch.txt", "w+") as f:
            f.write(traceback.format_exc())
    # ================
    CPU_ONLY = not torch.cuda.is_available()

    try:
        logger = logging.getLogger('serverLog')
        logger.setLevel(logging.DEBUG)
        server_log_path = f'{os.path.dirname(os.path.realpath(__file__))}/{"../../../" if PROD else ""}/server.log'
        fh = RotatingFileHandler(server_log_path, maxBytes=2*1024*1024, backupCount=5)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.info(f'New session. Version: {APP_VERSION}. Installation: {"CPU" if CPU_ONLY else "CPU+GPU"} | Prod: {PROD} | Log path: {server_log_path}')

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
        from python.plugins_manager import PluginManager
        plugin_manager = PluginManager(APP_VERSION, PROD, CPU_ONLY, logger)
        active_plugins = plugin_manager.get_active_plugins_count()
        logger.info(f'Plugin manager loaded. {active_plugins} active plugins.')
    except:
        logger.info("Plugin manager FAILED.")
        logger.info(traceback.format_exc())

    plugin_manager.run_plugins(plist=plugin_manager.plugins["start"]["pre"], event="pre start", data=None)


    # ======================== Models manager
    modelsPaths = {}
    try:
        from python.models_manager import ModelsManager
        models_manager = ModelsManager(logger, PROD, device="cpu")
    except:
        logger.info("Models manager failed to initialize")
        logger.info(traceback.format_exc())
    # ========================



    print("Models ready")
    logger.info("Models ready")


    # Server
    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        pass
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
            global modelsPaths
            post_data = ""
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = json.loads(self.rfile.read(content_length).decode('utf-8')) if content_length else {}
                req_response = "POST request for {}".format(self.path)

                print("POST")
                print(self.path)

                # For headless mode
                if self.path == "/setAvailableVoices":
                    modelsPaths = json.loads(post_data["modelsPaths"])
                if self.path == "/getAvailableVoices":
                    models = {}
                    for gameId in modelsPaths.keys():
                        models[gameId] = []

                        modelJSONs = sorted(os.listdir(modelsPaths[gameId]))
                        for fname in modelJSONs:
                            if fname.endswith(".json"):
                                with open(f'{modelsPaths[gameId]}/{fname}', "r") as f:
                                    jsons = f.read()
                                    metadata = json.loads(jsons)

                                    models[gameId].append({
                                        "modelType": metadata["modelType"],
                                        "author": metadata["author"] if "author" in metadata else "",
                                        "emb_size": metadata["emb_size"] if "emb_size" in metadata else 1,
                                        "voiceId": metadata["games"][0]["voiceId"],
                                        "voiceName": metadata["games"][0]["voiceName"],
                                        "gender": metadata["games"][0]["gender"] if "gender" in metadata["games"][0] else "other",
                                        "emb_i": metadata["games"][0]["emb_i"] if "emb_i" in metadata["games"][0] else 0
                                    })
                    req_response = json.dumps(models)


                if self.path == "/setVocoder":
                    logger.info("POST {}".format(self.path))
                    logger.info(post_data)
                    vocoder = post_data["vocoder"]
                    modelPath = post_data["modelPath"]
                    hifi_gan = "waveglow" not in vocoder

                    if vocoder=="qnd":
                        req_response = models_manager.load_model("hifigan", f'{"./resources/app" if PROD else "."}/python/hifigan/hifi.pt')
                    elif not hifi_gan:
                        req_response = models_manager.load_model(vocoder, modelPath)

                    req_response = "" if req_response is None else req_response


                if self.path == "/stopServer":
                    logger.info("POST {}".format(self.path))
                    logger.info("STOPPING SERVER")
                    sys.exit()

                if self.path == "/normalizeAudio":
                    input_path = post_data["input_path"]
                    output_path = post_data["output_path"]
                    req_response = normalize_audio(input_path, output_path)

                if self.path == "/customEvent":
                    logger.info("POST {}".format(self.path))
                    plugin_manager.run_plugins(plist=plugin_manager.plugins["custom-event"], event="custom-event", data=post_data)

                if self.path == "/setDevice":
                    logger.info("POST {}".format(self.path))
                    logger.info(post_data)
                    use_gpu = post_data["device"]=="gpu" or "cuda" in post_data["device"]
                    models_manager.set_device('cuda' if use_gpu else 'cpu')

                if self.path == "/loadModel":
                    logger.info("POST {}".format(self.path))
                    logger.info(post_data)
                    ckpt = post_data["model"]
                    modelType = post_data["modelType"]
                    instance_index = post_data["instance_index"] if "instance_index" in post_data else 0
                    modelType = modelType.lower().replace(".", "_").replace(" ", "")
                    post_data["pluginsContext"] = json.loads(post_data["pluginsContext"])
                    n_speakers = post_data["model_speakers"] if "model_speakers" in post_data else None
                    base_lang = post_data["base_lang"] if "base_lang" in post_data else None


                    plugin_manager.run_plugins(plist=plugin_manager.plugins["load-model"]["pre"], event="pre load-model", data=post_data)
                    models_manager.load_model(modelType, ckpt+".pt", instance_index=instance_index, n_speakers=n_speakers, base_lang=base_lang)
                    plugin_manager.run_plugins(plist=plugin_manager.plugins["load-model"]["post"], event="post load-model", data=post_data)

                    if modelType=="fastpitch1_1":
                        models_manager.models_bank["fastpitch1_1"][instance_index].init_arpabet_dicts()

                if self.path == "/synthesize":
                    logger.info("POST {}".format(self.path))
                    post_data["pluginsContext"] = json.loads(post_data["pluginsContext"])
                    instance_index = post_data["instance_index"] if "instance_index" in post_data else 0


                    # Handle the case where the vocoder remains selected on app start-up, with auto-HiFi turned off, but no setVocoder call is made before synth
                    continue_synth = True
                    if "waveglow" in post_data["vocoder"]:
                        waveglowPath = post_data["waveglowPath"]
                        req_response = models_manager.load_model(post_data["vocoder"], waveglowPath, instance_index=instance_index)
                        if req_response=="ENOENT":
                            continue_synth = False

                    device = post_data["device"] if "device" in post_data else models_manager.device_label
                    models_manager.set_device(device, instance_index=instance_index)

                    if continue_synth:
                        plugin_manager.set_context(post_data["pluginsContext"])
                        plugin_manager.run_plugins(plist=plugin_manager.plugins["synth-line"]["pre"], event="pre synth-line", data=post_data)

                        modelType = post_data["modelType"]
                        text = post_data["sequence"]
                        pace = float(post_data["pace"])
                        out_path = post_data["outfile"]
                        base_lang = post_data["base_lang"] if "base_lang" in post_data else None
                        base_emb = post_data["base_emb"] if "base_emb" in post_data else None
                        pitch = post_data["pitch"] if "pitch" in post_data else None
                        energy = post_data["energy"] if "energy" in post_data else None
                        duration = post_data["duration"] if "duration" in post_data else None
                        speaker_i = post_data["speaker_i"] if "speaker_i" in post_data else None
                        useSR = post_data["useSR"] if "useSR" in post_data else None
                        vocoder = post_data["vocoder"]
                        globalAmplitudeModifier = float(post_data["globalAmplitudeModifier"]) if "globalAmplitudeModifier" in post_data else None
                        pitch_data = [pitch, duration, energy]
                        old_sequence = post_data["old_sequence"] if "old_sequence" in post_data else None

                        req_response = models_manager.models(modelType.lower().replace(".", "_").replace(" ", ""), instance_index=instance_index).infer(plugin_manager, text, out_path, vocoder=vocoder, \
                            speaker_i=speaker_i, pitch_data=pitch_data, pace=pace, old_sequence=old_sequence, globalAmplitudeModifier=globalAmplitudeModifier, base_lang=base_lang, base_emb=base_emb, useSR=useSR)
                        plugin_manager.run_plugins(plist=plugin_manager.plugins["synth-line"]["post"], event="post synth-line", data=post_data)


                if self.path == "/synthesize_batch":
                    post_data["pluginsContext"] = json.loads(post_data["pluginsContext"])

                    plugin_manager.set_context(post_data["pluginsContext"])
                    plugin_manager.run_plugins(plist=plugin_manager.plugins["batch-synth-line"]["pre"], event="pre batch-synth-line", data=post_data)
                    modelType = post_data["modelType"]
                    linesBatch = post_data["linesBatch"]
                    speaker_i = post_data["speaker_i"]
                    vocoder = post_data["vocoder"]
                    outputJSON = post_data["outputJSON"]

                    with torch.no_grad():
                        try:
                            req_response = models_manager.models(modelType.lower().replace(".", "_").replace(" ", "")).infer_batch(plugin_manager, linesBatch, outputJSON=outputJSON, vocoder=vocoder, speaker_i=speaker_i)
                        except RuntimeError as e:
                            if "CUDA out of memory" in str(e):
                                req_response = "CUDA OOM"
                            else:
                                req_response = traceback.format_exc()
                                logger.info(req_response)
                        except:
                            e = traceback.format_exc()
                            if "CUDA out of memory" in str(e):
                                req_response = "CUDA OOM"
                            else:
                                req_response = e
                                logger.info(e)
                    post_data["req_response"] = req_response
                    plugin_manager.run_plugins(plist=plugin_manager.plugins["batch-synth-line"]["post"], event="post batch-synth-line", data=post_data)


                if self.path == "/runSpeechToSpeech":
                    logger.info("POST {}".format(self.path))
                    input_path = post_data["input_path"]
                    style_emb = post_data["style_emb"]
                    options = post_data["options"]
                    audio_out_path = post_data["audio_out_path"]
                    useSR = post_data["useSR"]

                    removeNoise = post_data["removeNoise"]
                    removeNoiseStrength = post_data["removeNoiseStrength"]

                    final_path = prepare_input_audio(PROD, logger, input_path, removeNoise, removeNoiseStrength)

                    models_manager.init_model("speaker_rep")
                    models_manager.load_model("speaker_rep", f'{"./resources/app" if PROD else "."}/python/xvapitch/speaker_rep/speaker_rep.pt')

                    try:
                        models_manager.models("xvapitch").run_speech_to_speech(final_path, audio_out_path.replace(".wav", "_tempS2S.wav"), style_emb, models_manager, plugin_manager, useSR=useSR)

                        data_out = ""
                        req_response = data_out

                        # For use by /outputAudio
                        post_data["input_path"] = audio_out_path.replace(".wav", "_tempS2S.wav")
                        post_data["output_path"] = audio_out_path

                    except RuntimeError:
                        req_response = traceback.format_exc()
                        logger.info(req_response)



                if self.path == "/batchOutputAudio":
                    input_paths = post_data["input_paths"]
                    output_paths = post_data["output_paths"]
                    processes = post_data["processes"]
                    options = json.loads(post_data["options"])
                    # For plugins
                    extraInfo = {}
                    if "extraInfo" in post_data:
                        extraInfo = json.loads(post_data["extraInfo"])
                        extraInfo["pluginsContext"] = json.loads(post_data["pluginsContext"])
                        extraInfo["audio_options"] = options
                        extraInfo["input_paths"] = input_paths
                        extraInfo["output_paths"] = output_paths
                        extraInfo["processes"] = processes
                        extraInfo["ffmpeg"] = ffmpeg

                    plugin_manager.run_plugins(plist=plugin_manager.plugins["mp-output-audio"]["pre"], event="pre mp-output-audio", data=extraInfo)
                    req_response = mp_ffmpeg_output(PROD, logger, processes, input_paths, output_paths, options)
                    plugin_manager.run_plugins(plist=plugin_manager.plugins["mp-output-audio"]["post"], event="post mp-output-audio", data=extraInfo)


                if self.path == "/outputAudio" or self.path == "/runSpeechToSpeech":
                    isBatchMode = post_data["isBatchMode"]
                    if not isBatchMode:
                        logger.info("POST /outputAudio")

                    input_path = post_data["input_path"]
                    output_path = post_data["output_path"]
                    options = json.loads(post_data["options"])
                    # For plugins
                    extraInfo = {}
                    if "extraInfo" in post_data:
                        extraInfo = json.loads(post_data["extraInfo"])
                        extraInfo["pluginsContext"] = json.loads(post_data["pluginsContext"])
                        extraInfo["audio_options"] = options
                        extraInfo["input_path"] = input_path
                        extraInfo["output_path"] = output_path
                        extraInfo["ffmpeg"] = ffmpeg

                    plugin_manager.run_plugins(plist=plugin_manager.plugins["output-audio"]["pre"], event="pre output-audio", data=extraInfo)
                    input_path = post_data["input_path"]
                    output_path = post_data["output_path"]

                    req_response = run_audio_post(PROD, None if isBatchMode else logger, input_path, output_path, options)
                    plugin_manager.run_plugins(plist=plugin_manager.plugins["output-audio"]["post"], event="post output-audio", data=extraInfo)

                if self.path == "/refreshPlugins":
                    logger.info("POST {}".format(self.path))
                    status = plugin_manager.refresh_active_plugins()
                    logger.info("status")
                    logger.info(status)
                    req_response = ",".join(status)


                if self.path == "/getWavV3StyleEmb":
                    logger.info("POST {}".format(self.path))
                    wav_path = post_data["wav_path"]
                    models_manager.init_model("speaker_rep")
                    load_resp = models_manager.load_model("speaker_rep", f'{"./resources/app" if PROD else "."}/python/xvapitch/speaker_rep/speaker_rep.pt')
                    if load_resp=="ENOENT":
                        req_response = "ENOENT"
                    else:
                        style_emb = models_manager.models("speaker_rep").compute_embedding(wav_path).squeeze().cpu().detach().numpy()
                        req_response = ",".join([str(v) for v in style_emb])


                if self.path == "/computeEmbsAndDimReduction":
                    logger.info("POST {}".format(self.path))
                    models_manager.init_model("resemblyzer")
                    embs = models_manager.models("resemblyzer").reduce_data_dimension(post_data["mappings"], post_data["includeAllVoices"], post_data["onlyInstalled"], post_data["algorithm"])
                    req_response = embs

                if self.path == "/checkReady":
                    use_gpu = post_data["device"]=="gpu"
                    modelsPaths = json.loads(post_data["modelsPaths"])
                    models_manager.set_device('cuda' if use_gpu else 'cpu')
                    req_response = "ready"

                if self.path == "/updateARPABet":
                    if "fastpitch1_1" in list(models_manager.models_bank.keys()):
                        models_manager.models_bank["fastpitch1_1"].refresh_arpabet_dicts()

                if self.path == "/start_microphone_recording":
                    start_microphone_recording(logger, f'{"./resources/app" if PROD else "."}')
                    req_response = ""

                if self.path == "/move_recorded_file":
                    file_path = post_data["file_path"]
                    move_recorded_file(PROD, logger, f'{"./resources/app" if PROD else "."}', file_path)

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
        # server = HTTPServer(("",8008), Handler)
        server = ThreadedHTTPServer(("",8008), Handler)
        # Prevent issues with socket reuse
        server.allow_reuse_address = True
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