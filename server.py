import os
import traceback

PROD = (not (os.getcwd() == "F:\\Speech\\xVA-Synth") and "Plan.todo" in os.getcwd())
# PROD = True
CPU_ONLY = False
CPU_ONLY = True

with open("./DEBUG.txt", "w+") as f:
    f.write(os.getcwd())
with open(f'{"./resources/app" if PROD else "."}/FASTPITCH_LOADING', "w+") as f:
    f.write("")
with open(f'{"./resources/app" if PROD else "."}/WAVEGLOW_LOADING', "w+") as f:
    f.write("")
with open(f'{"./resources/app" if PROD else "."}/SERVER_STARTING', "w+") as f:
    f.write("")

try:
    import numpy
    import pyinstaller_imports

    import logging
    from logging.handlers import RotatingFileHandler
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer
except:
    with open("./DEBUG_err_imports.txt", "w+") as f:
        f.write(traceback.format_exc())

try:
    import torch
except:
    with open("./DEBUG_err_import_torch.txt", "w+") as f:
        f.write(traceback.format_exc())

print("Start")

fastpitch_model = 0

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
    logger.info("New session")
except:
    with open("./DEBUG_err_logger.txt", "w+") as f:
        f.write(traceback.format_exc())

try:
    import fastpitch
except:
    print(traceback.format_exc())
    logger.info(traceback.format_exc())


# User settings
user_settings = {"use_gpu": not CPU_ONLY, "hifi_gan": False}
try:
    with open(f'{"./resources/app" if PROD else "."}/usersettings.csv', "r") as f:
        data = f.read().split("\n")
        head = data[0].split(",")
        values = data[1].split(",")
        for h, hv in enumerate(head):
            user_settings[hv] = values[h]
    if CPU_ONLY:
        user_settings["use_gpu"] = False
    logger.info(str(user_settings))
except:
    pass

def write_settings ():
    with open(f'{"./resources/app" if PROD else "."}/usersettings.csv', "w+") as f:
        head = list(user_settings.keys())
        vals = ",".join([str(user_settings[h]) for h in head])
        f.write("\n".join([",".join(head), vals]))

use_gpu = user_settings["use_gpu"]=="True"
print(f'user_settings, {user_settings}')
try:
    fastpitch_model = fastpitch.init(PROD, use_gpu=use_gpu, hifi_gan=user_settings["hifi_gan"]=="True")
except:
    print(traceback.format_exc())
    logger.info(traceback.format_exc())


# Server
with open("./DEBUG_start.txt", "w+") as f:
    f.write("Starting")

print("Models ready")
logger.info("Models ready")
try:
    os.remove(f'{"./resources/app" if PROD else "."}/WAVEGLOW_LOADING')
except:
    pass


def setDevice (use_gpu):
    global fastpitch_model
    fastpitch_model.device = torch.device('cuda' if use_gpu else 'cpu')
    fastpitch_model = fastpitch_model.to(fastpitch_model.device)

    if fastpitch_model.waveglow is not None:
        fastpitch_model.waveglow.set_device(fastpitch_model.device)
        fastpitch_model.denoiser.set_device(fastpitch_model.device)
    fastpitch_model.hifi_gan.to(fastpitch_model.device)
setDevice(user_settings["use_gpu"])


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
            global fastpitch_model
            logger.info("POST {}".format(self.path))

            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            pitch_durations_text = "POST request for {}".format(self.path)

            print("POST")
            print(self.path)
            logger.info(post_data)

            if self.path == "/setMode":
                hifi_gan = post_data["hifi_gan"]=="qnd"
                user_settings["hifi_gan"] = hifi_gan
                write_settings()

                if not hifi_gan and fastpitch_model.waveglow is None:
                    use_gpu = user_settings["use_gpu"]=="True"
                    fastpitch_model = fastpitch.init_waveglow(use_gpu, fastpitch_model)

            if self.path == "/setDevice":
                use_gpu = post_data["device"]=="gpu"
                setDevice(use_gpu)

                user_settings["use_gpu"] = use_gpu
                write_settings()

            if self.path == "/loadModel":
                ckpt = post_data["model"]
                fastpitch_model = fastpitch.loadModel(fastpitch_model, ckpt=ckpt, n_speakers=post_data["model_speakers"], device=fastpitch_model.device)

            if self.path == "/synthesize":
                text = post_data["sequence"]
                out_path = post_data["outfile"]
                pitch = post_data["pitch"] if "pitch" in post_data else None
                duration = post_data["duration"] if "duration" in post_data else None
                speaker_i = post_data["speaker_i"]
                hifi_gan = post_data["hifi_gan"] if "hifi_gan" in post_data else False
                pitch_data = [pitch, duration]

                pitch_durations_text = fastpitch.infer(user_settings, text, out_path, fastpitch=fastpitch_model, hifi_gan=hifi_gan, speaker_i=speaker_i, pitch_data=pitch_data)

            self._set_response()
            logger.info("pitch_durations_text")
            logger.info(pitch_durations_text)
            self.wfile.write(pitch_durations_text.encode("utf-8"))
        except Exception as e:
            with open("./DEBUG_request.txt", "w+") as f:
                f.write(traceback.format_exc())
                f.write(str(post_data))
            logger.info("Post Error:\n {}".format(repr(e)))
            print(traceback.format_exc())
            logger.info(traceback.format_exc())

server = HTTPServer(("",8008), Handler)
with open("./DEBUG_server_up.txt", "w+") as f:
    f.write("Starting")
print("Server ready")
logger.info("Server ready")
try:
    os.remove(f'{"./resources/app" if PROD else "."}/SERVER_STARTING')
except:
    pass
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
server.server_close()