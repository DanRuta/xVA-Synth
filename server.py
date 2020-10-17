
with open("./FASTPITCH_LOADING", "w+") as f:
    f.write("")
with open("./WAVEGLOW_LOADING", "w+") as f:
    f.write("")
with open("./SERVER_STARTING", "w+") as f:
    f.write("")

import os
import logging
from logging.handlers import RotatingFileHandler
import json
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

print("Start")

fastpitch_model = 0

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
import pyinstaller_imports

try:
    import fastpitch
except:
    print(traceback.format_exc())
    logger.info(traceback.format_exc())


use_gpu = True
try:
    fastpitch_model = fastpitch.init(use_gpu=use_gpu)
except:
    print(traceback.format_exc())
    logger.info(traceback.format_exc())
print("Models ready")
logger.info("Models ready")
os.remove("./WAVEGLOW_LOADING")

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
        try:
            global fastpitch_model
            logger.info("POST {}".format(self.path))

            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            pitch_durations_text = "POST request for {}".format(self.path)

            print("POST")
            print(self.path)
            print(post_data)
            logger.info(post_data)

            if self.path == "/loadModel":
                fastpitch_model = fastpitch.loadModel(fastpitch_model, ckpt=post_data["model"])

            if self.path == "/synthesize":
                text = post_data["sequence"]
                out_path = post_data["outfile"]
                pitch = post_data["pitch"] if "pitch" in post_data else None
                duration = post_data["duration"] if "duration" in post_data else None
                pitch_data = [pitch, duration]

                pitch_durations_text = fastpitch.infer(text, out_path, fastpitch=fastpitch_model, pitch_data=pitch_data)

            self._set_response()
            logger.info("pitch_durations_text")
            logger.info(pitch_durations_text)
            self.wfile.write(pitch_durations_text.encode("utf-8"))
        except Exception as e:
            logger.info("Post Error:\n {}".format(repr(e)))
            print(traceback.format_exc())
            logger.info(traceback.format_exc())

server = HTTPServer(("",8008), Handler)
print("Server ready")
logger.info("Server ready")
os.remove("./SERVER_STARTING")
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
server.server_close()