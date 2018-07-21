
import os
import eval
import logging
import json
import traceback
from sys import argv
from http.server import BaseHTTPRequestHandler, HTTPServer

print("Start")

model = 0

logger = logging.getLogger('serverLog')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('{}\server.log'.format(os.path.dirname(os.path.realpath(__file__))))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info("New session")

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
            global model
            logger.info("POST {}".format(self.path))

            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length).decode('utf-8'))

            print("POST")
            print(self.path)
            print(post_data)
            logger.info(post_data)

            if self.path == "/loadModel":
                model = eval.loadModel(model=post_data["model"], outputs=post_data["outputs"], cmudict=post_data["cmudict"])

            if self.path == "/synthesize":
                eval.syntesize(model, list(map(int, post_data["sequence"].split(","))), post_data["outfile"])

            self._set_response()
            self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))
        except Exception as e:
            logger.info("Post Error:\n {}".format(repr(e)))
            logger.info(traceback.format_exc())

server = HTTPServer(("",8008), Handler)
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
server.server_close()