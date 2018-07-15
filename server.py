
import eval
import json
from sys import argv
from http.server import BaseHTTPRequestHandler, HTTPServer

print("Start")

model = 0
loaded = False

class Handler(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()

    def do_GET(self):
        self._set_response()
        self.wfile.write("[DEBUG] Get request for {}".format(self.path).encode("utf-8"))

    # def do_POST(self):
    #     content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
    #     post_data = self.rfile.read(content_length) # <--- Gets the data itself
    #     logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
    #             str(self.path), str(self.headers), post_data.decode('utf-8'))

    #     self._set_response()
    #     self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))


    def do_POST(self):

        global model

        content_length = int(self.headers['Content-Length'])
        post_data = json.loads(self.rfile.read(content_length).decode('utf-8'))

        print("POST")
        print(self.path)
        print(post_data)

        if self.path == "/loadModel":
            model = eval.loadModel(model="model.ckpt-{}".format(post_data["model"]), outputs=post_data["outputs"], cmudict=post_data["cmudict"])

        if self.path == "/synthesize":
            eval.syntesize(model, list(map(int, post_data["sequence"].split(","))))

        self._set_response()
        self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))

server = HTTPServer(("",8008), Handler)
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
server.server_close()