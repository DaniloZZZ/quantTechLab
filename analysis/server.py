#!/usr/bin/python3
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import multiprocessing.dummy as thr

VAL = [0.]

class MyServer(BaseHTTPRequestHandler):
    def _ok(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
    def do_GET(self):
        global VAL
        print('path', self.path)
        portion = self.path[1:]
        try:
            d = float(portion)
            VAL[0] = d
            self._ok()
            self.wfile.write(bytes(f"got {d}", 'utf-8'))
        except ValueError:
            print("fail", self.path)
            self.send_response(400)

    def do_POST(self):
        print( "incomming http: ", self.path )
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        self.send_response(200)
        client.close()
        #import pdb; pdb.set_trace()

def bind(val):
    global VAL
    VAL = val

def start(name="", port=80):
    myServer = HTTPServer((name, port), MyServer)
    print(time.asctime(), "Server Starts - %s:%s" % (name,port))

    p = thr.Process(
            target=lambda : myServer.serve_forever()
            ,args = ()
            )
    try:
        p.start()
    except KeyboardInterrupt:
        pass
        myServer.server_close()
    return p
