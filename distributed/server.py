import socket
import time
import zmq

hostname = socket.gethostname()

print("Running master node on %s" % hostname)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:2001")

while True:
    #  Wait for next request from client
    message = socket.recv()
    print("Received request: %s" % message)

    #  Do some 'work'
    time.sleep(1)

    #  Send reply back to client
    socket.send(b"World")
