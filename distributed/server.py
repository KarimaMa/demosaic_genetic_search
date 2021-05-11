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

    msg_str = message.decode("utf-8")
    print("Received request: %s" % message)
    worker_id = msg_str.split(" ")[-1]
    #  Do some 'work'
    time.sleep(1)

    #  Send reply back to client
    reply_str = f"World to worker {worker_id}"
    reply = reply_str.encode("utf-8")
    socket.send(reply)
