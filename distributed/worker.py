import argparse
import time
import zmq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host")
    args = parser.parse_args()
    context = zmq.Context()

    #  Socket to talk to server
    print("Connecting to hello world server %s…" % args.host)
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://%s:5555" % args.host)

    #  Do 10 requests, waiting each time for a response
    for request in range(10):
        print("Sending request %s …" % request)
        socket.send(b"Hello")

        #  Get the reply.
        message = socket.recv()
        print("Received reply %s [ %s ]" % (request, message))
