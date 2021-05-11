import argparse
import time
import zmq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host")
    parser.add_argument("worker_id")
    args = parser.parse_args()
    context = zmq.Context()

    #  Socket to talk to server
    print("Connecting to hello world server %s…" % args.host)
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://%s:5555" % args.host)

    #  Do 10 requests, waiting each time for a response
    for request in range(10):
        print("Sending request %s …" % request)
        request_str = f"Hello from worker {args.worker_id}"
        request = request_str.encode("utf-8")
        socket.send(request)

        #  Get the reply.
        message = socket.recv()
        print("Received reply %s [ %s ]" % (request, message))
        time.sleep(5)