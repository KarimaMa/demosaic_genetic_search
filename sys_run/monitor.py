import time
import signal
from functools import wraps
import errno
import os
import sys
sys.path.append(sys.path[0].split("/")[0])
import logging

class TimeoutError(Exception):
	pass

class Monitor():
	def __init__(self, name, timeout, logger):
		self.name = name
		self.logger = logger
		self.timeout = timeout
		self.logger.info(f"\n--- {name} Timeout : {timeout} ---\n")
	
	def set_error_msg(self, msg):
		self.error_message = msg

	def handle_timeout(self, signum, frame):
		self.logger.info(f"\n---- {self.name} timed out ----")
		self.logger.info(self.error_message)
		self.logger.info("--------------------------------------")
		raise TimeoutError(f"\n---- {self.name} timed out ----")

	def __enter__(self):
		signal.signal(signal.SIGALRM, self.handle_timeout)
		signal.alarm(self.timeout)

	def __exit__(self, type, value, traceback):
		signal.alarm(0)



if __name__ == "__main__":
	import argparse
	import util

	parser = argparse.ArgumentParser("Monitor")
	parser.add_argument('--load_tree_timeout', type=int, default=11)
	parser.add_argument('--mutate_tree_timeout', type=int, default=11)
	parser.add_argument('--lower_tree_timeout', type=int, default=11)
	parser.add_argument('--train_timeout', type=int, default=11)
	parser.add_argument('--save_model_timeout', type=int, default=11)

	args = parser.parse_args()

	log_format = '%(asctime)s %(levelname)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
	  format=log_format, datefmt='%m/%d %I:%M:%S %p')
	util.create_dir("monitortest")
	monitor_logger = util.create_logger('monitor_logger', logging.INFO, log_format, \
	                            os.path.join("monitortest", 'monitor_log'))

	monitor = Monitor("sleep monitor", 3, monitor_logger)
	monitor.set_error_msg("we didn't make it, sad")
	def sleeper(sec):
		print("staring my nappy nap")
		time.sleep(sec)
		print("finished my stupid nap")

	with monitor:
		try:
			sleeper(7)
		except TimeoutError:
			print("sleeper timed out")
		else:
			print("finished sleeping")
	print("doing other stuff now")


