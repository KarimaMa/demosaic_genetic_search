from collections import deque

class ProcessQueue:
	def __init__(self):
		self.queue = deque()

	def add(self, p):
		self.queue.append(p)

	def take(self):
		return self.queue.popleft()

	def is_empty(self):
		return len(self.queue) == 0

