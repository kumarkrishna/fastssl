import time

class Timer:
    def __init__(self):
        self.total_time = 0
    
    def start(self):
        self.start_time = time.time()

    def reset(self):
        self.total_time = 0

    def get_curr_runtime(self):
        return time.time() - self.start_time
    
    def end(self):
        self.end_time = time.time()
        self.total_time += self.end_time - self.start_time
        