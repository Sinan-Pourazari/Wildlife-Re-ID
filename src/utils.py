
class Update_tracker:
    def __init__(self, update_interval, cooldown =40):
        self.update_interval = update_interval
        self.counter_full=False
        self.curr_counter = 0

    def get_state(self):
        return self.counter_full

    def update(self):
        if self.curr_counter < self.update_interval:
            self.curr_counter +=1
        else:
            self.counter_full = True
        
    def reset(self):
        self.counter_full = False
        self.curr_counter = 0