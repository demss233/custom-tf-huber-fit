import math

"""
This is a cosine-based 1Cycle learning rate scheduler.
You can wrap this with a Keras Callback to use it directly with model.fit(...).

Note:
- PyTorch has this built-in as `OneCycleLR`, but in TensorFlow you have to roll your own.
- The logic behind cosine annealing and warm-up scheduling is pretty standard — check any solid ML book or link below.

Refs:
- PyTorch CosineAnnealingLR: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
- Keras Callback: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback
"""

class CycleScheduling():
    def __init__(self, max_lr, total_steps, pct_start = 0.3, base_lr = None, final_lr = None):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.step_num = 0

        self.base_lr = base_lr if base_lr else max_lr / 10
        self.final_lr = final_lr if final_lr else max_lr / 100

        self.up_phase = int(self.total_steps * self.pct_start)
        self.down_phase = self.total_steps - self.up_phase
    
    def get_lr(self, logs = None):
        if self.step_num < self.up_phase:
            lr = self._anneal(self.base_lr, self.max_lr, self.step_num / self.up_phase)
        else:
            lr = self._anneal(self.max_lr, self.final_lr, (self.step_num - self.up_phase) / self.down_phase)

        self.step_num += 1
        return lr
    
    def _anneal(self, start, end, pct):
        # Cosine annealing. for f(x) = cos(pi * x), f'(x) = -sin(pi * x) * pi which is decreasing on (0, pi/2) ~ (0, 1..)
        # pct is present on [0, 1]
        cos_out = math.cos(math.pi * pct) + 1
        return end + 0.5 * (start - end) * cos_out
