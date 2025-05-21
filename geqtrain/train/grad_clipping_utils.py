# from https://github.com/vgsatorras/en_flows/blob/main/utils.py#L70
import numpy as np
import torch

# Gradient clipping
class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def is_empty(self):
        return len(self.items) == 0

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def median(self):
        return np.median(self.items)

    def std(self):
        return np.std(self.items)

@torch.no_grad()
def gradient_clipping(model, gradnorm_queue, max_gradient_norm, is_master):

    if gradnorm_queue.is_empty():
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
        gradnorm_queue.add(float(grad_norm))
        return grad_norm, max_gradient_norm

    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    # max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()
    max_grad_norm = min(max_gradient_norm, gradnorm_queue.median()) # + gradnorm_queue.std() # this should be aroudn 83.85% acceptance (if assuming gradients ~Normal)

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    # if float(grad_norm) > max_grad_norm:
    #     gradnorm_queue.add(float(max_grad_norm))
    # else:
    #     gradnorm_queue.add(float(grad_norm))
    gradnorm_queue.add(float(grad_norm))

    if is_master and float(grad_norm) > max_grad_norm:
        print(f'Clipped gradient with value {grad_norm:.1f} '
              f'while allowed {max_grad_norm:.1f}')
    return grad_norm, max_grad_norm