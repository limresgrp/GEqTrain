import gc
import torch

def clean_cuda(cls=None):
    '''
    cls: ptr to trainer instance, not used
    '''
    gc.collect()
    torch.cuda.empty_cache()
