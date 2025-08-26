# components/distributed.py
import os
import torch
import torch.distributed as dist
from typing import Optional
from torch.utils.data import DistributedSampler, Sampler
from geqtrain.train.sampler import EnsembleDistributedSampler
from geqtrain.nn import DDP

class DistributedManager:
    """A helper class to manage distributed training setup and operations."""
    def __init__(self, config: dict = None):
        config = {} if config is None else config

        if not dist.is_available() or not torch.cuda.is_available() or not config.get('ddp', False):
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_distributed = False
            # Respect user-provided device, otherwise default
            self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            return

        # This logic is portable between SLURM and torchrun
        self.rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
        self.world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
        self.local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
        
        self.is_distributed = self.world_size > 1
        
        # Determine device: priority is user config -> automatic DDP assignment -> default
        if config.get('device'):
            self.device = torch.device(config.get('device'))
        elif self.is_distributed:
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.is_distributed:
            torch.cuda.set_device(self.device)
            dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    def broadcast_object(self, obj, src=0):
        """Broadcasts a python object from source to all other processes."""
        if not self.is_distributed:
            return obj
        
        # DDP requires objects to be in a list for broadcasting
        obj_list = [obj] if self.is_master else [None]
        dist.broadcast_object_list(obj_list, src=src)
        return obj_list[0]

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        model.to(self.device)
        if not self.is_distributed:
            return model
        return DDP(model, device_ids=[self.device.index])

    def get_sampler(self, dataset, shuffle: bool, use_ensemble: bool) -> Optional[Sampler]:
        if not self.is_distributed:
            return None
            
        sampler_class = EnsembleDistributedSampler if use_ensemble else DistributedSampler
        return sampler_class(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle
        )

    def sync_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.is_distributed:
            return tensor
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor / self.world_size

    def sync_dict_of_tensors(self, data_dict: dict, keys: Optional[set] = None):
        if not self.is_distributed:
            return data_dict
            
        keys_to_sync = keys if keys is not None else data_dict.keys()
        
        for key in keys_to_sync:
            if key in data_dict and torch.is_tensor(data_dict[key]):
                tensor = data_dict[key].to(self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                data_dict[key] = tensor / self.world_size
        return data_dict

    def cleanup(self):
        if self.is_distributed:
            dist.destroy_process_group()