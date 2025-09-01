# components/distributed.py
import logging
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
        self.config = {} if config is None else config
        self._is_initialized = False

        if not dist.is_available() or not torch.cuda.is_available() or not self.config.get('ddp', False):
            self.world_size, self.rank, self.local_rank = 1, 0, 0
            self.is_distributed = False
            self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            return

        # This logic is portable between SLURM and torchrun
        self.rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
        self.world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
        self.local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
        
        self.is_distributed = self.world_size > 1
        
        # Determine device: priority is user config -> automatic DDP assignment -> default
        if self.config.get('device'):
            self.device = torch.device(self.config.get('device'))
        elif self.is_distributed:
            # When SLURM sets CUDA_VISIBLE_DEVICES per process, torch.cuda.device_count()
            # will be 1. local_rank % 1 will always be 0.
            # This correctly assigns the only visible device to the process.
            device_id = self.local_rank % torch.cuda.device_count()
            self.device = torch.device("cuda", device_id)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.init_process_group()

    def init_process_group(self):
        """Initializes the distributed process group."""
        if not self.is_distributed or self._is_initialized:
            return
        torch.cuda.set_device(self.device)
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)
        self._is_initialized = True
        logging.info(f"Initialized process group for rank {self.rank} on device {self.device}.")

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
        """Wraps the model with DDP after it has been moved to the correct device."""
        model.to(self.device)
        if not self.is_distributed:
            return model
        
        find_unused = self.config.get('find_unused_parameters', False)
        if find_unused:
            logging.info(
                "Initializing DDP with `find_unused_parameters=True`. "
                "Note: This may slightly slow down training."
            )
        return DDP(model, find_unused_parameters=find_unused)

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