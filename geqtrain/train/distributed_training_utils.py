import torch
import torch.distributed as dist
import os
import logging
from typing import Dict, Iterable

def setup_distributed_training(rank:int, world_size:int):
    """Initialize the process group for distributed training
    Args:
        rank (int): rank of the current process (i.e. device id assigned to the process)
        world_size (int): number of processes (i.e. number of GPUs that are going to be used for training)
    """
    for device_id in range(world_size):
        # Before init the process group, call torch.cuda.set_device(args.rank) to assign different GPUs to different processes.
        # https://github.com/pytorch/pytorch/issues/18689
        torch.cuda.set_device(device_id)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup_distributed_training(rank):
    logging.info(f"Rank: {rank} | Destroying process group")
    dist.barrier() # might not be needed but it's ok to keep it
    dist.destroy_process_group()

def configure_dist_training(args):

    def get_free_port():
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        addr = s.getsockname()
        s.close()
        return addr[1]

    # Manually set the environment variables for multi-GPU setup
    world_size = args.world_size
    # Number of GPUs/processes to use
    os.environ['WORLD_SIZE'] = str(world_size)
    if args.master_addr:
        os.environ['MASTER_ADDR'] = str(args.master_addr)
    if args.master_port:
        port = get_free_port() if args.master_port == 'rand' else args.master_port
        os.environ['MASTER_PORT'] = str(port)

    # os.environ['TORCH_CPP_LOG_LEVEL'] = 'INFO'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL' # keep this if possible to detect runtime errors, modify only for prod runs

    return world_size


def sync_tensor_across_GPUs(tensor: torch.Tensor, world_size:int, group=None) -> list[torch.Tensor]:
    """Gather tensors with the same number of dimensions but different lengths across multiple GPUs.

    This function must be called by all processes.
    This function gathers tensors from multiple GPUs, ensuring that tensors with different lengths but the same number of dimensions are correctly aggregated. The function is modified from: https://stackoverflow.com/a/78934638.
    Args:
        tensor (torch.Tensor): The tensor to be gathered from each GPU.
        world_size (int): The number of GPUs involved in the gathering process.
        group (optional): The process group to work on. If None, the default process group is used.
    Returns:
        list[torch.Tensor]: A list of gathered tensors. If the tensors are multi-dimensional, they are concatenated along the first dimension. If the tensors are scalar, they are stacked along the first dimension.
    """
    if group is None:
        group = dist.group.WORLD

    # Gather lengths of tensors on each GPU
    shape = torch.as_tensor(tensor.shape, device=tensor.device) # local for each gpu
    shapes = [torch.empty_like(shape) for _ in range(world_size)]
    dist.all_gather(shapes, shape, group=group) # now shapes have been communicated to all gpus

    # Gather data
    inputs = [tensor] * world_size # local input for each gpu

    # create output buffer wrt the shapes of the tensors on the other gpus
    # builds a list of tensors with the same shape as the tensors on the other gpus
    # if tensor is scalar -> then we are handling the a loss func output value
    if tensor.dim() == 0:
        outputs = [
            torch.empty((), dtype=tensor.dtype, device=tensor.device)
            for _ in shapes
        ]
    # if not scalar then we are handling a batched multi-dim-tensor (e.g. node lvl predictions or graph lvl predictions)
    else:
        outputs = [
            torch.empty(*_shape, dtype=tensor.dtype, device=tensor.device)
            for _shape in shapes
        ]

    dist.all_to_all(outputs, inputs, group=group)

    # to gather metrics: create a single tensor with the cat the atoms from the batches of the different processes
    if not tensor.dim() == 0:
        outputs = torch.cat(outputs, dim=0)
        assert sum(s[0] for s in shapes).item() == outputs.shape[0]
        return outputs

    # to gather loss values: create a list of tensors with the loss values from the batches of the different processes
    outputs = torch.stack(outputs, dim=0)
    assert outputs.shape[0] == world_size # one loss val from each gpu
    return outputs


def sync_dic_of_tensors_across_GPUs(tensor_dict: Dict[str, torch.Tensor], world_size:int, keys_to_sync:Iterable[str]=None) -> None:
    """Synchronize a dictionary of tensors across multiple GPUs.
    This function must be called by all processes.
    It synchronizes the tensors in the provided dictionary across all GPUs involved in the distributed training.
    The synchronization is done in place, modifying the original dictionary.

    Args:
        tensor_dict (Dict[str, torch.Tensor]): The dictionary containing tensors to be synchronized.
        keys_to_sync (Iterable[str], optional): The keys of the tensors in the dictionary that need to be synchronized. If None, all tensors in the dictionary will be synchronized.

    Note:
        Default behavior: assume all values indexed via keys_to_sync are torch.tensors that need to be sync
        After synchronization, the content of the dictionary will be rank-dependent. It is recommended to avoid accessing the dictionary content after this step unless necessary.
    """
    # todo: enforce inability to access to content in dict after this step since all content will be rank dependant? option pop everything but keys_to_sync
    if keys_to_sync is None:
        keys_to_sync = tensor_dict.keys()
    for k in keys_to_sync:
        tensor_dict[k] = sync_tensor_across_GPUs(tensor_dict[k], world_size)