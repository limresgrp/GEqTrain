import torch
import torch.distributed as dist
import os
import logging
from typing import Dict, Iterable


def get_distributed_env():
    try:
        # torchrun sets these
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError:
        # fallback to SLURM
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
    return rank, world_size, local_rank

def setup_distributed_training():
    """Initialize the process group for distributed training
    """
    rank, world_size, local_rank = get_distributed_env()

    logging.info(f"--> [Rank {rank}] PRE-INIT: Attempting to initialize process group. "
                 f"World size from env: {world_size}, Rank from env: {rank}")

    # 1. Set the device for the current process *before* initializing the process group.
    #    This is the most critical change.
    device_id = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)

    # 2. Initialize the process group.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Query the actual initialized group to confirm its properties
    actual_world_size = dist.get_world_size()
    actual_rank = dist.get_rank()
    logging.info(f"--> [Rank {rank}] POST-INIT: Process group initialized successfully. "
                 f"Actual World Size: {actual_world_size}, Actual Rank: {actual_rank}")


def cleanup_distributed_training():
    logging.info(f"Destroying process group")
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

    # torchrun automatically sets MASTER_ADDR and MASTER_PORT.
    # We should only set them from `args` if they are not already present in the environment.
    # This makes the script compatible with both torchrun and manual launches.
    if 'MASTER_ADDR' not in os.environ and args.master_addr:
        os.environ['MASTER_ADDR'] = str(args.master_addr)
    
    if 'MASTER_PORT' not in os.environ and args.master_port:
        port = get_free_port() if args.master_port == 'rand' else args.master_port
        os.environ['MASTER_PORT'] = str(port)

    # os.environ['TORCH_CPP_LOG_LEVEL'] = 'INFO'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL' # keep this if possible to detect runtime errors, modify only for prod runs


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


def sync_dict_of_tensors_across_GPUs(tensor_dict: Dict[str, torch.Tensor], world_size:int, keys_to_sync:Iterable[str]=None) -> None:
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

    sorted_keys = sorted(keys_to_sync) # needs sorting to ensure that all gpus will try to sync for the same k
    for k in sorted_keys:
        tensor_dict[k] = sync_tensor_across_GPUs(tensor_dict[k], world_size)

def check_dataloader_split(trainer):
    """Gathers sampler indices from all ranks to ensure they are unique."""
    if not dist.is_initialized():
        return
    
    # Each rank gets the list of indices its sampler will iterate over
    # Note: list(sampler) consumes the sampler for one epoch, so this is for debugging
    sampler_indices = torch.tensor(list(trainer.dl_train.sampler), device=trainer.torch_device)

    # Gather the size of each rank's index list to rank 0
    local_size = torch.tensor([len(sampler_indices)], device=trainer.torch_device)
    all_sizes = [torch.tensor([0], device=trainer.torch_device) for _ in range(trainer.world_size)]
    if trainer.rank == 0:
        dist.gather(local_size, gather_list=all_sizes, dst=0)
        all_sizes = [size.item() for size in all_sizes]
    else:
        dist.gather(local_size, gather_list=[], dst=0)
    
    # Broadcast the sizes so all ranks know how much to receive
    dist.broadcast_object_list([all_sizes], src=0)
    
    # Now gather all indices to rank 0
    if trainer.rank == 0:
        gathered_indices = [torch.empty(size, dtype=sampler_indices.dtype, device=trainer.torch_device) for size in all_sizes]
        dist.gather(sampler_indices, gather_list=gathered_indices, dst=0)
    else:
        dist.gather(sampler_indices, gather_list=[], dst=0)

    # Rank 0 performs the verification
    if trainer.rank == 0:
        logging.info("--- Verifying Dataloader Split ---")
        total_indices_processed = sum(all_sizes)
        logging.info(f"Total indices from all samplers: {total_indices_processed}")
        logging.info(f"Original dataset size: {len(trainer.dataset_train)}")
        
        all_indices_flat = torch.cat(gathered_indices).cpu().numpy()
        unique_indices = set(all_indices_flat)

        if len(unique_indices) == total_indices_processed:
            logging.info("SUCCESS: All sampler indices are unique across all ranks.")
        else:
            logging.error(f"ERROR: Found {total_indices_processed - len(unique_indices)} duplicate indices among ranks!")
            
        if total_indices_processed >= len(trainer.dataset_train) or trainer.dl_train.sampler.drop_last:
             logging.info("SUCCESS: The samplers cover the full dataset (or drop_last=True).")
        else:
             logging.warning(f"WARNING: Samplers only cover {total_indices_processed}/{len(trainer.dataset_train)} of the dataset. Is drop_last=False?")
        logging.info("------------------------------------")

def check_gradient_synchronization(trainer):
    """Gathers a gradient from each rank and checks if they are identical."""
    if not dist.is_initialized():
        return

    # 1. Pick a representative parameter to check (e.g., the bias of the last layer)
    # The name will be slightly different because of the DDP wrapper
    param_name = None
    for name, param in trainer.model.named_parameters():
        if "final_pooling.bias" in name: # Using a known small parameter
            param_name = name
            break
    
    if param_name is None:
        logging.warning("Could not find a suitable parameter to check for gradient sync.")
        return

    # 2. Get the gradient tensor for this parameter on the current rank
    local_grad = None
    for name, param in trainer.model.named_parameters():
        if name == param_name:
            local_grad = param.grad
            break

    # 3. Gather all gradient tensors to rank 0
    if trainer.rank == 0:
        grad_list = [torch.empty_like(local_grad) for _ in range(trainer.world_size)]
        dist.gather(local_grad, gather_list=grad_list, dst=0)
    else:
        dist.gather(local_grad, gather_list=[], dst=0)
    
    # 4. Rank 0 performs the verification
    if trainer.rank == 0:
        logging.info(f"--- Verifying Gradient Synchronization (checking '{param_name}') ---")
        all_synced = True
        for i in range(1, trainer.world_size):
            if not torch.allclose(grad_list[0], grad_list[i]):
                logging.error(f"ERROR: Gradients on Rank 0 and Rank {i} DO NOT match!")
                all_synced = False
                break
        
        if all_synced:
            logging.info("SUCCESS: Gradients are correctly synchronized across all ranks.")
        logging.info("-------------------------------------------------")