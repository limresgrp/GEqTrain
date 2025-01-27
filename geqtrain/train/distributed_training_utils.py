import torch
import torch.distributed as dist
import os
import logging


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
    dist.barrier()
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
