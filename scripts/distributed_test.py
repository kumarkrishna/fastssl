import os
import torch
import torch.distributed
from torch.distributed import ReduceOp
from torch.utils.data.distributed import DistributedSampler

def setup():
    assert torch.distributed.is_available()
    print("PyTorch Distributed available.")
    print("  Backends:")
    print(f"    Gloo: {torch.distributed.is_gloo_available()}")
    print(f"    NCCL: {torch.distributed.is_nccl_available()}")
    print(f"    MPI:  {torch.distributed.is_mpi_available()}")

    # DDP Job is being run via `srun` on a slurm cluster.
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    # SLURM var -> torch.distributed vars in case needed
    # NOTE: Setting these values isn't exactly necessary, but some code might assume it's
    # being run via torchrun or torch.distributed.launch, so setting these can be a good idea.
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size

def main():
    rank, world_size = setup()
    print(rank, world_size)

if __name__ == "__main__":
    main()
    # world_size = torch.cuda.device_count()
    # mp.spawn(main, args=(),nprocs=world_size)

