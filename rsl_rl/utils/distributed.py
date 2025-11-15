import os
from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    """Container for distributed training metadata."""

    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    distributed: bool = False
    device: torch.device = torch.device("cpu")

    @property
    def is_distributed(self) -> bool:
        return self.distributed

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


_DIST_CONTEXT: Optional[DistributedContext] = None


def init_distributed(cpu: bool = False, backend: Optional[str] = None) -> DistributedContext:
    """Initialise torch.distributed (if needed) and return a context object."""
    global _DIST_CONTEXT
    if _DIST_CONTEXT is not None:
        return _DIST_CONTEXT

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    device_index = _determine_device_index(local_rank)
    distributed = world_size > 1

    if distributed:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available but WORLD_SIZE > 1.")
        if not cpu:
            torch.cuda.set_device(device_index)
        backend = backend or ("gloo" if cpu else "nccl")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device = torch.device("cpu" if cpu else f"cuda:{device_index}")
    _DIST_CONTEXT = DistributedContext(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed=distributed,
        device=device,
    )
    return _DIST_CONTEXT


def _determine_device_index(local_rank: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return local_rank
    visible = visible.strip()
    if visible == str(local_rank):
        return 0
    return local_rank


def get_context() -> DistributedContext:
    """Return the current distributed context (initialising with defaults if needed)."""
    global _DIST_CONTEXT
    if _DIST_CONTEXT is None:
        _DIST_CONTEXT = DistributedContext()
    return _DIST_CONTEXT


def cleanup():
    """Tear down the process group."""
    global _DIST_CONTEXT
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    _DIST_CONTEXT = None


def barrier():
    ctx = get_context()
    if ctx.is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()


def split_num_envs(total_envs: int, ctx: DistributedContext) -> int:
    """Shard a total number of environments across ranks."""
    if not ctx.is_distributed:
        return total_envs
    base = total_envs // ctx.world_size
    remainder = total_envs % ctx.world_size
    local = base + (1 if ctx.rank < remainder else 0)
    if local == 0:
        raise ValueError(
            f"Rank {ctx.rank} would receive 0 envs. Increase env count ({total_envs}) or reduce world size ({ctx.world_size})."
        )
    return local


def _as_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.tensor(value, device=device, dtype=torch.float32)


def distributed_sum(value: Any, device: Optional[torch.device] = None) -> float:
    ctx = get_context()
    tensor = _as_tensor(value, device or ctx.device)
    if ctx.is_distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def distributed_mean(value: Any, device: Optional[torch.device] = None) -> float:
    ctx = get_context()
    tensor = _as_tensor(value, device or ctx.device)
    if ctx.is_distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= ctx.world_size
    return float(tensor.item())


def distributed_max(value: Any, device: Optional[torch.device] = None) -> float:
    ctx = get_context()
    tensor = _as_tensor(value, device or ctx.device)
    if ctx.is_distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def broadcast_object(obj: Any, src: int = 0) -> Any:
    ctx = get_context()
    if not ctx.is_distributed:
        return obj
    objects = [obj]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def all_reduce_tensor(tensor: torch.Tensor, average: bool = False) -> torch.Tensor:
    ctx = get_context()
    if ctx.is_distributed:
        dist.all_reduce(tensor)
        if average:
            tensor /= ctx.world_size
    return tensor
