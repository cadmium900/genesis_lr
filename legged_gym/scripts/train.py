import os
import sys


def _configure_cuda_scope():
    """Limit visible GPUs per process when launched via torchrun."""
    local_rank = os.environ.get("LOCAL_RANK")
    world_size = os.environ.get("WORLD_SIZE")
    if local_rank is None or world_size is None:
        return
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = local_rank


def _mirror_allocator_env():
    """Mirror legacy CUDA allocator env var to the new name to silence warnings."""
    legacy_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if legacy_conf and not os.environ.get("PYTORCH_ALLOC_CONF"):
        os.environ["PYTORCH_ALLOC_CONF"] = legacy_conf


_configure_cuda_scope()
_mirror_allocator_env()

os.environ["TI_OFFLINE_CACHE_FILE_PATH"] = os.path.expanduser("cache")
base_path = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))

# Prepend it to sys.path at runtime
if base_path not in sys.path:
    sys.path.insert(0, base_path)

import argparse
import numpy as np
from datetime import datetime
import time

import genesis as gs
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from rsl_rl.utils.distributed import init_distributed, cleanup as dist_cleanup
import torch
import shutil


def _cache_ready_flag_path():
    cache_root = os.environ["TI_OFFLINE_CACHE_FILE_PATH"]
    cache_root = os.path.abspath(cache_root)
    os.makedirs(cache_root, exist_ok=True)
    return os.path.join(cache_root, ".genesis_cache_ready")


def _wait_for_cache_ready(dist_ctx, timeout_s=1800):
    """Block non-main ranks until the Taichi cache has been built."""
    if dist_ctx.is_main_process:
        _cache_ready_flag_path()  # ensure directory exists
        return
    flag_path = _cache_ready_flag_path()
    if os.path.exists(flag_path):
        return
    start = time.time()
    rank = getattr(dist_ctx, "rank", -1)
    print(f"[Rank {rank}] Waiting for Genesis cache to be built by the main process...")
    while not os.path.exists(flag_path):
        time.sleep(1.0)
        if timeout_s is not None and (time.time() - start) > timeout_s:
            print(f"[Rank {rank}] Cache wait timed out after {timeout_s}s, continuing anyway.")
            break


def _mark_cache_ready(dist_ctx):
    if not dist_ctx.is_main_process:
        return
    flag_path = _cache_ready_flag_path()
    with open(flag_path, "w", encoding="utf-8") as flag_file:
        flag_file.write(str(time.time()))

def train(args):
    dist_ctx = init_distributed(cpu=args.cpu)
    if dist_ctx.device.type == "cuda":
        torch.cuda.set_device(dist_ctx.device)
    try:
        _wait_for_cache_ready(dist_ctx)
        gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level='warning')
        _mark_cache_ready(dist_ctx)
        # Make environment and algorithm runner
        env, env_cfg = task_registry.make_env(name=args.task, args=args, dist_ctx=dist_ctx)
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, dist_ctx=dist_ctx)

        # Copy env.py and env_config.py to log_dir for backup
        log_dir = ppo_runner.log_dir
        if dist_ctx.is_main_process and log_dir is not None:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            if env_cfg.asset.name == args.task:
                robot_file_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task+".py")
                robot_config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task+"_config.py")
            else:
                robot_file_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task, args.task+".py")
                robot_config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task, args.task+"_config.py")
            shutil.copy(robot_file_path, log_dir)
            shutil.copy(robot_config_path, log_dir)

        # Start training session
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    finally:
        dist_cleanup()

if __name__ == '__main__':
    args = get_args()
    if args.debug:
        args.offline = True
        args.num_envs = 1
    train(args)
