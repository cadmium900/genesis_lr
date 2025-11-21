# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import sys
import numpy as np
import atexit

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv
import genesis as gs
import os, platform
from rsl_rl.utils import distributed as dist_utils

def is_wsl():
    return platform.system() == "Linux" and "microsoft" in platform.release().lower()

if os.name == "nt" and not is_wsl():
    import msvcrt
else:
    import select
    import termios
    import tty

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu',
                 dist_ctx=None):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.all_cfg = train_cfg
        # self.wandb_run_name = (
        #     datetime.now().strftime("%b%d_%H-%M-%S")
        #     + "_"
        #     + self.cfg["experiment_name"]
        #     + "_"
        #     + self.cfg["run_name"]
        # )
        self.dist_ctx = dist_ctx or dist_utils.get_context()
        self.is_distributed = getattr(self.dist_ctx, "is_distributed", False)
        self.world_size = getattr(self.dist_ctx, "world_size", 1)
        self.rank = getattr(self.dist_ctx, "rank", 0)
        self.is_main_process = self.rank == 0
        self.device = torch.device(device)
        self.env = env
        self._camera = None
        self._camera_recording = False
        self._last_camera_frame_time = 0.0
        self._last_camera_target = None
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, dist_ctx=self.dist_ctx, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.should_write_logs = self.log_dir is not None and self.is_main_process
        self.should_collect_stats = self.should_write_logs or self.is_distributed
        cfg_obj = getattr(self.env, "cfg", None)
        cfg_env = getattr(cfg_obj, "env", None) if cfg_obj is not None else None
        global_envs = getattr(cfg_env, "global_num_envs", None) if cfg_env is not None else None
        self.global_num_envs = global_envs if global_envs is not None else self.env.num_envs * self.world_size
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # terminal handling for interactive commands
        self._use_linux_keyboard = False
        self._stdin_fd = None
        self._stdin_settings = None
        self._terminal_restore_registered = False
        if os.name != "nt":
            if sys.stdin.isatty():
                self._use_linux_keyboard = self.is_main_process
                if self._use_linux_keyboard:
                    self._stdin_fd = sys.stdin.fileno()
                    self._stdin_settings = termios.tcgetattr(self._stdin_fd)
                    tty.setcbreak(self._stdin_fd)
                    if not self._terminal_restore_registered:
                        atexit.register(self._restore_terminal)
                        self._terminal_restore_registered = True
            else:
                self._use_linux_keyboard = False
        else:
            self._use_linux_keyboard = self.is_main_process

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        try:
            # initialize writer
            if self.should_write_logs and self.writer is None:
                # wandb.init(
                #     project="genesis_lr",
                #     name=self.wandb_run_name,
                #     sync_tensorboard=True,
                #     config=self.all_cfg,
                # )
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            if init_at_random_ep_len:
                self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
            obs = self.env.get_observations()
            privileged_obs = self.env.get_privileged_observations()
            critic_obs = privileged_obs if privileged_obs is not None else obs
            obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
            self.alg.actor_critic.train() # switch to train mode (for dropout for example)

            ep_infos = []
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.env.num_envs, dtype=gs.tc_float, device=self.device)
            cur_episode_length = torch.zeros(self.env.num_envs, dtype=gs.tc_float, device=self.device)

            tot_iter = self.current_learning_iteration + num_learning_iterations
            for it in range(self.current_learning_iteration, tot_iter):
                start = time.time()
                # Rollout
                with torch.inference_mode():
                    for i in range(self.num_steps_per_env):

                        # Check for Nan
                        if not torch.isfinite(obs).all():
                            print("Bad OBS, fixing...")
                            bad_envs = torch.unique(torch.where(~torch.isfinite(obs))[0])
                            # if your vecenv supports per-env reset:
                            try:
                                self.env.reset_idx(bad_envs)  # or reset_envs()
                                obs = self.env.obs_buf  # refresh obs after reset if needed
                            except Exception:
                                # temporary stopgap to avoid crash
                                obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
                        actions = self.alg.act(obs, critic_obs)
                        obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                        critic_obs = privileged_obs if privileged_obs is not None else obs
                        obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                        self.alg.process_env_step(rewards, dones, infos)
                        self._maybe_render_camera_frame()

                        if self.should_collect_stats:
                            # Book keeping
                            if 'episode' in infos:
                                ep_infos.append(infos['episode'])
                            cur_reward_sum += rewards
                            cur_episode_length += 1
                            new_ids = (dones > 0).nonzero(as_tuple=False)
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                    stop = time.time()
                    collection_time = stop - start

                    # Learning step
                    start = stop
                    self.alg.compute_returns(critic_obs)

                mean_value_loss, mean_surrogate_loss = self.alg.update()
                stop = time.time()
                learn_time = stop - start
                if self.should_collect_stats:
                    self.log(locals())

                rq_quit = False
                rq_save = False
                rq_record_start = False
                rq_record_stop = False
                if os.name == "nt" and not is_wsl() and self.is_main_process:
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch().lower()
                        if ch == 'q':
                            rq_quit = True
                            print("\n[Quit Requested]")
                            self._restore_terminal()
                        elif ch == 's':
                            rq_save = True
                            print("\n[Save Requested]")
                        elif ch == 'b':
                            camera = self._resolve_camera()
                            if camera is not None:
                                if self._camera_recording:
                                    print("\n[Already Recording]")
                                else:
                                    print("\n[Start Recording Requested]")
                                    rq_record_start = True
                            else:
                                print("\n[No Floating Camera to Record From]")
                        elif ch == 'e':
                            camera = self._resolve_camera()
                            if camera is None:
                                print("\n[No Floating Camera to Record From]")
                            elif not self._camera_recording:
                                print("\n[No Active Recording to Stop]")
                            else:
                                print("\n[Stop Recording Requested]")
                                rq_record_stop = True
                elif self._use_linux_keyboard and self.is_main_process:
                    ch = self._poll_linux_key()
                    if ch == 'q':
                        rq_quit = True
                        print("\n[Quit Requested]")
                        self._restore_terminal()
                    elif ch == 's':
                        rq_save = True
                        print("\n[Save Requested]")
                    elif ch == 'b':
                        camera = self._resolve_camera()
                        if camera is not None:
                            if self._camera_recording:
                                print("\n[Already Recording]")
                            else:
                                print("\n[Start Recording Requested]")
                                rq_record_start = True
                        else:
                            print("\n[No Floating Camera to Record From]")
                    elif ch == 'e':
                        camera = self._resolve_camera()
                        if camera is None:
                            print("\n[No Floating Camera to Record From]")
                        elif not self._camera_recording:
                            print("\n[No Active Recording to Stop]")
                        else:
                            print("\n[Stop Recording Requested]")
                            rq_record_stop = True

                if self.is_distributed:
                    rq_quit = bool(dist_utils.distributed_max(1.0 if rq_quit else 0.0, device=self.device))
                    rq_save = bool(dist_utils.distributed_max(1.0 if rq_save else 0.0, device=self.device))
                    rq_record_start = bool(dist_utils.distributed_max(1.0 if rq_record_start else 0.0, device=self.device))
                    rq_record_stop = bool(dist_utils.distributed_max(1.0 if rq_record_stop else 0.0, device=self.device))

                if rq_record_start:
                    self._sync_camera_recording(action="start")
                if rq_record_stop:
                    self._sync_camera_recording(action="stop")

                if self.should_write_logs and self.log_dir is not None:
                    if rq_save or rq_quit or (it % self.save_interval == 0):
                        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                ep_infos.clear()

                if rq_quit:
                    self._restore_terminal()
                    break
            self.current_learning_iteration += num_learning_iterations

            if self.should_write_logs and self.log_dir is not None:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        finally:
            dist_utils.barrier()
            self._restore_terminal()

    def _maybe_render_camera_frame(self):
        """Render camera frames at a capped FPS when recording is active."""
        if not (self._camera_recording and self.is_main_process):
            return
        camera = self._resolve_camera()
        if camera is None:
            return
        self._update_camera_pose(camera)
        target_fps = self.cfg.get("recording_capture_fps", self.cfg.get("recording_fps", 10))
        try:
            target_fps = max(1, int(target_fps))
        except (TypeError, ValueError):
            target_fps = 10
        min_interval = 1.0 / target_fps
        now = time.time()
        if self._last_camera_frame_time and (now - self._last_camera_frame_time) < min_interval:
            return
        try:
            camera.render()
        except Exception as exc:
            print(f"[Camera] Failed to render frame: {exc}")
            self._camera_recording = False
            return
        self._last_camera_frame_time = now

    def _resolve_camera(self):
        """Cache and return the floating camera if present."""
        if self._camera is not None:
            return self._camera
        camera = getattr(self.env, "floating_camera", None)
        if camera is None and hasattr(self.env, "get_camera"):
            camera = self.env.get_camera()
        if camera is not None:
            self._camera = camera
        return camera

    def _update_camera_pose(self, camera, force=False):
        """Align camera pose with the configured reference environment's origin."""
        if camera is None:
            return
        env_cfg = getattr(self.env, "cfg", None)
        viewer_cfg = getattr(env_cfg, "viewer", None) if env_cfg is not None else None
        env_origins = getattr(self.env, "env_origins", None)
        base_positions = getattr(self.env, "base_pos", None)
        if viewer_cfg is None:
            return
        origin_candidate = None
        try:
            ref_env = int(getattr(viewer_cfg, "ref_env", 0))
        except (TypeError, ValueError):
            ref_env = 0
        if isinstance(base_positions, torch.Tensor) and base_positions.shape[0] > 0:
            ref_env = max(0, min(ref_env, base_positions.shape[0] - 1))
            origin_candidate = base_positions[ref_env].detach().cpu().numpy()
        elif isinstance(env_origins, torch.Tensor) and env_origins.numel() > 0:
            ref_env = max(0, min(ref_env, env_origins.shape[0] - 1))
            origin_candidate = env_origins[ref_env].detach().cpu().numpy()
        if origin_candidate is None:
            return
        if not force and self._last_camera_target is not None and np.allclose(origin_candidate, self._last_camera_target):
            return
        pos_offset = np.array(getattr(viewer_cfg, "pos", [3.0, 3.0, 3.0]), dtype=np.float32)
        lookat_offset = np.array(getattr(viewer_cfg, "lookat", [0.0, 0.0, 1.0]), dtype=np.float32)
        camera_pos = pos_offset + origin_candidate
        camera_lookat = origin_candidate #lookat_offset + origin_candidate
        if hasattr(self.env, "set_camera"):
            self.env.set_camera(pos=camera_pos, lookat=camera_lookat)
        else:
            camera.set_pose(pos=camera_pos, lookat=camera_lookat)
        self._last_camera_target = origin_candidate

    def _sync_camera_recording(self, action: str):
        """Pause all ranks while interacting with the floating camera."""
        if action not in ("start", "stop"):
            raise ValueError(f"Unsupported camera action '{action}'")

        dist_utils.barrier()
        camera = None
        success = 0.0
        if self.is_main_process:
            camera = self._resolve_camera()
            if camera is not None:
                try:
                    if action == "start":
                        self._update_camera_pose(camera, force=True)
                        camera.start_recording()
                        camera.render()
                    else:
                        save_dir = self.log_dir if self.log_dir is not None else os.getcwd()
                        filename = os.path.join(save_dir, f"{int(time.time())}.mp4")
                        fps = self.cfg.get("recording_fps", 10)
                        camera.stop_recording(save_to_filename=filename, fps=fps)
                    success = 1.0
                except Exception as exc:
                    print(f"[Camera] Failed to {action} recording: {exc}")
        dist_utils.barrier()
        success = bool(dist_utils.distributed_max(success, device=self.device))
        if action == "start" and success:
            self._camera_recording = True
            self._last_camera_frame_time = 0.0
        elif action == "stop":
            self._camera_recording = False

    def log(self, locs, width=80, pad=50):
        collection_time = dist_utils.distributed_max(locs['collection_time'], device=self.device)
        learn_time = dist_utils.distributed_max(locs['learn_time'], device=self.device)
        iteration_time = collection_time + learn_time
        mean_value_loss = dist_utils.distributed_mean(locs['mean_value_loss'], device=self.device)
        mean_surrogate_loss = dist_utils.distributed_mean(locs['mean_surrogate_loss'], device=self.device)

        self.tot_timesteps += self.num_steps_per_env * self.global_num_envs
        self.tot_time += iteration_time

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                values = []
                for ep_info in locs['ep_infos']:
                    value = ep_info[key]
                    if not isinstance(value, torch.Tensor):
                        value = torch.tensor([value], device=self.device)
                    if len(value.shape) == 0:
                        value = value.unsqueeze(0)
                    values.append(value.to(self.device))
                if not values:
                    continue
                stacked = torch.cat(values)
                value_sum = stacked.sum()
                value_count = stacked.numel()
                value_sum = dist_utils.distributed_sum(value_sum, device=self.device)
                value_count = dist_utils.distributed_sum(value_count, device=self.device)
                if value_count > 0:
                    mean_val = value_sum / value_count
                    if self.writer is not None:
                        self.writer.add_scalar('Episode/' + key, mean_val, locs['it'])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {mean_val:.6f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        mean_std_value = dist_utils.distributed_mean(mean_std.item(), device=self.device)
        total_frames = self.num_steps_per_env * self.global_num_envs
        fps = int(total_frames / iteration_time) if iteration_time > 0 else 0

        if self.writer is not None:
            self.writer.add_scalar('Loss/value_function', mean_value_loss, locs['it'])
            self.writer.add_scalar('Loss/surrogate', mean_surrogate_loss, locs['it'])
            self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
            self.writer.add_scalar('Policy/mean_noise_std', mean_std_value, locs['it'])
            self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
            self.writer.add_scalar('Perf/collection time', collection_time, locs['it'])
            self.writer.add_scalar('Perf/learning_time', learn_time, locs['it'])

        rewbuffer = list(locs['rewbuffer'])
        lenbuffer = list(locs['lenbuffer'])
        rew_sum_local = float(sum(rewbuffer)) if rewbuffer else 0.0
        rew_count_local = float(len(rewbuffer))
        len_sum_local = float(sum(lenbuffer)) if lenbuffer else 0.0
        len_count_local = float(len(lenbuffer))

        rew_sum = dist_utils.distributed_sum(rew_sum_local, device=self.device)
        rew_count = dist_utils.distributed_sum(rew_count_local, device=self.device)
        len_sum = dist_utils.distributed_sum(len_sum_local, device=self.device)
        len_count = dist_utils.distributed_sum(len_count_local, device=self.device)

        mean_reward = rew_sum / rew_count if rew_count > 0 else None
        mean_episode_length = len_sum / len_count if len_count > 0 else None

        if self.writer is not None and mean_reward is not None:
            self.writer.add_scalar('Train/mean_reward', mean_reward, locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', mean_reward, self.tot_time)
        if self.writer is not None and mean_episode_length is not None:
            self.writer.add_scalar('Train/mean_episode_length', mean_episode_length, locs['it'])
            self.writer.add_scalar('Train/mean_episode_length/time', mean_episode_length, self.tot_time)

        header = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        log_string = (f"""{'#' * width}\n"""
                      f"""{header.center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"""
                      f"""{'Value function loss:':>{pad}} {mean_value_loss:.4f}\n"""
                      f"""{'Surrogate loss:':>{pad}} {mean_surrogate_loss:.4f}\n"""
                      f"""{'Mean action noise std:':>{pad}} {mean_std_value:.2f}\n""")
        if mean_reward is not None:
            log_string += f"""{'Mean reward:':>{pad}} {mean_reward:.2f}\n"""
        if mean_episode_length is not None:
            log_string += f"""{'Mean episode length:':>{pad}} {mean_episode_length:.2f}\n"""

        log_string += ep_string
        eta_sec = self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it'])
        eta_hour = eta_sec / 3600.0
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {eta_sec:.1f}s {eta_hour:.1f}h\n"""
                       )

        if self.is_main_process:
            print(log_string)

    def _poll_linux_key(self):
        if not self._use_linux_keyboard:
            return None
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            ch = sys.stdin.read(1).lower()
            return ch
        return None

    def _restore_terminal(self):
        if self._stdin_fd is None or self._stdin_settings is None:
            return
        try:
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_settings)
            termios.tcflush(self._stdin_fd, termios.TCIOFLUSH)
        except termios.error:
            pass
        self._use_linux_keyboard = False

    def save(self, path, infos=None):
        if not self.is_main_process or path is None:
            return path
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)
        return path

    def load(self, path, load_optimizer=True):
        if path is None:
            return None
        map_location = self.device
        loaded_dict = torch.load(path, map_location=map_location) if self.is_main_process else None
        if self.is_distributed:
            loaded_dict = dist_utils.broadcast_object(loaded_dict, src=0)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
