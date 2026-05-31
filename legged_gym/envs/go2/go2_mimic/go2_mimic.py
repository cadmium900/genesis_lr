import time
import numpy as np
import os
import random
import genesis as gs
from collections import deque
from scipy.stats import vonmises
import torch
from torch import Tensor
from typing import Tuple, Dict
import math

from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from genesis.engine.solvers.avatar_solver import AvatarSolver
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from warnings import WarningMessage

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import wrap_to_pi, torch_rand_sqrt_float, quat_apply_yaw
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gs_utils import *
from legged_gym.utils.terrain import Terrain
from .go2_mimic_config import GO2MimicCfg
from . import anim_utils

class FbxScene:
    def __init__(self, urdf_file, fbx_file):
        self.urdf_file = urdf_file
        self.fbx_file = fbx_file
        self.manager, self.scene = anim_utils._load_scene(fbx_file)
        self.stacks = anim_utils._get_anim_stacks(self.scene)
        self.bones = anim_utils._collect_skeleton_nodes(self.scene)
        self.bones = anim_utils._resolve_bones(self.bones, None)
        self.stacks = anim_utils._resolve_stacks(self.stacks, None)
        if not self.stacks:
            print("No animation stacks found.", file=sys.stderr)
            return

        self.fps = anim_utils._get_fps(self.scene, None)
        if self.fps <= 0:
            print("Invalid FPS.", file=sys.stderr)
            return
        self.unit_scale = anim_utils._get_scene_unit_scale(self.scene)

        urdf_info = {}
        bone_limit_map = {}
        bone_rpy_map = {}
        name_map = {node.GetName(): node for node in self.bones}
        urdf_info = anim_utils._parse_urdf_joint_info(self.urdf_file)
        normalized = {anim_utils._normalize_name(name): name for name in name_map.keys()}
        for key, info in urdf_info.items():
            axis, lower, upper, rpy = info
            node = name_map.get(key)
            if node is None:
                node_name = normalized.get(anim_utils._normalize_name(key))
                if node_name:
                    node = name_map[node_name]
            if node is None:
                continue
            bone_limit_map[node.GetName()] = (axis, lower, upper)
            bone_rpy_map[node.GetName()] = rpy

        self.calibration_map: Dict[
            str, Tuple[int, float, float, float, float, str, np.ndarray, np.ndarray]
        ] = {}
        self.calibration_stack = None
        self.base_node = None
        self.base_rest_quat = None
        if bone_limit_map:
            self.calibration_stack = anim_utils._find_calibration_stack(self.stacks)
            if self.calibration_stack is None:
                print("Warning: no calibration stack found; skipping uncalibrated bones.")
            else:
                self.calibration_map = anim_utils._build_calibration_map(
                    self.scene,
                    self.calibration_stack,
                    self.bones,
                    bone_limit_map,
                    None,
                    self.fps
                )
        for node in self.bones:
            name = node.GetName()
            if name == "Base" or name.lower() == "base":
                self.base_node = node
                break
        if self.base_node is not None and self.calibration_stack is not None:
            self.scene.SetCurrentAnimationStack(self.calibration_stack)
            span = self.calibration_stack.GetLocalTimeSpan()
            start_sec = span.GetStart().GetSecondDouble()
            t = anim_utils._require_fbx().FbxTime()
            t.SetSecondDouble(float(start_sec))
            self.base_rest_quat = anim_utils._local_quat_from_global(self.base_node, t)

class Animator:
    def __init__(self, _fbx_scene):
        self.fbx_scene = _fbx_scene
        self.cur_anim = None
        self.cur_stack = None
        self.anim_span = None
        self.anim_start_sec = None
        self.anim_end_sec = None
        self.anim_total_frames = None
        self.anim_frame = 0
        self.fbx_time = anim_utils._require_fbx().FbxTime()

    def get_current_frame(self):
        return self.anim_frame

    def get_current_time(self):
        return self.anim_start_sec + self.anim_frame / self.fbx_scene.fps

    def set_current_frame(self, idx):
        if self.anim_total_frames is None:
            return
        idx = int(idx)
        max_frame = max(self.anim_total_frames - 1, 0)
        if idx < 0:
            idx = 0
        elif idx > max_frame:
            idx = max_frame
        self.anim_frame = idx
        time_sec = self.anim_start_sec + self.anim_frame / self.fbx_scene.fps
        self.fbx_time.SetSecondDouble(float(time_sec))

    def set_current_time(self, time_sec):
        if self.anim_total_frames is None or self.anim_start_sec is None or self.anim_end_sec is None:
            return
        if self.anim_end_sec <= self.anim_start_sec:
            self.anim_frame = 0
            self.fbx_time.SetSecondDouble(float(self.anim_start_sec))
            return
        if time_sec < self.anim_start_sec:
            time_sec = self.anim_start_sec
        elif time_sec > self.anim_end_sec:
            time_sec = self.anim_end_sec
        self.anim_frame = int(round((time_sec - self.anim_start_sec) * self.fbx_scene.fps))
        max_frame = max(self.anim_total_frames - 1, 0)
        if self.anim_frame > max_frame:
            self.anim_frame = max_frame
        elif self.anim_frame < 0:
            self.anim_frame = 0
        time_sec = self.anim_start_sec + self.anim_frame / self.fbx_scene.fps
        self.fbx_time.SetSecondDouble(float(time_sec))

    def activate(self):
        self.fbx_scene.scene.SetCurrentAnimationStack(self.cur_stack)

    def set_animation(self, anim_name):
        self.cur_anim = anim_name
        self.cur_stack = None
        for stack in self.fbx_scene.stacks:
            if anim_name in stack.GetName():
                self.cur_stack = stack
                break

        if self.cur_stack is None:
            return False

        self.activate()
        self.anim_span = self.cur_stack.GetLocalTimeSpan()
        self.anim_start_sec = self.anim_span.GetStart().GetSecondDouble()
        self.anim_end_sec = self.anim_span.GetStop().GetSecondDouble()
        self.anim_total_frames = int(
            math.floor((self.anim_end_sec - self.anim_start_sec) * self.fbx_scene.fps + 0.5)
        ) + 1
        if self.anim_total_frames <= 0:
            return False

        self.anim_frame = 0
        self.fbx_time.SetSecondDouble(float(self.anim_start_sec))
        return True

    def get_animation_angles(self):
        self.activate()
        urdf_angles, axis_angles, _local_quats = anim_utils._compute_urdf_joint_angles(
                    self.fbx_scene.bones,
                    self.fbx_time,
                    self.fbx_scene.calibration_map,
                    None,
                    None
                )
        return urdf_angles, axis_angles, _local_quats


class GO2Mimic(BaseTask):
    def __init__(self, cfg: GO2MimicCfg, sim_device, headless):
        start = time.time()
        # Current animation
        self.animator = None

        self.cfg = cfg
        self.last_record_time = time.time()
        self.video_capturing = False
        self.height_samples = None
        self.debug_viz = self.cfg.env.debug_viz
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_device, headless)

        self._init_buffers()
        self._prepare_reward_function()
        self.init_camera_pos = False
        self.init_done = True
        assert torch.device(self.device) == torch.device(gs.device), (self.device, gs.device)
        print(f"[__init__] Latency: {time.time()-start}")

    def create_sim(self):
        start = time.time()
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.sim_dt,
                substeps=self.sim_substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt * self.cfg.control.decimation),
                camera_pos=np.array(self.cfg.viewer.pos),
                camera_lookat=np.array(self.cfg.viewer.lookat),
                camera_fov=60,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=self.cfg.viewer.rendered_envs_idx),
            rigid_options=gs.options.RigidOptions(
                dt=self.sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                constraint_timeconst=0.01,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=self.cfg.asset.self_collisions,
            ),
            show_viewer=not self.headless,
        )
        print(f"[create_sim:Scene] Latency: {time.time()-start}")

        start = time.time()
        # query rigid solver
        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            elif isinstance(solver, AvatarSolver):
                continue
            self.rigid_solver = solver

        # add camera if needed
        if self.cfg.viewer.add_camera:
            self._setup_camera()

        # add terrain
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type =='plane':
            self.terrain = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        elif mesh_type =='heightfield':
            self.utils_terrain = Terrain(self.cfg.terrain)
            self._create_heightfield()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self.terrain.set_friction(self.cfg.terrain.friction)
        # specify the boundary of the heightfield
        self.terrain_x_range = torch.zeros(2, device=self.device)
        self.terrain_y_range = torch.zeros(2, device=self.device)
        if self.cfg.terrain.mesh_type =='heightfield':
            self.terrain_x_range[0] = -self.cfg.terrain.border_size + 1.0  # give a small margin(1.0m)
            self.terrain_x_range[1] = self.cfg.terrain.border_size + self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length - 1.0
            self.terrain_y_range[0] = -self.cfg.terrain.border_size + 1.0
            self.terrain_y_range[1] = self.cfg.terrain.border_size + self.cfg.terrain.num_cols * self.cfg.terrain.terrain_width - 1.0
        elif self.cfg.terrain.mesh_type =='plane': # the plane used has limited size,
                                                   # and the origin of the world is at the center of the plane
            self.terrain_x_range[0] = -self.cfg.terrain.plane_length/2+1
            self.terrain_x_range[1] = self.cfg.terrain.plane_length/2-1
            self.terrain_y_range[0] = -self.cfg.terrain.plane_length/2+1  # the plane is a square
            self.terrain_y_range[1] = self.cfg.terrain.plane_length/2-1
        print(f"[create_sim:Terrain] Latency: {time.time()-start}")

        start = time.time()
        self._create_envs()
        print(f"[create_sim:create_envs] Latency: {time.time()-start}")

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        self.terrain = self.scene.add_entity(
            gs.morphs.Terrain(
                pos=(-self.cfg.terrain.border_size, -self.cfg.terrain.border_size, 0.0),
                horizontal_scale=self.cfg.terrain.horizontal_scale,
                vertical_scale=self.cfg.terrain.vertical_scale,
                height_field=self.utils_terrain.height_field_raw,
            ),
            vis_mode="collision"
        )
        self.height_samples = torch.tensor(self.utils_terrain.heightsamples).view(self.utils_terrain.tot_rows, self.utils_terrain.tot_cols).to(self.device)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = self.actions.clone()
            self.actions = self.action_queue[torch.arange(self.num_envs), self.action_delay].clone()
        for _ in range(self.cfg.control.decimation):  # use self-implemented pd controller
            self.torques = self._compute_torques(self.actions)
            if self.num_build_envs == 0:
                torques = self.torques.squeeze()
                self.robot.control_dofs_force(torques, self.motors_dof_idx)
            else:
                self.robot.control_dofs_force(self.torques, self.motors_dof_idx)
            self.scene.step()
            self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
            self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.post_physics_step()

        base_height = torch.mean(self.base_pos[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        #print(f"{base_height[0]}")
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                print(f"[X] Removed 0-scale reward {key}")
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name =="termination":
                continue
            self.reward_names.append(name)
            if scale < 0.0:
                name = '_neg_reward_' + name
            else:
                name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=gs.tc_float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        prev_base_lin_vel_world = self.base_lin_vel_world.clone()

        R_wb = quat_to_mat(self.base_quat)         # world-from-body
        self.base_axis_fwd   = torch.nn.functional.normalize(R_wb[:, :, 0], dim=-1)  # body +X (dog-forward)
        self.base_axis_lat   = torch.nn.functional.normalize(R_wb[:, :, 1], dim=-1)  # body +Y
        self.base_axis_dn    = torch.nn.functional.normalize(-R_wb[:, :, 2], dim=-1)  # body -Z

        base_quat_rel = gs_quat_mul(self.base_quat, gs_inv_quat(self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1)))
        self.base_euler = gs_quat2euler(base_quat_rel)
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel_world[:] = torch.nan_to_num(self.robot.get_vel(), nan=0.0, posinf=0.0, neginf=0.0)
        self.base_lin_acc_world[:] = (self.base_lin_vel_world - prev_base_lin_vel_world) / self.dt
        acc_world_with_gravity = self.base_lin_acc_world - self.gravity_world
        self.base_lin_acc_body[:] = transform_by_quat(acc_world_with_gravity, inv_base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.base_lin_vel_world, inv_base_quat) # transform to base frame
        self.base_ang_vel[:] = transform_by_quat(torch.nan_to_num(self.robot.get_ang(), nan=0.0, posinf=0.0, neginf=0.0), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.prev_base_lin_vel_world[:] = self.base_lin_vel_world
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.link_contact_forces[:] = self.robot.get_links_net_contact_force()
        self.feet_pos[:] = self.robot.get_links_pos()[:, self.feet_indices, :]
        self.feet_vel[:] = self.robot.get_links_vel()[:, self.feet_indices, :]
        all_links_quat = self.robot.get_links_quat()
        self.feet_quat[:] = all_links_quat[:, self.feet_indices, :]

        flat_quat = self.feet_quat.reshape(-1, 4)
        foot_up = torch.zeros_like(flat_quat[:, :3])
        foot_up[:, 2] = 1.0
        flat_up_world = transform_by_quat(foot_up, flat_quat)
        self.feet_up_world[:] = flat_up_world.reshape(self.num_envs, len(self.feet_indices), 3)

        #print(f"angle_fl: {torch.rad2deg(angle_between_vectors(self.feet_pos[:, 0, :], self.base_head_pitch))}")
        pos_t = self.scene.rigid_solver.get_links_pos(self.links_idx, envs_idx=None)
        if len(pos_t) > 0:
            pos_np = pos_t.squeeze(0)
            self.robot_com[:] = (pos_np * self.robot_link_mass[:, None]).sum(axis=1) / self.robot_link_mass.sum()
        else:
            self.robot_com[:] = self.base_pos[:, :3]
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_base_pos_out_of_bound()
        self.check_termination()
        self._update_anim_targets()
        self.compute_reward()
        # Periodic Reward Framework phi cycle
        # step after computing reward but before resetting the env
        self.gait_time += self.dt


        if self.anim_base_quat_seq is not None:
            self.target_quat = normalize(self.anim_base_quat_targets)
            self.target_fwd = gs_quat_apply(self.target_quat, self.forward_vec)
            self.target_pitch = torch.atan2(
                self.target_fwd[:, 2],
                torch.norm(self.target_fwd[:, :2], dim=1),
            ).unsqueeze(1)
        else:
            self.target_quat = torch.zeros(
                self.num_envs,
                4,
                dtype=gs.tc_float,
                device=self.device,
            )
            self.target_pitch = torch.zeros(
                self.num_envs,
                1,
                dtype=gs.tc_float,
                device=self.device,
            )
        if self.anim_base_height_seq is not None:
            self.anim_base_height = self.anim_base_height_targets.unsqueeze(1)
        else:
            self.anim_base_height = torch.zeros(
                self.num_envs,
                1,
                dtype=gs.tc_float,
                device=self.device,
            )
        if self.anim_dof_seq is not None:
            self.anim_dof_pos = (self.anim_dof_targets - self.default_dof_pos) * self.obs_scales.dof_pos
        else:
            self.anim_dof_pos = torch.zeros(
                self.num_envs,
                self.num_actions,
                dtype=gs.tc_float,
                device=self.device,
            )


        # +self.dt/2 in case of float precision errors
        is_over_limit = (self.gait_time >= (self.gait_period - self.dt / 2))
        if self.anim_frame_counts is not None and self.anim_frame_counts.shape[0] > 1:
            walking = torch.norm(self.commands[:, :2], dim=1) > 0.0
            walking_mask = walking.unsqueeze(1)
            wrapped = torch.remainder(self.gait_time, self.gait_period)
            clamped = torch.where(is_over_limit, self.gait_period, self.gait_time)
            self.gait_time = torch.where(walking_mask, wrapped, clamped)
        else:
            self.gait_time = torch.where(is_over_limit, self.gait_period, self.gait_time)
        self.phi = self.gait_time / self.gait_period
        # print(
        #     f"pid {os.getpid()} step {self.common_step_counter} "
        #     f"episode_len {int(self.episode_length_buf[0].item())} "
        #     f"gait_time {self.gait_time[0,0]} self.dt {self.dt} self_phi {self.phi[0]}"
        # )

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if self.num_build_envs > 0:
            self.reset_idx(env_ids)

        self._calc_periodic_reward_obs()
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        if self.debug_viz:
            self._draw_debug_vis(env_ids)

    def get_base_acc_with_gravity(self):
        """Base-frame linear acceleration including gravity (simulated accelerometer reading)."""
        return self.base_lin_acc_body

    def _draw_debug_vis(self, env_ids):
        """ Draws visualizations for debugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """

        R_wb = quat_to_mat(self.base_quat)
        dir_w = R_wb[:, :, 0]                               # body X in world (forward)
        dir_w = torch.nn.functional.normalize(dir_w, dim=-1)
        vec_w = dir_w * 1.0
        pos_w = self.base_pos[:, :3]  # [N,3]

        self.scene.clear_debug_objects()
        num_envs = self.num_envs

        #base_w_vel = self.base_lin_vel[:, :].detach().cpu().numpy()
        #print(f"{self.base_axis_fwd[0]}")

        self.scene.draw_debug_arrow(
             pos=pos_w[0].detach().cpu().numpy(),
             #vec=self.base_lin_vel[0, :].detach().cpu().numpy(),
             vec=self.robot.get_vel()[0].detach().cpu().numpy(),
             radius=0.02,
             color=(0.0, 0.0, 1.0, 0.8)
        )


        pos_rl = self.feet_pos[0, 2, :]
        pos_rr = self.feet_pos[0, 3, :]

        inv_base_quat = gs_inv_quat(self.base_quat)
        v = torch.cat([self.commands[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)  # Add Z=0 for 3D rotation

        # Rotate commands_local by 90 degrees around the Y-axis
        rotation_90_y = torch.tensor([0.0, -0.70710678, 0.0, 0.70710678], device=self.device)  # Quaternion for 90° rotation around Y-axis
        commands_local_rotated = transform_by_quat(v, rotation_90_y)
        v = transform_by_quat(commands_local_rotated, self.base_quat)[0, :]
        self.scene.draw_debug_arrow(
              pos=pos_w[0].detach().cpu().numpy(),
              vec=v.detach().cpu().numpy(),
              radius=0.02,
              color=(0.0, 1.0, 0.0, 0.5)
         )

#foot        v = self.tgt_rl_dir[0, :3]
#        self.scene.draw_debug_line(pos_rl.detach().cpu().numpy(), v.detach().cpu().numpy(), radius=0.02, color=(1.0, 0.0, 0.0, 0.5))

#foot        v = self.tgt_rr_dir[0, :3]
#        self.scene.draw_debug_line(pos_rr.detach().cpu().numpy(), v.detach().cpu().numpy(), radius=0.02, color=(1.0, 0.0, 0.0, 0.5))

        # Extract the heading angle (yaw) from commands
        # angle = wrap_to_pi(self.commands[:, 3] - 2.0 * self.commands[:, 2])
        # direction_world = torch.stack([torch.cos(angle),
        #                                torch.sin(angle),
        #                                torch.zeros_like(angle)], dim=-1)
        # direction_world = transform_by_quat(direction_world, inv_base_quat)[0, :]
        # v = torch.nn.functional.normalize(direction_world[:], dim=-1)
        # self.scene.draw_debug_arrow(
        #     pos=pos_w[0].detach().cpu().numpy(),
        #     vec=v.detach().cpu().numpy(),
        #     radius=0.01,
        #     color=(1.0, 1.0, 0.0, 1.0)
        # )

        # self.scene.draw_debug_arrow(
        #      pos=pos_w[0].detach().cpu().numpy(),
        #      #vec=vec_w[0].detach().cpu().numpy(),
        #      vec=self.base_head_pitch[0].detach().cpu().numpy(),
        #      radius=0.01,
        #      color=(0.0, 1.0, 1.0, 1.0)
        # )

        # self.scene.draw_debug_arrow(
        #      pos=pos_w[0].detach().cpu().numpy(),
        #      #vec=vec_w[0].detach().cpu().numpy(),
        #      vec=self.commands[0, :3].detach().cpu().numpy(),
        #      radius=0.01,
        #      color=(1.0, 0.0, 0.0, 1.0)
        # )

        # Show local frame
        # R_wb = quat_to_mat(self.base_quat)  # Rotation matrix from body to world
        # x_axis_world = R_wb[:, :, 0]  # Local X-axis in world frame
        # y_axis_world = R_wb[:, :, 1]  # Local Y-axis in world frame
        # z_axis_world = R_wb[:, :, 2]  # Local Z-axis in world frame

        # # Draw debug arrows for the local axes
        # self.scene.draw_debug_arrow(pos=self.base_pos[0].detach().cpu().numpy(),
        #                             vec=x_axis_world[0].detach().cpu().numpy(),
        #                             radius=0.02, color=(1.0, 0.0, 0.0, 1.0))  # Red for X-axis
        # self.scene.draw_debug_arrow(pos=self.base_pos[0].detach().cpu().numpy(),
        #                             vec=y_axis_world[0].detach().cpu().numpy(),
        #                             radius=0.02, color=(0.0, 1.0, 0.0, 1.0))  # Green for Y-axis
        # self.scene.draw_debug_arrow(pos=self.base_pos[0].detach().cpu().numpy(),
        #                             vec=z_axis_world[0].detach().cpu().numpy(),
        #                             radius=0.02, color=(0.0, 0.0, 1.0, 1.0))  # Blue for Z-axis


        # axes = [R[:, :, 0], R[:, :, 1], R[:, :, 2]]
        # cols = [(1,0,0,1),(0,1,0,1),(0,0,1,1)]
        # for i,(axis,col) in enumerate(zip(axes, cols)):
        #     self.scene.draw_debug_arrow(
        #         pos=pos_w[0].detach().cpu().numpy(),
        #         vec=axis[0].detach().cpu().numpy(),
        #         radius=0.02, color=col
        #     )



        # Entity/robot COM in world coordinates (from the solver)
        # indices of the robot's links
        # (1, N, 3) or (N, 3) depending on build → convert to (N, 3)
        # pos_t = self.scene.rigid_solver.get_links_pos(self.links_idx, envs_idx=0)
        # if len(pos_t) > 0:
        #     pos_np = pos_t.squeeze(0)
        #     robot_com = (pos_np * self.robot_link_mass[:, None]).sum(axis=0) / self.robot_link_mass.sum()
        #     #print(f"Base COM dist {robot_com - self.base_pos[0, :3].detach().cpu().numpy()}")

        #     # draw expects a single position, give it a flat list/tuple
        #     self.scene.draw_debug_sphere(robot_com.tolist(), radius=0.1, color=(1.0, 0.0, 0.0, 0.5))

        # draw height points
        if not self.cfg.terrain.measure_heights:
            return
        # self.scene.clear_debug_objects(self.base_pos[0,:], )
        # height_points = quat_apply_yaw(self.base_quat.repeat(
        #     1, self.num_height_points), self.height_points)
        # height_points[0, :, 0] += self.base_pos[0, 0]
        # height_points[0, :, 1] += self.base_pos[0, 1]
        # height_points[0, :, 2] = self.measured_heights[0, :]
        # # print(f"shape of height_points: ", height_points.shape) # (num_envs, num_points, 3)
        # self.scene.draw_debug_spheres(height_points[0, :], radius=0.03, color=(0, 0, 1, 0.7))  # only draw for the first env

    def check_base_pos_out_of_bound(self):
        """ Check if the base position is out of the terrain bounds
        """
        x_out_of_bound = (self.base_pos[:, 0] >= self.terrain_x_range[1]) | (self.base_pos[:, 0] <= self.terrain_x_range[0])
        y_out_of_bound = (self.base_pos[:, 1] >= self.terrain_y_range[1]) | (self.base_pos[:, 1] <= self.terrain_y_range[0])
        out_of_bound_buf = x_out_of_bound | y_out_of_bound
        envs_idx = out_of_bound_buf.nonzero(as_tuple=False).to(dtype=gs.tc_int).flatten()

        # reset base position to initial position
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_pos[envs_idx] += self.env_origins[envs_idx]

        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.link_contact_forces[:, self.termination_indices, :], dim=-1) > 1.0, dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

        proj_grav_over_limit = self.base_axis_fwd[:, 2] < self.termination_z
        self.reset_buf |= proj_grav_over_limit

        self._check_anim_termination()

        # After a while if robot not standing enough upright, reset
        # elapsed_limit = self.episode_length_buf > self.max_episode_length * 0.25
        # upright_fail = torch.logical_and(elapsed_limit, self.base_axis_fwd[:, 2] < 0.5)
        # self.reset_buf |= upright_fail

    def _check_anim_termination(self):
        if self.anim_frame_counts is None or self.anim_frame_count <= 0:
            return
        if self.gait_period.numel() == 0:
            return
        phase = (self.gait_time / self.gait_period).squeeze(-1)
        phase = torch.clamp(phase, 0.0, 1.0)
        active = phase >= self.cfg.rewards.anim_termination_phase
        if not torch.any(active):
            return

        anim_idx = torch.clamp(self.anim_index, 0, self.anim_frame_counts.shape[0] - 1)
        frame_counts = self.anim_frame_counts[anim_idx]
        max_frame = torch.clamp(frame_counts - 1, min=0)
        frame_float = phase * max_frame.to(phase.dtype)
        frame_idx = torch.clamp((frame_float + 0.5).to(torch.long), min=0)
        frame_idx = torch.minimum(frame_idx, max_frame.to(torch.long))

        term = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # orient_thresh = self.cfg.rewards.anim_termination_orient_rad
        # if self.anim_base_quat_seq is not None and orient_thresh > 0.0:
        #     base_init = self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1)
        #     base_rel = gs_quat_mul(self.base_quat, gs_inv_quat(base_init))
        #     base_rel = normalize(base_rel)
        #     target = normalize(self.anim_base_quat_seq[anim_idx, frame_idx])
        #     q_err = gs_quat_mul(base_rel, gs_inv_quat(target))
        #     w = torch.clamp(torch.abs(q_err[:, 0]), max=1.0)
        #     angle = 2.0 * torch.acos(w)
        #     term |= angle > orient_thresh

        height_thresh = self.cfg.rewards.anim_termination_height
        if self.anim_base_height_seq is not None and height_thresh > 0.0:
            if self.feet_pos.numel() > 0:
                min_foot_z = self.feet_pos[:, :, 2].min(dim=1).values
            else:
                min_foot_z = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
            cur_h = self.base_pos[:, 2] - min_foot_z
            target_h = self.anim_base_height_seq[anim_idx, frame_idx]
            term |= torch.abs(cur_h - target_h) > height_thresh

        # dof_thresh = self.cfg.rewards.anim_termination_dof
        # if self.anim_dof_seq is not None and dof_thresh > 0.0:
        #     target_dof = self.anim_dof_seq[anim_idx, frame_idx]
        #     dof_err = torch.abs(self._angle_diff(self.dof_pos, target_dof)).mean(dim=-1)
        #     term |= dof_err > dof_thresh

        self.reset_buf |= term & active



    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._neg_reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _compute_target_dof_pos(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self.cfg.control.action_scale
        target_dof_pos = actions_scaled + self.default_dof_pos

        return target_dof_pos


    def _compute_torques(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self.cfg.control.action_scale
        torques = (
            self._kp_scale * self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
            - self._kd_scale * self.d_gains * self.dof_vel
        )
        return torques

    def compute_observations(self):
        """ Computes observations
        """
        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,    # cmd     3 [0,1,2]
            self.projected_gravity,                        # g       3 [3,4,5]
            self.base_lin_acc_body,                        # acc     3 [6,7,8]
            self.base_ang_vel * self.obs_scales.ang_vel,   # omega   3 [9,10,11]
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # p_t     12 [12..23]
            self.dof_vel * self.obs_scales.dof_vel,        # dp_t    12 [24..35]
            self.actions,                                  # a_{t-1} 12 [36..47]
            self.clock_input,                              # clock   2 [48..49]
            self.gait_period,                              # gait period 1 [50]
            self.anim_index.unsqueeze(1).to(self.commands_scale.dtype),  # anim index 1 [51]
            self.target_quat,                                   # anim target quat 4 [52..55]
            self.target_pitch,                                  # anim target pitch 1 [56]
            self.anim_base_height,                              # anim base height target 1 [57]
            self.anim_dof_pos,                                  # anim dof targets 12 [58..69]
        ), dim=-1)

        if self.cfg.domain_rand.randomize_ctrl_delay:
            # normalize to [0, 1]
            ctrl_delay = (self.action_delay /
                          self.cfg.domain_rand.ctrl_delay_step_range[1]).unsqueeze(1)

        if self.num_privileged_obs is not None:  # critic_obs, no noise
            self.privileged_obs_buf = torch.cat((
                self.base_lin_vel * self.obs_scales.lin_vel,   # v_t     3
                self.base_lin_acc_body,
                self.commands[:, :3] * self.commands_scale,    # cmd_t   3
                self.projected_gravity,                        # g_t     3
                self.base_ang_vel * self.obs_scales.ang_vel,   # omega_t 3
                (self.dof_pos - self.default_dof_pos) *
                self.obs_scales.dof_pos,                       # p_t     12
                self.dof_vel * self.obs_scales.dof_vel,        # dp_t    12
                self.actions,                                  # a_{t-1} 12
                self.clock_input,                              # clock   2
                self.gait_period,                              # gait period 1
                self.anim_index.unsqueeze(1).to(self.commands_scale.dtype),  # anim index 1
                self.target_quat,                                   # anim target quat 4
                self.target_pitch,                                  # anim target pitch 1
                self.anim_base_height,                              # anim base height target 1
                self.anim_dof_pos,                                  # anim dof targets 12
                # domain randomization parameters
                self._rand_push_vels[:, :2],                   # 2
                self._added_base_mass,                         # 1
                self._friction_values,                         # 1
                self._base_com_bias,                           # 3
                # ctrl_delay,                                    # 1
                self._kp_scale,                                # 12
                self._kd_scale,                                # 12
                self._joint_armature,                          # 1
                self._joint_stiffness,                         # 1
                self._joint_damping,                           # 1
                # privileged infos
            ), dim=-1)

        # add perceptive inputs if not blind
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # # In ActorCritic.update_distribution

        # add noise if needed
        if self.add_noise:
            obs_now = obs_buf.clone()
            obs_now += (2 * torch.rand_like(obs_now) - 1) * \
                self.noise_scale_vec
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)
        self.obs_buf = torch.cat([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=-1)
        self.critic_history.append(self.privileged_obs_buf)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.critic_history.maxlen)], dim=-1)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length ==0):
            self.update_command_curriculum(env_ids)
        self._resample_behavior_params(env_ids)
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        # domain randomization
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(env_ids)
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(env_ids)
        if self.cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(env_ids)
        if self.cfg.domain_rand.randomize_joint_armature:
            self._randomize_joint_armature(env_ids)
        if self.cfg.domain_rand.randomize_joint_stiffness:
            self._randomize_joint_stiffness(env_ids)
        if self.cfg.domain_rand.randomize_joint_damping:
            self._randomize_joint_damping(env_ids)
        # reset buffers
        self.llast_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # Periodic Reward Framework buffer reset
        self.gait_time[env_ids] = 0.0
        self.phi[env_ids] = 0.0
        self.clock_input[env_ids] = 0.0
        if self.anim_dof_seq is not None and self.anim_frame_counts is not None:
            anim_idx = torch.clamp(self.anim_index[env_ids], 0, self.anim_frame_counts.shape[0] - 1)
            self.anim_frame_idx[env_ids] = 0
            self.anim_dof_targets[env_ids] = self.anim_dof_seq[anim_idx, 0]
            if self.anim_base_quat_seq is not None:
                self.anim_base_quat_targets[env_ids] = self.anim_base_quat_seq[anim_idx, 0]
            if self.anim_base_height_seq is not None:
                self.anim_base_height_targets[env_ids] = self.anim_base_height_seq[anim_idx, 0]

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["cur_command_x"] = self.commands[0, 0]
            self.extras["episode"]["cur_command_y"] = self.commands[0, 1]
            self.extras["episode"]["cur_command_yaw"] = self.commands[0, 2]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # Behavior parameters
        self.extras["episode"]["gait_period"] = torch.mean(self.gait_period[:])
        self.extras["episode"]["pitch_target"] = torch.mean(self.pitch_target[:])

        # reset action queue and delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[env_ids] *= 0.
            self.action_queue[env_ids] = 0.
            self.action_delay[env_ids] = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],
                                                       self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (len(env_ids),), device=self.device, requires_grad=False)

        # clear obs and critic history for the envs that are reset
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

        # resample domain randomization parameters
        self._episodic_domain_randomization(env_ids)

    def _randomize_friction(self, env_ids=None):
        ''' Randomize friction of all links'''
        min_friction, max_friction = self.cfg.domain_rand.friction_range

        solver = self.rigid_solver

        ratios = gs.rand((len(env_ids), 1), dtype=float).repeat(1, solver.n_geoms) \
        * (max_friction - min_friction) + min_friction
        self._friction_values[env_ids] = ratios[:,
            0].unsqueeze(1).detach().clone()

        solver.set_geoms_friction_ratio(
            ratios, torch.arange(0, solver.n_geoms), env_ids)

    def _randomize_base_mass(self, env_ids=None):
        ''' Randomize base mass'''
        min_mass, max_mass = self.cfg.domain_rand.added_mass_range
        base_link_id = 1
        added_mass = gs.rand((len(env_ids), 1), dtype=float) * \
                             (max_mass - min_mass) + min_mass
        self._added_base_mass[env_ids] = added_mass[:].detach().clone()
        self.rigid_solver.set_links_mass_shift(added_mass, [base_link_id, ], env_ids)

    def _randomize_com_displacement(self, env_ids):
        min_displacement, max_displacement = self.cfg.domain_rand.com_displacement_range
        base_link_id = 1

        com_displacement = gs.rand((len(env_ids), 1, 3), dtype=float) \
                            * (max_displacement - min_displacement) + min_displacement
        self._base_com_bias[env_ids] = com_displacement[:, 0, :].detach().clone()

        self.rigid_solver.set_links_COM_shift(com_displacement, [base_link_id,], env_ids)

    def _randomize_joint_armature(self, env_ids):
        """ Randomize joint armature of the robot
        """
        min_armature, max_armature = self.cfg.domain_rand.joint_armature_range
        armature = torch.rand((1,), dtype=gs.tc_float, device=self.device) \
        * (max_armature - min_armature) + min_armature # scalar
        self._joint_armature[env_ids, 0] = armature[0].detach().clone()
        armature = armature.repeat(self.num_actions)  # repeat for all motors
        self.robot.set_dofs_armature(
            armature, self.motors_dof_idx, envs_idx=env_ids) # all environments share the same armature
        # This armature will be Refreshed when envs are reset

    def _randomize_joint_stiffness(self, env_ids):
        """ Randomize joint stiffness of the robot
        """
        min_stiffness, max_stiffness = self.cfg.domain_rand.joint_stiffness_range
        stiffness = torch.rand((1,), dtype=gs.tc_float, device=self.device) \
        * (max_stiffness - min_stiffness) + min_stiffness
        self._joint_stiffness[env_ids, 0] = stiffness[0].detach().clone()
        stiffness = stiffness.repeat(self.num_actions)
        self.robot.set_dofs_stiffness(
            stiffness, self.motors_dof_idx, envs_idx=env_ids)

    def _randomize_joint_damping(self, env_ids):
        """ Randomize joint damping of the robot
        """
        min_damping, max_damping = self.cfg.domain_rand.joint_damping_range
        damping = torch.rand((1,), dtype=gs.tc_float, device=self.device) \
        * (max_damping - min_damping) + min_damping
        self._joint_damping[env_ids, 0] = damping[0].detach().clone()
        damping = damping.repeat(self.num_actions)
        self.robot.set_dofs_damping(
            damping, self.motors_dof_idx, envs_idx=env_ids)

    def _reset_dofs(self, envs_idx):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[envs_idx] = (self.default_dof_pos) + gs_rand_float(-0.3, 0.3, (len(envs_idx), self.num_actions), self.device)
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.robot.zero_all_dofs_velocity(envs_idx)

    def _reset_root_states(self, envs_idx):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base pos: xy [-1, 1]
        if self.custom_origins:
            self.base_pos[envs_idx] = self.base_init_pos
            self.base_pos[envs_idx] += self.env_origins[envs_idx]
            self.base_pos[envs_idx, :2] += gs_rand_float(-1.0, 1.0, (len(envs_idx), 2), self.device)
        else:
            self.base_pos[envs_idx] = self.base_init_pos
            self.base_pos[envs_idx] += self.env_origins[envs_idx]
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)

        # base quat
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        base_euler = gs_rand_float(-0.1, 0.1, (len(envs_idx), 3), self.device)  # roll, pitch [-0.1, 0.1]
        base_euler[:, 2] = gs_rand_float(*self.cfg.init_state.yaw_angle_range, (len(envs_idx),), self.device)  # yaw angle
        self.base_quat[envs_idx] = gs_quat_mul(gs_euler2quat(base_euler), self.base_quat[envs_idx],)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.zero_all_dofs_velocity(envs_idx)

        # update projected gravity
        inv_base_quat = gs_inv_quat(self.base_quat)
        self.projected_gravity = gs_transform_by_quat(self.global_gravity, inv_base_quat)
        self.base_lin_vel_world[envs_idx] = 0.0
        self.prev_base_lin_vel_world[envs_idx] = 0.0
        self.base_lin_acc_world[envs_idx] = 0.0
        self.base_lin_acc_body[envs_idx] = 0.0
        self.base_lin_vel_world_est[envs_idx] = 0.0
        self.base_lin_vel_est[envs_idx] = 0.0
        self.acc_bias_est[envs_idx] = 0.0

        # reset root states - velocity
        self.base_lin_vel[envs_idx] = (gs_rand_float(-0.5, 0.5, (len(envs_idx), 3), self.device))
        self.base_ang_vel[envs_idx] = (gs_rand_float(-0.5, 0.5, (len(envs_idx), 3), self.device))

        base_vel = torch.concat([self.base_lin_vel[envs_idx], self.base_ang_vel[envs_idx]], dim=1)
        self.robot.set_dofs_velocity(velocity=base_vel, dofs_idx_local=[0, 1, 2, 3, 4, 5], envs_idx=envs_idx)

        if not self.init_camera_pos:
            self.init_camera_pos = True
            if not self.headless:
                # Set camera
                # extract world position of env 0
                origin0 = self.env_origins[0].detach().cpu().numpy()  # shape (3,)

                # set camera position a bit above it
                cam_pos = origin0 + np.array([-2.0, 0.0, 2.0])   # 2 units up
                cam_lookat = origin0                             # look at env center

                self.scene.viewer.set_camera_pose(
                    pos=cam_pos,
                    lookat=cam_lookat
                )

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if "anim_base_orient" not in self.episode_sums:
            return
        if torch.mean(self.episode_sums["anim_base_orient"][env_ids]) / \
            self.max_episode_length > 0.5 * self.reward_scales["anim_base_orient"]:
            self.commands[env_ids, 0] = gs_rand_float(*self.cfg.commands.ranges.lin_vel_x, (len(env_ids),), self.device)
            self.commands[env_ids, 1] = gs_rand_float(*self.cfg.commands.ranges.lin_vel_y, (len(env_ids),), self.device)
            self.commands[env_ids, 2] = gs_rand_float(*self.cfg.commands.ranges.ang_vel_yaw, (len(env_ids),), self.device)

            # set small commands to zero
            #self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.min_normal).unsqueeze(1)

        else:
            self.commands[env_ids, :3] = 0.0


    def _resample_behavior_params(self, env_ids):
        if len(env_ids) == 0:
            return

        self._sync_gait_period(env_ids)

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        if self.push_interval_s > 0 and not self.debug:

            gate = self._biped_orientation_gate()

            max_push_vel_xy = self.cfg.domain_rand.max_push_vel_xy
            # in Genesis, base link also has DOF, it's 6DOF if not fixed.
            dofs_vel = self.robot.get_dofs_velocity()  # (num_envs, num_dof) [0:3] ~ base_link_vel
            push_vel = gs_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
            self._rand_push_vels[:, :2] = push_vel.detach().clone()
            push_vel[((self.common_step_counter + self.env_identities) % int(self.push_interval_s / self.dt) != 0)] = 0.0
            push_vel[gate < 0.5] = 0.0 # Don't push if not standing up
            dofs_vel[:, :2] += push_vel
            self.robot.set_dofs_velocity(dofs_vel)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.base_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.utils_terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if "tracking_lin_vel" not in self.episode_sums:
            return
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > \
                self.cfg.commands.curriculum_threshold * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    # ------------- Callbacks --------------
    def _calc_periodic_reward_obs(self):
        """Calculate the periodic reward observations.
        """
        phase = 2 * torch.pi * self.phi
        self.clock_input[:, 0] = torch.sin(phase).squeeze(-1)
        self.clock_input[:, 1] = torch.cos(phase).squeeze(-1)

    def _build_anim_cache(self):
        """Precompute URDF joint angles for every animation stack."""
        dof_names = self.cfg.asset.dof_names
        anim_names = list(self.cfg.asset.anim_stack)
        if not anim_names:
            raise RuntimeError("cfg.asset.anim_stack is empty.")

        seqs = []
        base_seqs = []
        base_height_seqs = []
        frame_counts = []
        resolved_names = []
        base_node = self.fbx_scene.base_node
        if base_node is None:
            raise RuntimeError("Base bone not found in FBX skeleton.")
        foot_keys = [name.lower() for name in self.cfg.asset.foot_name]
        foot_nodes = [
            node
            for node in self.fbx_scene.bones
            if any(key in node.GetName().lower() for key in foot_keys)
        ]
        height_nodes = foot_nodes if foot_nodes else list(self.fbx_scene.bones)
        if not height_nodes:
            height_nodes = [base_node]
        if not foot_nodes:
            print("Warning: no FBX foot bones found; base height uses min Z across all bones.")
        base_rest_quat = self.fbx_scene.base_rest_quat
        if base_rest_quat is None:
            base_rest_quat = anim_utils._euler_deg_to_quat_xyz(base_node.LclRotation.Get())

        for anim_name in anim_names:
            if not self.animator.set_animation(anim_name):
                print("Animation stack list")
                for stack in self.fbx_scene.stacks:
                    print(f"-> {stack.GetName()}")
                raise RuntimeError(f"Animation stack not found: {anim_name}")
            stack_name = self.animator.cur_stack.GetName()
            frames = self.animator.anim_total_frames or 0
            if frames <= 0:
                raise RuntimeError(f"Invalid frame count for stack: {stack_name}")
            seq = np.zeros((frames, len(dof_names)), dtype=np.float32)
            base_seq = np.zeros((frames, 4), dtype=np.float32)
            base_height_axis_seq = np.zeros((3, frames), dtype=np.float32)
            prev_axis_angles = {}
            missing = set()
            for frame_idx in range(frames):
                time_sec = self.animator.anim_start_sec + frame_idx / self.fbx_scene.fps
                self.animator.set_current_time(time_sec)
                urdf_angles, _, _ = anim_utils._compute_urdf_joint_angles(
                    self.fbx_scene.bones,
                    self.animator.fbx_time,
                    self.fbx_scene.calibration_map,
                    None,
                    prev_axis_angles,
                )
                for i, name in enumerate(dof_names):
                    angle = urdf_angles.get(name)
                    if angle is None:
                        missing.add(name)
                        angle = 0.0
                    seq[frame_idx, i] = angle
                base_local = anim_utils._local_quat_from_global(base_node, self.animator.fbx_time)
                base_rel = anim_utils._quat_mul(base_local, anim_utils._quat_inv(base_rest_quat))
                base_seq[frame_idx] = np.array(
                    [base_rel[3], base_rel[0], base_rel[1], base_rel[2]], dtype=np.float32
                )
                base_global = anim_utils._global_pos_from_node(base_node, self.animator.fbx_time)
                min_vec = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
                for node in height_nodes:
                    pos = anim_utils._global_pos_from_node(node, self.animator.fbx_time)
                    min_vec = np.minimum(min_vec, pos)
                base_height_axis_seq[:, frame_idx] = base_global - min_vec
            if missing:
                raise ValueError(f"Missing FBX joints for DOFs: {sorted(missing)}")
            axis_idx = int(np.argmax(base_height_axis_seq.mean(axis=1)))
            if axis_idx != 2:
                axis_name = ["x", "y", "z"][axis_idx]
                print(f"Info: using FBX {axis_name}-axis for base height in stack {stack_name}.")
            base_height_seq = base_height_axis_seq[axis_idx]
            height_scale = getattr(self.fbx_scene, "unit_scale", 1.0)
            height_scale *= getattr(self.cfg.asset, "anim_height_scale", 1.0)
            base_height_seq = base_height_seq * float(height_scale)
            seqs.append(seq)
            base_seqs.append(base_seq)
            base_height_seqs.append(base_height_seq)
            frame_counts.append(frames)
            resolved_names.append(stack_name)

        max_frames = max(frame_counts)
        anim_count = len(seqs)
        anim_tensor = torch.zeros(
            anim_count,
            max_frames,
            len(dof_names),
            dtype=gs.tc_float,
            device=self.device,
        )
        base_quat_tensor = torch.zeros(
            anim_count,
            max_frames,
            4,
            dtype=gs.tc_float,
            device=self.device,
        )
        base_height_tensor = torch.zeros(
            anim_count,
            max_frames,
            dtype=gs.tc_float,
            device=self.device,
        )
        for idx, seq in enumerate(seqs):
            seq_t = torch.tensor(seq, dtype=gs.tc_float, device=self.device)
            anim_tensor[idx, : seq_t.shape[0], :] = seq_t
            if seq_t.shape[0] < max_frames:
                anim_tensor[idx, seq_t.shape[0] :, :] = seq_t[-1:]
            base_t = torch.tensor(base_seqs[idx], dtype=gs.tc_float, device=self.device)
            base_quat_tensor[idx, : base_t.shape[0], :] = base_t
            if base_t.shape[0] < max_frames:
                base_quat_tensor[idx, base_t.shape[0] :, :] = base_t[-1:]
            base_height_t = torch.tensor(base_height_seqs[idx], dtype=gs.tc_float, device=self.device)
            base_height_tensor[idx, : base_height_t.shape[0]] = base_height_t
            if base_height_t.shape[0] < max_frames:
                base_height_tensor[idx, base_height_t.shape[0] :] = base_height_t[-1:]

        self.anim_names = resolved_names
        self.anim_dof_seq = anim_tensor
        self.anim_base_quat_seq = base_quat_tensor
        self.anim_base_height_seq = base_height_tensor
        self.anim_frame_counts = torch.tensor(
            frame_counts, dtype=gs.tc_int, device=self.device
        )
        self.anim_frame_count = int(max_frames)
        self.anim_name = self.anim_names[0]

    def _sync_gait_period(self, env_ids=None):
        """Keep gait_period equal to the selected animation duration."""
        if self.anim_frame_counts is None or not hasattr(self, "anim_index"):
            return
        if env_ids is None:
            anim_idx = torch.clamp(self.anim_index, 0, self.anim_frame_counts.shape[0] - 1)
        else:
            anim_idx = torch.clamp(self.anim_index[env_ids], 0, self.anim_frame_counts.shape[0] - 1)
        frame_counts = self.anim_frame_counts[anim_idx].to(self.gait_period.dtype)
        durations = torch.clamp(frame_counts - 1.0, min=1.0) / self.fbx_scene.fps
        if env_ids is None:
            self.gait_period[:, 0] = durations
        else:
            self.gait_period[env_ids, 0] = durations

    def _update_anim_targets(self):
        """Update per-env animation targets based on gait_time."""
        if self.anim_dof_seq is None or self.anim_frame_counts is None or self.anim_frame_count <= 0:
            return
        walking = None
        if self.anim_frame_counts.shape[0] > 1:
            walking = torch.norm(self.commands[:, :2], dim=1) > 0.0
            self.anim_index = torch.where(
                walking,
                torch.ones_like(self.anim_index),
                torch.zeros_like(self.anim_index),
            )
        self._sync_gait_period()
        if walking is not None:
            walking_mask = walking.unsqueeze(1)
            self.gait_time = torch.where(walking_mask, self.gait_time, self.gait_period)
        phase = self.gait_time / self.gait_period
        phase = torch.clamp(phase, 0.0, 1.0).squeeze(-1)
        if hasattr(self, "reset_buf"):
            phase = torch.where(self.reset_buf, torch.zeros_like(phase), phase)
        anim_idx = torch.clamp(self.anim_index, 0, self.anim_frame_counts.shape[0] - 1)
        frame_counts = self.anim_frame_counts[anim_idx]
        max_frame = torch.clamp(frame_counts - 1, min=0)
        frame_float = phase * max_frame.to(phase.dtype)
        frame_idx = torch.clamp((frame_float + 0.5).to(torch.long), min=0)
        frame_idx = torch.minimum(frame_idx, max_frame.to(torch.long))
        self.anim_frame_idx = frame_idx
        self.anim_dof_targets = self.anim_dof_seq[anim_idx, frame_idx]
        if self.anim_base_quat_seq is not None:
            self.anim_base_quat_targets = self.anim_base_quat_seq[anim_idx, frame_idx]
        if self.anim_base_height_seq is not None:
            self.anim_base_height_targets = self.anim_base_height_seq[anim_idx, frame_idx]

    def _clock_phase(self):
        """Recover phase in [0, 1) from sin/cos clock inputs."""
        phase = torch.atan2(self.clock_input[:, 0], self.clock_input[:, 1])
        phase = (phase + 2 * torch.pi) % (2 * torch.pi)
        return phase / (2 * torch.pi)

    def _post_physics_step_callback(self):
        # Update stationary mask based on command magnitude
        cmd_mag = torch.norm(self.commands[:, :3], dim=1)
        self.is_stationary[:] = cmd_mag < self.cfg.commands.min_normal

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        if self.cfg.commands.heading_command:
            # forward = body +Z in world
            forward = self.base_axis_dn
            heading = torch.atan2(forward[:, 1], forward[:, 2])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots:
            self._push_robots()
        # Periodic Reward Framework. resample phase and theta
        self._resample_behavior_params(env_ids)

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError(
                "Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(
                1, self.num_height_points), self.height_points[env_ids]) + (self.base_pos[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(
                1, self.num_height_points), self.height_points) + (self.base_pos[:, :3]).unsqueeze(1)

        points += self.cfg.terrain.border_size
        points = (points/self.cfg.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.cfg.terrain.vertical_scale

    def _terrain_height_at_points(self, points_xy: torch.Tensor) -> torch.Tensor:
        """Return terrain height right under the queried XY points (shape: [N, K, 2]).
        Falls back to zeros on non-heightfield terrain."""
        if self.height_samples is None or self.height_samples.numel() == 0:
            return torch.zeros(points_xy.shape[:-1], device=self.device, dtype=gs.tc_float)

        px = ((points_xy[..., 0] + self.cfg.terrain.border_size) / self.cfg.terrain.horizontal_scale).long()
        py = ((points_xy[..., 1] + self.cfg.terrain.border_size) / self.cfg.terrain.horizontal_scale).long()

        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        flat_px = px.view(-1)
        flat_py = py.view(-1)

        h1 = self.height_samples[flat_px, flat_py]
        h2 = self.height_samples[flat_px + 1, flat_py]
        h3 = self.height_samples[flat_px, flat_py + 1]
        h = torch.min(torch.min(h1, h2), h3)

        return h.view(*points_xy.shape[:-1]) * self.cfg.terrain.vertical_scale

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, dtype=gs.tc_float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # Observation layout (per frame): cmd(3), grav(3), base_lin_acc(3), base_ang_vel(3),
        # dof_pos(12), dof_vel(12), last_actions(12), clock(2), gait(1), anim_index(1),
        # anim_target_quat(4), anim_target_pitch(1), anim_base_height(1), anim_dof_targets(12)
        i = 0
        noise_vec[i:i+3] = 0.  # commands
        i += 3
        noise_vec[i:i+3] = noise_scales.gravity * noise_level
        i += 3
        noise_vec[i:i+3] = noise_scales.lin_vel * noise_level  # accel (no obs scaling applied)
        i += 3
        noise_vec[i:i+3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        i += 3
        noise_vec[i:i+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos                    # p_t
        i += self.num_actions
        noise_vec[i:i+self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # dp_t
        i += self.num_actions
        noise_vec[i:i+self.num_actions] = 0.  # a_{t-dt}
        i += self.num_actions
        noise_vec[i:i+2] = 0.  # clock
        i += 2
        noise_vec[i:i+1] = 0.  # gait period
        i += 1
        noise_vec[i:i+1] = 0.  # anim index
        i += 1
        noise_vec[i:i+4] = 0.  # anim target quat
        i += 4
        noise_vec[i:i+1] = 0.  # anim target pitch
        i += 1
        noise_vec[i:i+1] = 0.  # anim base height
        i += 1
        noise_vec[i:i+self.num_actions] = 0.  # anim dof targets
        i += self.num_actions
        if self.cfg.terrain.measure_heights and hasattr(self, "num_height_points"):
            end = i + self.num_height_points
            if end <= noise_vec.shape[0]:
                noise_vec[i:end] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        return noise_vec

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.forward_vec = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.forward_vec[:, 0] = 1.0
        self.is_stationary = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.tgt_rl_dir = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.tgt_rr_dir = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.vel_dir = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.vel_tgt = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_init_pos = torch.tensor(self.cfg.init_state.pos, device=self.device, dtype=gs.tc_float).contiguous()
        self.base_init_quat = torch.tensor(self.cfg.init_state.rot, device=self.device)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel_world = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_lin_acc_world = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_lin_acc_body = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)  # accelerometer-style (includes gravity)
        self.base_lin_vel_world_est = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel_est = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.prev_base_lin_vel_world = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.acc_bias_est = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.biped_base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.biped_base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        self.gravity_world = torch.tensor(self.cfg.sim.gravity, device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        self.commands = torch.zeros((self.num_envs, self.cfg.commands.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                            device=self.device, dtype=gs.tc_float, requires_grad=False,)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.llast_actions = torch.zeros(self.num_envs, self.num_actions, dtype=gs.tc_float, device=self.device, requires_grad=False)  # last last actions
        self.dof_pos = torch.zeros_like(self.actions, device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros_like(self.actions, device=self.device, dtype=gs.tc_float)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.robot_com = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_axis_fwd = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_axis_lat = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_axis_dn = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        self.last_contacts = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device, dtype=gs.tc_int)
        self.link_contact_forces = torch.zeros((self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float)
        self.feet_pos = torch.zeros((self.num_envs, len(self.feet_indices), 3), device=self.device, dtype=gs.tc_float)
        self.feet_vel = torch.zeros((self.num_envs, len(self.feet_indices), 3), device=self.device, dtype=gs.tc_float)
        self.feet_quat = torch.zeros((self.num_envs, len(self.feet_indices), 4), device=self.device, dtype=gs.tc_float)
        self.feet_up_world = torch.zeros((self.num_envs, len(self.feet_indices), 3), device=self.device, dtype=gs.tc_float)
        self.continuous_push = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.env_identities = torch.arange(self.num_envs, device=self.device, dtype=gs.tc_int,)
        self.terrain_heights = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float, )
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        self.target_quat = torch.zeros(
            self.num_envs,
            4,
            dtype=gs.tc_float,
            device=self.device,
        )
        self.target_pitch = torch.zeros(
            self.num_envs,
            1,
            dtype=gs.tc_float,
            device=self.device,
        )
        self.anim_base_height = torch.zeros(
            self.num_envs,
            1,
            dtype=gs.tc_float,
            device=self.device,
        )
        self.anim_dof_pos = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=gs.tc_float,
            device=self.device,
        )


        # randomize action delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(
                self.num_envs, self.cfg.domain_rand.ctrl_delay_step_range[1]+1, self.num_actions, dtype=gs.tc_float, device=self.device, requires_grad=False)
            self.action_delay = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],
                                              self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (self.num_envs,), device=self.device, requires_grad=False)

        self.default_dof_pos = torch.tensor(
            [self.cfg.init_state.default_joint_angles[name]
                for name in self.cfg.asset.dof_names],
            device=self.device,
            dtype=gs.tc_float,
        )
        # PD control
        stiffness = self.cfg.control.stiffness
        damping = self.cfg.control.damping

        self.p_gains, self.d_gains = [], []
        for dof_name in self.cfg.asset.dof_names:
            for key in stiffness.keys():
                if key in dof_name:
                    self.p_gains.append(stiffness[key])
                    self.d_gains.append(damping[key])
        self.p_gains = torch.tensor(self.p_gains, device=self.device)
        self.d_gains = torch.tensor(self.d_gains, device=self.device)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)
        # PD control params
        self.robot.set_dofs_kp(self.p_gains, self.motors_dof_idx)
        self.robot.set_dofs_kv(self.d_gains, self.motors_dof_idx)

        # obs_history
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,
                    dtype=gs.tc_float,
                    device=self.device,
                )
            )
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.single_num_privileged_obs,
                    dtype=gs.tc_float,
                    device=self.device,
                )
            )
        # Periodic Reward Framework
        self.gait_time = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device)
        self.phi = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device)
        self.gait_period = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device)
        self.gait_period[:] = self.cfg.rewards.behavior_params_range.gait_period_range[1]
        self.clock_input = torch.zeros(self.num_envs, 2, dtype=gs.tc_float, device=self.device)

        self.dummy_obs = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device)

        self.pitch_target = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device)

        self.anim_frame_idx = torch.zeros(self.num_envs, dtype=gs.tc_int, device=self.device)
        self.anim_index = torch.zeros(self.num_envs, dtype=gs.tc_int, device=self.device)
        self.anim_dof_targets = torch.zeros(
            self.num_envs,
            len(self.cfg.asset.dof_names),
            dtype=gs.tc_float,
            device=self.device,
        )
        self.anim_base_quat_targets = torch.zeros(self.num_envs, 4, dtype=gs.tc_float, device=self.device)
        self.anim_base_height_targets = torch.zeros(self.num_envs, dtype=gs.tc_float, device=self.device)
        if not hasattr(self, "anim_dof_seq"):
            self.anim_dof_seq = None
        if not hasattr(self, "anim_frame_count"):
            self.anim_frame_count = 0
        if not hasattr(self, "anim_name"):
            self.anim_name = None
        if not hasattr(self, "anim_cache"):
            self.anim_cache = {}
        if not hasattr(self, "anim_frame_counts"):
            self.anim_frame_counts = None
        if not hasattr(self, "anim_names"):
            self.anim_names = []
        if not hasattr(self, "anim_base_quat_seq"):
            self.anim_base_quat_seq = None
        if not hasattr(self, "anim_base_height_seq"):
            self.anim_base_height_seq = None
        self._sync_gait_period()

        # When the Z value of the forward vector reach below this Z value, terminate
        self.termination_z = self.cfg.domain_rand.termination_z

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _create_envs(self):
        start = time.time()
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # Create fbx animator (single shared scene)
        self.fbx_scene = FbxScene(
            self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
            self.cfg.asset.fbx_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
        )
        self.animator = Animator(self.fbx_scene)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=os.path.join(asset_root, asset_file),
                merge_fixed_links= True,  # if merge_fixed_links is True, then one link may have multiple geometries, which will cause error in set_friction_ratio
                links_to_keep= self.cfg.asset.links_to_keep,
                pos=np.array(self.cfg.init_state.pos),
                quat=np.array(self.cfg.init_state.rot),
                fixed= self.cfg.asset.fix_base_link,
            ),
            vis_mode="collision",
            visualize_contact=self.debug,
        )
        print(f"[create_envs:robot] Latency {time.time() - start}")

        start = time.time()
        self.scene.build(n_envs=self.num_envs)
        print(f"[create_envs:scene.build] Latency {time.time() - start}")

        start = time.time()
        self._get_env_origins()
        print(f"[create_envs:_get_env_origins] Latency {time.time() - start}")

        start = time.time()
        self._init_domain_params()
        self.robot_link_mass = np.array([lnk.get_mass() for lnk in self.robot.links], dtype=np.float64)
        self.robot_link_mass = torch.from_numpy(self.robot_link_mass).to(self.device).to(gs.tc_float)
        print(f"[create_envs:init_domain_params] Latency {time.time() - start}")

        start = time.time()
        dof_idx = [self.robot.get_joint(name).dof_start for name in self.cfg.asset.dof_names]
        self.motors_dof_idx = torch.as_tensor(dof_idx, dtype=gs.tc_int, device=gs.device)

        self._validate_joint_indices(dof_idx)

        self.joint_dof_mapping = {name: idx for idx, name in enumerate(self.cfg.asset.dof_names)}
        for k, v in self.joint_dof_mapping.items():
            print(f"{k}: {v}")

        self._build_anim_cache()

        self.dof_arm_left_idx = [self.joint_dof_mapping["FL_hip_joint"],
                                 self.joint_dof_mapping["FL_thigh_joint"],
                                 self.joint_dof_mapping["FL_calf_joint"]]
        self.dof_arm_right_idx = [self.joint_dof_mapping["FR_hip_joint"],
                                 self.joint_dof_mapping["FR_thigh_joint"],
                                 self.joint_dof_mapping["FR_calf_joint"]]


        # find link indices, termination links, penalized links, and feet
        def find_link_indices(names):
            link_indices = list()
            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices

        self.termination_indices = find_link_indices(self.cfg.asset.terminate_after_contacts_on)
        all_link_names = [link.name for link in self.robot.links]
        links_idx = np.array([link.idx for link in self.robot.links], dtype=np.int32)
        self.links_idx = torch.as_tensor(links_idx, dtype=gs.tc_int, device=gs.device)

        self.penalized_indices = find_link_indices(self.cfg.asset.penalize_contacts_on)
        self.feet_names = [link.name for link in self.robot.links if self.cfg.asset.foot_name[0] in link.name]
        self.feet_indices = find_link_indices(self.feet_names)
        assert len(self.termination_indices) > 0
        assert len(self.feet_indices) > 0

        # dof position limits
        self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motors_dof_idx), dim=1)
        self.torque_limits = self.robot.get_dofs_force_range(self.motors_dof_idx)[1]
        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = (m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit)
            self.dof_pos_limits[i, 1] = (m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit)
        print(f"[create_envs:dof_links] Latency {time.time() - start}")

        start = time.time()
        # randomize friction
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(np.arange(self.num_envs))
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(np.arange(self.num_envs))
        # randomize COM displacement
        if self.cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(np.arange(self.num_envs))
        # randomize joint armature
        if self.cfg.domain_rand.randomize_joint_armature:
            self._randomize_joint_armature(np.arange(self.num_envs))
        # randomize joint stiffness
        if self.cfg.domain_rand.randomize_joint_stiffness:
            self._randomize_joint_stiffness(np.arange(self.num_envs))
        # randomize joint damping
        if self.cfg.domain_rand.randomize_joint_damping:
            self._randomize_joint_damping(np.arange(self.num_envs))

        # distinguish between 4 feet
        for i in range(len(self.feet_indices)):
            if "FL" in self.feet_names[i]:
                self.foot_index_fl = self.feet_indices[i]
            elif "FR" in self.feet_names[i]:
                self.foot_index_fr = self.feet_indices[i]
            elif "RL" in self.feet_names[i]:
                self.foot_index_rl = self.feet_indices[i]
            elif "RR" in self.feet_names[i]:
                self.foot_index_rr = self.feet_indices[i]
        self.front_foot_indices = torch.tensor(
            [self.foot_index_fl, self.foot_index_fr],
            device=self.device,
            dtype=torch.long,
        )
        print(f"[create_envs:randomize] Latency {time.time() - start}")

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.utils_terrain.env_origins).to(self.device).to(gs.tc_float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing='ij')
            # plane has limited size, we need to specify spacing base on num_envs, to make sure all robots are within the plane
            # restrict envs to a square of [plane_length/2, plane_length/2]
            spacing = self.cfg.env.env_spacing
            if num_rows * self.cfg.env.env_spacing > self.cfg.terrain.plane_length / 2 or \
                num_cols * self.cfg.env.env_spacing > self.cfg.terrain.plane_length / 2:
                spacing = min((self.cfg.terrain.plane_length / 2) / (num_rows-1),
                              (self.cfg.terrain.plane_length / 2) / (num_cols-1))
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
            self.env_origins[:, 0] -= self.cfg.terrain.plane_length / 4
            self.env_origins[:, 1] -= self.cfg.terrain.plane_length / 4

    def _init_domain_params(self):
        self._friction_values = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device, requires_grad=False)
        self._added_base_mass = torch.ones(self.num_envs, 1, dtype=gs.tc_float, device=self.device, requires_grad=False)
        self._rand_push_vels = torch.zeros(self.num_envs, 3, dtype=gs.tc_float, device=self.device, requires_grad=False)
        self._base_com_bias = torch.zeros(self.num_envs, 3, dtype=gs.tc_float, device=self.device, requires_grad=False)
        self._joint_armature = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device, requires_grad=False)
        self._joint_stiffness = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device, requires_grad=False)
        self._joint_damping = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device, requires_grad=False)

        self._kp_scale = torch.ones(self.num_envs, self.num_actions, dtype=gs.tc_float, device=self.device)
        self._kd_scale = torch.ones(self.num_envs, self.num_actions, dtype=gs.tc_float, device=self.device)

    def _episodic_domain_randomization(self, env_ids):
        """ Update scale of Kp, Kd, rfi lim"""
        if len(env_ids) == 0:
            return

        if self.cfg.domain_rand.randomize_pd_gain:
            self._kp_scale[env_ids] = gs_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_actions), device=self.device)
            self._kd_scale[env_ids] = gs_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_actions), device=self.device)

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.dt
        # use self-implemented pd controller
        self.sim_dt = self.dt / self.cfg.control.decimation
        self.sim_substeps = 1
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.push_interval_s = self.cfg.domain_rand.push_interval_s

        self.dof_names = self.cfg.asset.dof_names
        self.debug = self.cfg.env.debug

    def _validate_joint_indices(self, dof_idx):
        """Validate that configured DOFs exist in the loaded robot and map uniquely."""
        expected_n = len(self.cfg.asset.dof_names)
        if len(dof_idx) != expected_n:
            raise ValueError(f"DOF count mismatch: config {expected_n}, robot {len(dof_idx)}")

        duplicate_idx = [idx for idx in set(dof_idx) if dof_idx.count(idx) > 1]
        if duplicate_idx:
            raise ValueError(f"Duplicate DOF indices found: {duplicate_idx} for {self.cfg.asset.dof_names}")

        robot_joint_map = {j.name: j for j in self.robot.joints}
        missing_robot = [name for name in self.cfg.asset.dof_names if name not in robot_joint_map]
        if missing_robot:
            raise ValueError(f"Configured joints not found in loaded robot: {missing_robot}")

        mismatch = {}
        for name, idx in zip(self.cfg.asset.dof_names, dof_idx):
            joint = robot_joint_map[name]
            if joint.dof_start != idx:
                mismatch[name] = {"robot_dof_start": joint.dof_start, "configured_idx": idx}
        if mismatch:
            raise ValueError(f"DOF start mismatch in loaded robot: {mismatch}")

        mapping = {name: {"dof_idx": idx, "robot_dof_start": robot_joint_map[name].dof_start} for name, idx in zip(self.cfg.asset.dof_names, dof_idx)}
        print(f"[INFO] DOF mapping (name -> dof_idx / robot_dof_start): {mapping}")

    def _neg_reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _neg_reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _neg_reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _neg_reward_action_smoothness(self):
        # Penalize action smoothness
        action_smoothness_cost = torch.sum(torch.square(self.actions - 2*self.last_actions + self.llast_actions), dim=-1)
        return action_smoothness_cost

    def _neg_reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(10.*(torch.norm(self.link_contact_forces[:, self.penalized_indices, :], dim=-1) > 0.1), dim=1)

    def _neg_reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _neg_reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _neg_reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def count_bad(self, loc):
        n_nan_loc   = torch.isnan(loc).sum().item()
        n_inf_loc   = torch.isinf(loc).sum().item()
        return (n_nan_loc, n_inf_loc)

    def _angle_diff(self, a, b):
        # wrap to (-pi, pi]
        return torch.atan2(torch.sin(a - b), torch.cos(a - b))

    def _biped_orientation_gate(self, force_angle=None):
        # Gate shaping rewards: 1 when aligned, 0 when far from target
        pitch_angle = torch.atan2(self.base_axis_fwd[:, 2], torch.norm(self.base_axis_fwd[:, :2], dim=1))
        if force_angle is None:
            target = 1.570796 # 90.0 degrees
        else:
            target = force_angle

        error = torch.abs(pitch_angle - target)

        window = max(self.cfg.rewards.biped_shaping_pitch_window * target, 1e-6)
        gate = torch.clamp(error / window, 0.0, 1.0)
        return 1.0 - gate

    def _reward_anim_dof_pos(self):
        if self.anim_dof_seq is None:
            print("Error anim")
            return torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        diff = self._angle_diff(self.dof_pos, self.anim_dof_targets)
        cost = 1.0 - torch.cos(diff)
        err = cost.mean(dim=-1)
        k = self.cfg.rewards.anim_dof_k
        reward = 1.0 / (1.0 + k * err)
        return reward

    def _reward_anim_base_height(self):
        if self.anim_base_height_seq is None:
            return torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)

        base_height = torch.mean(self.base_pos[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        rew = torch.square(base_height - self.anim_base_height_targets)
        reward = torch.exp(-rew / self.cfg.rewards.base_height_tracking_sigma)
        return reward

    def _reward_anim_base_orient(self):
        if self.anim_base_quat_seq is None:
            return torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)

        target = normalize(self.anim_base_quat_targets)
        target_fwd = gs_quat_apply(target, self.forward_vec)

        base_pitch = torch.atan2(self.base_axis_fwd[:, 2], torch.norm(self.base_axis_fwd[:, :2], dim=1))
        target_pitch = torch.atan2(target_fwd[:, 2], torch.norm(target_fwd[:, :2], dim=1))

        pitch_error = torch.abs(base_pitch - target_pitch)
        #print(f"time {self.gait_time[0]} Target {target_pitch[0]} vs {base_pitch[0]} ")
        tracking_reward = torch.exp(-pitch_error / self.cfg.rewards.euler_tracking_sigma)
        return tracking_reward


    def _reward_tracking_ang_vel(self):
        up = self.base_axis_fwd                           # biped up in world
        # base_ang_vel is in base frame in your code; get world ang vel instead:
        w_world = self.robot.get_ang()                    # [N,3] world
        yaw_rate = torch.sum(w_world * up, dim=1)         # project onto up axis

        cmd_yaw = self.commands[:, 2]
        err = (cmd_yaw - yaw_rate)**2

        reward = torch.exp(-err / self.cfg.rewards.tracking_sigma)
        gate = self._biped_orientation_gate()
        return torch.lerp(torch.zeros_like(reward), reward, gate)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # Ensure the z component of base_lin_vel is negative
        frame_lin_vel = self.base_lin_vel[:, :].clone()
        frame_lin_vel[:, 0] = -self.base_lin_vel[:, 2]

        cmd = self.commands[:, :2]
        vel = frame_lin_vel[:, :2]

        # cmd_mag = torch.norm(cmd, dim=1)
        # # Deadzone: don’t care about tracking for tiny commands
        # deadzone = 0.05
        # cmd_gate = (cmd_mag > deadzone).float()
        lin_vel_error = torch.sum(torch.square(cmd - vel), dim=1)

        reward = torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma) # * cmd_gate
        gate = self._biped_orientation_gate()
        return torch.lerp(torch.zeros_like(reward), reward, gate)
