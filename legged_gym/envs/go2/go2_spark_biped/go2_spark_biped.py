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

from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from genesis.engine.solvers.avatar_solver import AvatarSolver
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from warnings import WarningMessage

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import wrap_to_pi
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gs_utils import *
from .go2_spark_biped_config import GO2SparkBipedCfg


class GO2SparkBiped(BaseTask):
    def __init__(self, cfg: GO2SparkBipedCfg, sim_device, headless):
        start = time.time()
        self.cfg = cfg
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

    def set_camera(self, pos, lookat):
        """ Set camera position and direction
        """
        self.floating_camera.set_pose(
            pos=pos,
            lookat=lookat
        )

    # ------------- Callbacks --------------
    def _setup_camera(self):
        ''' Set camera position and direction
        '''
        self.floating_camera = self.scene.add_camera(
            res= (1280, 960),
            pos=np.array(self.cfg.viewer.pos),
            lookat=np.array(self.cfg.viewer.lookat),
            fov=60,
            GUI=True,
        )

        self._recording = False
        self._recorded_frames = []

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

        R_wb = quat_to_mat(self.base_quat)         # world-from-body
        self.base_axis_fwd   = torch.nn.functional.normalize(R_wb[:, :, 0], dim=-1)  # body +X (dog-forward)
        self.base_axis_lat   = torch.nn.functional.normalize(R_wb[:, :, 1], dim=-1)  # body +Y
        self.base_axis_dn    = torch.nn.functional.normalize(-R_wb[:, :, 2], dim=-1)  # body -Z

        base_quat_rel = gs_quat_mul(self.base_quat, gs_inv_quat(self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1)))
        self.base_euler = gs_quat2euler(base_quat_rel)
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat) # transform to base frame
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.link_contact_forces[:] = self.robot.get_links_net_contact_force()
        self.feet_pos[:] = self.robot.get_links_pos()[:, self.feet_indices, :]
        self.feet_vel[:] = self.robot.get_links_vel()[:, self.feet_indices, :]

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
        self.compute_reward()
        # Periodic Reward Framework phi cycle
        # step after computing reward but before resetting the env
        self.gait_time += self.dt
        # +self.dt/2 in case of float precision errors
        is_over_limit = (self.gait_time >= (self.gait_period - self.dt / 2))
        over_limit_indices = is_over_limit.nonzero(as_tuple=False).flatten()
        self.gait_time[over_limit_indices] = 0.0
        self.phi = self.gait_time / self.gait_period
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

        #v = self.vel_tgt[0, :3]
        v = self.tgt_rl_dir[0, :3]
        self.scene.draw_debug_line(pos_rl.detach().cpu().numpy(), v.detach().cpu().numpy(), radius=0.02, color=(1.0, 0.0, 0.0, 0.5))
        # self.scene.draw_debug_arrow(
        #      pos=pos_rl.detach().cpu().numpy(),
        #      vec=v.detach().cpu().numpy(),
        #      radius=0.02,
        #      color=(1.0, 0.0, 0.0, 0.5)
        # )

        #v = self.vel_dir[0, :3]
        v = self.tgt_rr_dir[0, :3]
        self.scene.draw_debug_line(pos_rr.detach().cpu().numpy(), v.detach().cpu().numpy(), radius=0.02, color=(1.0, 0.0, 0.0, 0.5))
        # self.scene.draw_debug_arrow(
        #      pos=pos_rr.detach().cpu().numpy(),
        #      vec=v.detach().cpu().numpy(),
        #      radius=0.02,
        #      color=(0.0, 1.0, 0.0, 0.5)
        # )

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

        proj_grav_over_limit = self.base_axis_fwd[:, 2] < self.termination_z # 0.6
        self.reset_buf |= proj_grav_over_limit


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
            self.commands[:, :3] * self.commands_scale,    # cmd     3
            self.projected_gravity,                        # g       3
            self.base_ang_vel * self.obs_scales.ang_vel,   # omega   3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,                       # p_t     12
            self.dof_vel * self.obs_scales.dof_vel,        # dp_t    12
            self.actions,                                  # a_{t-1} 12
            self.clock_input,                              # clock   4
            self.gait_period,                              # gait period 1
            self.base_height_target,                       # base height target 1
            self.foot_clearance_target,                    # foot clearance target 1
            self.fixed_pitch_target,                             # pitch target 1
            self.theta,                                    # theta, gait offset, 4
        ), dim=-1)

        if self.cfg.domain_rand.randomize_ctrl_delay:
            # normalize to [0, 1]
            ctrl_delay = (self.action_delay /
                          self.cfg.domain_rand.ctrl_delay_step_range[1]).unsqueeze(1)

        if self.num_privileged_obs is not None:  # critic_obs, no noise
            self.privileged_obs_buf = torch.cat((
                self.base_lin_vel * self.obs_scales.lin_vel,   # v_t     3
                self.commands[:, :3] * self.commands_scale,    # cmd_t   3
                self.projected_gravity,                        # g_t     3
                self.base_ang_vel * self.obs_scales.ang_vel,   # omega_t 3
                (self.dof_pos - self.default_dof_pos) *
                self.obs_scales.dof_pos,                       # p_t     12
                self.dof_vel * self.obs_scales.dof_vel,        # dp_t    12
                self.actions,                                  # a_{t-1} 12
                self.clock_input,                              # clock   4
                self.gait_period,                              # gait period 1
                self.base_height_target,                       # base height target 1
                self.foot_clearance_target,                    # foot clearance target 1
                self.fixed_pitch_target,                             # pitch target 1
                self.theta,                                    # theta, gait offset, 4
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
        self.clock_input[env_ids, :] = 0.0

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
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # Behavior parameters
        self.extras["episode"]["gait_period"] = torch.mean(self.gait_period[:])
        self.extras["episode"]["pitch_target"] = torch.mean(self.pitch_target[:])
        self.extras["episode"]["foot_clearance_target"] = torch.mean(self.foot_clearance_target[:])
        self.extras["episode"]["base_height_target"] = torch.mean(self.base_height_target[:])
        self.extras["episode"]["theta_fr"] = torch.mean(self.theta[:, 1])

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
        if torch.mean(self.episode_sums["orientation"][env_ids]) / \
            self.max_episode_length > 0.5 * self.reward_scales["orientation"]:
            self.commands[env_ids, 0] = gs_rand_float(*self.cfg.commands.ranges.lin_vel_x, (len(env_ids),), self.device)
            self.commands[env_ids, 1] = gs_rand_float(*self.cfg.commands.ranges.lin_vel_y, (len(env_ids),), self.device)
            self.commands[env_ids, 2] = gs_rand_float(*self.cfg.commands.ranges.ang_vel_yaw, (len(env_ids),), self.device)

            self.pitch_target[env_ids, :] = self.fixed_pitch_target[env_ids, :]

            # set small commands to zero
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.min_normal).unsqueeze(1)
        else:
            self.commands[env_ids, :3] = 0.0
            #self.commands[env_ids, 0] = 1.0
            self.pitch_target[env_ids, :] = self.fixed_pitch_target[env_ids, :]


    def _resample_behavior_params(self, env_ids):
        if len(env_ids) == 0:
            return

        if torch.mean(self.episode_sums["base_height"][env_ids]) / \
            self.max_episode_length > 0.9 * self.reward_scales["base_height"]:
            self.base_height_target[env_ids, :] = gs_rand_float(
                self.cfg.rewards.behavior_params_range.base_height_target_range[0],
                self.cfg.rewards.behavior_params_range.base_height_target_range[1],
                (len(env_ids), 1), device=self.device
            )

            self.gait_period[env_ids, :] = gs_rand_float(
                self.cfg.rewards.behavior_params_range.gait_period_range[0],
                self.cfg.rewards.behavior_params_range.gait_period_range[1],
                (len(env_ids), 1), device=self.device
            )

        if torch.mean(self.episode_sums["hind_foot_clearance"][env_ids]) / \
            self.max_episode_length > 0.75 * self.reward_scales["hind_foot_clearance"]:
            self.foot_clearance_target[env_ids, :] = gs_rand_float(
                self.cfg.rewards.behavior_params_range.foot_clearance_target_range[0],
                self.cfg.rewards.behavior_params_range.foot_clearance_target_range[1],
                (len(env_ids), 1), device=self.device
            )

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        if self.push_interval_s > 0 and not self.debug:
            max_push_vel_xy = self.cfg.domain_rand.max_push_vel_xy
            # in Genesis, base link also has DOF, it's 6DOF if not fixed.
            dofs_vel = self.robot.get_dofs_velocity()  # (num_envs, num_dof) [0:3] ~ base_link_vel
            push_vel = gs_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
            self._rand_push_vels[:, :2] = push_vel.detach().clone()
            push_vel[((self.common_step_counter + self.env_identities) % int(self.push_interval_s / self.dt) != 0)] = 0.0
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
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > \
                self.cfg.commands.curriculum_threshold * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    # ------------- Callbacks --------------
    def _calc_periodic_reward_obs(self):
        """Calculate the periodic reward observations.
        """
        for i in range(4):
            self.clock_input[:, i] = torch.sin(2 * torch.pi * (self.phi + self.theta[:, i].unsqueeze(1))).squeeze(-1)

    def _post_physics_step_callback(self):
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
        noise_vec[:3] = 0.  # commands
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[9:9+1*self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos                    # p_t
        noise_vec[9+1*self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # dp_t
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0.  # a_{t-dt}

        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements

        return noise_vec

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.forward_vec = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.forward_vec[:, 0] = 1.0
        self.tgt_rl_dir = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.tgt_rr_dir = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.vel_dir = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.vel_tgt = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_init_pos = torch.tensor(self.cfg.init_state.pos, device=self.device, dtype=gs.tc_float).contiguous()
        self.base_init_quat = torch.tensor(self.cfg.init_state.rot, device=self.device)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.biped_base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.biped_base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
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

        self.foot_clearance_target = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device)
        self.foot_clearance_target[:, :] = self.cfg.rewards.behavior_params_range.foot_clearance_target_range[0]
        self.last_contacts = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device, dtype=gs.tc_int)
        self.link_contact_forces = torch.zeros((self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float)
        self.feet_pos = torch.zeros((self.num_envs, len(self.feet_indices), 3), device=self.device, dtype=gs.tc_float)
        self.feet_vel = torch.zeros((self.num_envs, len(self.feet_indices), 3), device=self.device, dtype=gs.tc_float)
        self.continuous_push = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.env_identities = torch.arange(self.num_envs, device=self.device, dtype=gs.tc_int,)
        self.terrain_heights = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float, )
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

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
        self.theta = torch.zeros(self.num_envs, 4, dtype=gs.tc_float, device=self.device)
        self.gait_time = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device)
        self.phi = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device)
        self.gait_period = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device)
        self.gait_period[:] = self.cfg.rewards.behavior_params_range.gait_period_range[1]
        self.clock_input = torch.zeros(self.num_envs, 4, dtype=gs.tc_float, device=self.device, )

        self.dummy_obs = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device)

        self.pitch_target = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device)
        self.pitch_target[:, :] = self.cfg.rewards.behavior_params_range.pitch_target_range[1]
        self.fixed_pitch_target = torch.ones(self.num_envs, 1, dtype=gs.tc_float, device=self.device)
        self.base_height_target = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device)
        self.base_height_target[:, :] = self.cfg.rewards.behavior_params_range.base_height_target_range[1]

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

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=os.path.join(asset_root, asset_file),
                merge_fixed_links= True,  # if merge_fixed_links is True, then one link may have multiple geometries, which will cause error in set_friction_ratio
                links_to_keep= self.cfg.asset.links_to_keep,
                pos=np.array(self.cfg.init_state.pos),
                quat=np.array(self.cfg.init_state.rot),
                fixed= self.cfg.asset.fix_base_link,
            ),
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

    def _neg_reward_hip_pos(self):
        # Negative reward
        hip_joint_indices = [0, 3, 6, 9]
        dof_pos_error = torch.sum(torch.square(self.dof_pos[:, hip_joint_indices] - self.default_dof_pos[hip_joint_indices]), dim=-1)
        return dof_pos_error

    # def _neg_reward_ang_vel_xy(self):
    #     # Penalize
    #     return torch.sum(torch.square(self.robot.get_ang()[:, 1:2]), dim=1)  # small penalty

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

    # def _neg_reward_pitch_rate(self):
    #     # up = body +X in world; pitch back => base_ang around body Y in world coordinates
    #     w_world = self.robot.get_ang()
    #     pitch_rate = torch.sum(w_world * self.base_axis_lat, dim=1)  # rotate around +Y
    #     # only penalize when leaning back (axis_fwd.z dropping below target)
    #     back_lean = (self.base_axis_fwd[:, 2] < self.fixed_pitch_target.squeeze(1)).float()
    #     return back_lean * (pitch_rate**2)

    def count_bad(self, loc):
        n_nan_loc   = torch.isnan(loc).sum().item()
        n_inf_loc   = torch.isinf(loc).sum().item()
        return (n_nan_loc, n_inf_loc)

    # def _reward_thigh_angle(self):
    #     # Rear thighs
    #     thighs_joint_indices = [7, 10]
    #     angle_req = 90.0 * torch.pi / 180.0
    #     thighs_dof_pos = torch.tensor([angle_req, angle_req],  device=self.device, dtype=gs.tc_float)
    #     dof_pos_error = torch.sum(torch.square(self.dof_pos[:, thighs_joint_indices] - thighs_dof_pos), dim=-1)
    #     return torch.exp(-dof_pos_error / self.cfg.rewards.thigh_angle_sigma)

    # def _reward_calf_angle(self):
    #     calfs_joint_indices = [8, 11]
    #     angle_req = 10.0 * torch.pi / 180.0

    #     calfs_dof_pos = torch.tensor([angle_req, angle_req],  device=self.device, dtype=gs.tc_float)
    #     dof_pos_error = torch.sum(torch.abs(self.dof_pos[:, calfs_joint_indices] - calfs_dof_pos), dim=-1)
    #     return torch.exp(-dof_pos_error / self.cfg.rewards.calf_angle_sigma)

    def _reward_orientation(self):
        pitch_error = torch.square(self.base_axis_fwd[:, 2] - self.fixed_pitch_target.squeeze(1))
        tracking_reward = torch.exp(-pitch_error / self.cfg.rewards.euler_tracking_sigma)
        return tracking_reward

    # def _reward_planar_grf(self):
    #     """
    #     Align net ground-reaction force in the support with the commanded planar direction.
    #     Works for any (cmd_x, cmd_y). Zero when no contact or no command.
    #     """
    #     eps = 1e-8
    #     # Desired direction in WORLD from commands and current body axes
    #     cmd_xy = self.commands[:, :2]                       # [N,2] (x=fwd, y=lat)
    #     cmd_mag = torch.norm(cmd_xy, dim=1)                 # [N]

    #     # Build desired world vector in robot's sagittal/lateral plane
    #     d_world = (cmd_xy[:, 0].unsqueeze(1) * self.base_axis_dn) + \
    #               (cmd_xy[:, 1].unsqueeze(1) * self.base_axis_lat)     # [N,3]
    #     d_dir = torch.nn.functional.normalize(d_world + eps, dim=1)    # [N,3]

    #     # Net contact force in WORLD (sum feet that are in contact)
    #     f_rl = self.link_contact_forces[:, self.foot_index_rl, :]
    #     f_rr = self.link_contact_forces[:, self.foot_index_rr, :]

    #     # contact gating (avoid noise)
    #     thr = 1.0
    #     c_rl = (torch.norm(f_rl, dim=1) > thr).float().unsqueeze(1)
    #     c_rr = (torch.norm(f_rr, dim=1) > thr).float().unsqueeze(1)

    #     F_sum = f_rl * c_rl + f_rr * c_rr      # [N,3]
    #     any_contact = ((c_rl + c_rr).sum(dim=1) > 0).float() # [N]

    #     # Remove the "up" (Real UP for feet) component; keep force in sagittal-lateral plane
    #     up_w = -self.base_axis_dn                                          # [N,3]
    #     F_up = torch.sum(F_sum * up_w, dim=1, keepdim=True) * up_w
    #     F_planar = F_sum - F_up                                            # [N,3]
    #     F_norm = torch.norm(F_planar, dim=1) + eps                         # [N]

    #     # Projection along desired direction (positive = propulsive in commanded direction)
    #     proj = torch.sum(F_planar * d_dir, dim=1)                          # [N]
    #     proj_pos = torch.clamp(proj, min=0.0)                              # [N]

    #     # Alignment (cosine in [0,1])
    #     cos = proj / F_norm
    #     align = torch.clamp((cos + 1.0) * 0.5, 0.0, 1.0)                   # [N]

    #     # Gate by command magnitude (0 when near zero command), and by contact
    #     # k sets how much force is "enough" before saturating; tune k
    #     k = 0.05
    #     mag_term = torch.tanh(k * proj_pos)                                # [N]
    #     gate = torch.tanh(2.0 * cmd_mag) * any_contact                     # [N] smooth gate

    #     # Optional: softly penalize lateral component (force orthogonal to d_dir)
    #     perp = torch.norm(F_planar - proj.unsqueeze(1) * d_dir, dim=1)     # [N]
    #     sigma = 50.0
    #     lateral_suppression = torch.exp(-(perp * perp) / sigma)            # [N]

    #     # Final reward
    #     return mag_term * align * lateral_suppression * gate

    # def _reward_com_ahead_of_support(self):
    #     """
    #     Encourage COM to lead the support point along the commanded planar direction.
    #     Works for any (cmd_x, cmd_y). Uses world XY plane.
    #     """
    #     eps = 1e-8

    #     # --- Support midpoint (use contacts if available; else fallback to hind feet) ---
    #     thr = 1.0
    #     # Feet world positions (XY)
    #     p_rl = self.feet_pos[:, 2, :2]
    #     p_rr = self.feet_pos[:, 3, :2]

    #     # Contact gates
    #     fmag = torch.norm(self.link_contact_forces, dim=-1)  # [N, n_links]
    #     c_rl = (fmag[:, self.foot_index_rl] > thr).float().unsqueeze(1)
    #     c_rr = (fmag[:, self.foot_index_rr] > thr).float().unsqueeze(1)

    #     # Weighted midpoint over contacting feet; if none, use hind feet midpoint
    #     num_c = c_rl + c_rr  # [N,1]
    #     sum_xy = p_rl * c_rl + p_rr * c_rr
    #     p_mid_contacts = torch.where(num_c > 0, sum_xy / (num_c + eps), 0.5 * (p_rl + p_rr))

    #     # --- Desired planar direction from commands in WORLD ---
    #     # Map cmd_x -> base_axis_dn (forward = -Z), cmd_y -> base_axis_lat (+Y)
    #     cmd_xy = self.commands[:, :2]                                     # [N,2]
    #     cmd_mag = torch.norm(cmd_xy, dim=1)                               # [N]
    #     d_world = (cmd_xy[:, 0].unsqueeze(1) * self.base_axis_dn) + \
    #               (cmd_xy[:, 1].unsqueeze(1) * self.base_axis_lat)        # [N,3]

    #     # Project to XY plane and normalize
    #     d_xy = torch.nn.functional.normalize(d_world[:, :2] + eps, dim=1) # [N,2]

    #     # --- COM lead along commanded direction ---
    #     com_xy = self.robot_com[:, :2]
    #     lead = torch.sum((com_xy - p_mid_contacts) * d_xy, dim=1)         # signed meters

    #     # Target lead grows with command magnitude (0.03–0.15 m typical)
    #     target = 0.03 + 0.12 * torch.tanh(2.0 * cmd_mag)                  # [N]

    #     # Gate the reward when command is ~zero and when no contact
    #     gate_cmd = torch.tanh(4.0 * cmd_mag)                              # [N]
    #     any_contact = (num_c.squeeze(1) > 0).float()                      # [N]

    #     # Shape the error → reward
    #     sigma = 0.03   # ≈ (5.5 cm)^2
    #     rew = torch.exp(-((lead - target)**2) / sigma)

    #     return rew * gate_cmd * any_contact


    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.base_pos[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        rew = torch.square(base_height - self.base_height_target.squeeze(1))
        return torch.exp(-rew / self.cfg.rewards.base_height_tracking_sigma)

    def _reward_foot_step(self):
        rl_force = torch.norm(self.link_contact_forces[:, self.foot_index_rl, :], dim=-1).view(-1, 1)
        rl_speed = torch.norm(self.feet_vel[:, 2, :], dim=-1).view(-1, 1)
        rl_pos = self.feet_pos[:, 2, :]
        rr_force = torch.norm(self.link_contact_forces[:, self.foot_index_rr, :], dim=-1).view(-1, 1)
        rr_speed = torch.norm(self.feet_vel[:, 3, :], dim=-1).view(-1, 1)
        rr_pos = self.feet_pos[:, 3, :]

        lat = self.base_axis_lat[:]
        fwd = self.base_axis_dn[:]
        vx, vy = self.commands[:, 0], self.commands[:, 1]

        angle = torch.atan2(vy, vx) + torch.pi/2.0
        cos_t = torch.cos(angle).unsqueeze(1)
        sin_t = torch.sin(angle).unsqueeze(1)
        vector_length = torch.hypot(vx, vy)
        vector_length_col = vector_length.unsqueeze(1)
        dir_vec = cos_t * lat + sin_t * fwd
        dir_vec[:, 2] = 0.0
        dir_in_plane = torch.nn.functional.normalize(dir_vec, dim=1)

        d = self.cfg.commands.foot_step_distance * vector_length_col

        # Target swing-foot placement is offset from the *stance* foot
        tgt_for_left  = rl_pos + d * dir_in_plane  # when right is stance, left should step here
        tgt_for_right = rr_pos + d * dir_in_plane  # when left is stance, right should step here

        self.tgt_rl_dir = tgt_for_left
        self.tgt_rr_dir = tgt_for_right

        # Select per batch
        right_stance = (rr_force > 0.5)
        tgt  = torch.where(right_stance, tgt_for_left,  tgt_for_right)
        cur  = torch.where(right_stance, rl_pos, rr_pos)

        # Squared distance to target
        pos_error = torch.sum((tgt - cur) ** 2, dim=1) * 1.0

        exp_rl_frc = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device)
        exp_rl_spd = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device)
        exp_rr_frc = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device)
        exp_rr_spd = torch.zeros(self.num_envs, 1, dtype=gs.tc_float, device=self.device)

        phi = self.phi % 1.0

        swing_indices = (phi >= 0.0) & (phi < 0.5)
        swing_indices = swing_indices.nonzero(as_tuple=False).flatten()
        stance_indices = (phi >= 0.5) & (phi <= 1.0)
        stance_indices = stance_indices.nonzero(as_tuple=False).flatten()

        exp_rl_frc[swing_indices, :] = -1
        exp_rl_spd[swing_indices, :] = 0
        exp_rr_frc[swing_indices, :] = 0
        exp_rr_spd[swing_indices, :] = -1

        exp_rl_frc[stance_indices, :] = 0
        exp_rl_spd[stance_indices, :] = -1
        exp_rr_frc[stance_indices, :] = -1
        exp_rr_spd[stance_indices, :] = 0

        phase_reward = ((exp_rl_spd * rl_speed + exp_rl_frc * rl_force).flatten() +
                       (exp_rr_spd * rr_speed + exp_rr_frc * rr_force).flatten())

        return torch.exp(-pos_error + phase_reward)


    # def _reward_tracking_lin_vel(self):
    #     # Create a circle around up vector, the radius is the command velocity, the angle, the direction
    #     pitch_err = torch.square(self.base_axis_fwd[:, 2] - self.pitch_target.squeeze(1))

    #     # Our fwd vector is UP when biped
    #     up = self.base_axis_fwd[:]
    #     lat = self.base_axis_lat[:]
    #     fwd = self.base_axis_dn[:]

    #     vx, vy = self.commands[:, 0], self.commands[:, 1]
    #     v_mag = torch.clamp(torch.sqrt(vx*vx + vy*vy), 0.0, self.cfg.commands.max_curriculum)  # cap to achievable
    #     vel_theta = torch.atan2(vy, vx)
    #     vel_radius = torch.linalg.norm(torch.stack([vx, vy], dim=1), dim=1)

    #     cos_t = torch.cos(vel_theta).unsqueeze(1)                     # (N,1)
    #     sin_t = torch.sin(vel_theta).unsqueeze(1)                     # (N,1)
    #     dir_in_plane = torch.nn.functional.normalize(cos_t * lat + sin_t * fwd, dim=1)  # (N,3) unit in-plane

    #     circle_height = 0.6
    #     center = self.base_pos + up * circle_height
    #     circle_pt = center + dir_in_plane * vel_radius.unsqueeze(1)   # (N,3)
    #     tgt_dir = vec_direction(self.base_pos, circle_pt) * v_mag.unsqueeze(1)


    #     vel = self.robot.get_vel()[:, :3]
    #     vel_dir = torch.nn.functional.normalize(vel + 1e-8, dim=1)

    #     self.vel_tgt = tgt_dir
    #     self.vel_dir = vel_dir
    #     aligned = (vel_dir * tgt_dir).sum(dim=1).clamp(-1.0, 1.0)
    #     angle_error = torch.acos(aligned)
    #     normalized_error = angle_error / torch.pi
    #     rew_angle = torch.exp(-normalized_error/self.cfg.rewards.tracking_sigma)
    #     stability = torch.exp(-pitch_err / 0.9)
    #     rew = rew_angle * stability
    #     return rew

    # GPT suggestion
    def _reward_tracking_ang_vel(self):
        up = self.base_axis_fwd                           # biped up in world
        # base_ang_vel is in base frame in your code; get world ang vel instead:
        w_world = self.robot.get_ang()                    # [N,3] world
        yaw_rate = torch.sum(w_world * up, dim=1)         # project onto up axis

        cmd_yaw = self.commands[:, 2]
        err = (cmd_yaw - yaw_rate)**2
        posture = torch.exp(-((self.base_axis_fwd[:, 2] - self.fixed_pitch_target.squeeze(1))**2) / 0.1)
        return torch.exp(-err / self.cfg.rewards.tracking_sigma) * posture

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)

        # Ensure the z component of base_lin_vel is negative
        frame_lin_vel = self.base_lin_vel[:, :].clone()
        frame_lin_vel[:, 0] = -self.base_lin_vel[:, 2]
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - frame_lin_vel[:, :2]), dim=1)

        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)


    # def _reward_tracking_ang_vel(self):
    #     # z yaw is the same axis; base frame is fine
    #     pitch_err = torch.square(self.base_axis_fwd[:, 2] - self.pitch_target.squeeze(1))

    #     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 0])
    #     rew_velocity = torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    #     stability = torch.exp(-pitch_err / 0.2)
    #     rew = rew_velocity * stability
    #     return rew

    def _contact_mag(self, idx):  # smooth contact metric
        return torch.norm(self.link_contact_forces[:, idx, :], dim=-1)

    def _reward_front_feet_off(self):
        f_fl = self._contact_mag(self.foot_index_fl)
        f_fr = self._contact_mag(self.foot_index_fr)
        x = f_fl + f_fr
        return torch.exp(-(x*x)/500.0)

    def _reward_front_feet_on_ground_push(self):
        # Get contact forces for the front left and front right feet
        f_fl = self.link_contact_forces[:, self.foot_index_fl, :]  # Contact force vector for front left foot
        f_fr = self.link_contact_forces[:, self.foot_index_fr, :]  # Contact force vector for front right foot

        # Transform forces to the robot's local frame
        inv_base_quat = gs_inv_quat(self.base_quat)
        f_fl_local = transform_by_quat(f_fl, inv_base_quat)  # Transform to local frame
        f_fr_local = transform_by_quat(f_fr, inv_base_quat)  # Transform to local frame

        # Extract the X (forward/backward) and Z (vertical) components
        f_fl_push = f_fl_local[:, 2]
        f_fr_push = f_fr_local[:, 2]

        # Reward backward force (negative X-component)
        push_reward = torch.exp(-((f_fl_push + f_fr_push)**2) / 50.0)

        # Ensure reward is only applied when feet are in contact
        contact_fl = self._contact_mag(self.foot_index_fl) > 1.0
        contact_fr = self._contact_mag(self.foot_index_fr) > 1.0
        contact = contact_fl | contact_fr  # Either foot in contact

        rew = torch.where(contact.float() > 0.0, push_reward * contact.float(), 1.0)

        # Apply reward only when in contact
        return rew

    def _reward_front_arm_angle(self):
        pitch_err = torch.square(self.base_axis_fwd[:, 2] - self.fixed_pitch_target.squeeze(1))

        inv_base_quat = gs_inv_quat(self.base_quat)
        foot_fl_local = transform_by_quat(self.feet_pos[:, 0, :] - self.base_pos, inv_base_quat)
        foot_fr_local = transform_by_quat(self.feet_pos[:, 1, :] - self.base_pos, inv_base_quat)

        forward_local = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand_as(foot_fl_local)  # biped forward = -Z
        angle_fl = angle_between_vectors(foot_fl_local, forward_local)
        angle_fr = angle_between_vectors(foot_fr_local, forward_local)

        angle_req = 10.0 * torch.pi / 180.0
        #err = torch.where(pitch_err < 0.5, ((angle_fl - angle_req)**2 + (angle_fr - angle_req)**2), 0.0)
        err = ((angle_fl - angle_req)**2 + (angle_fr - angle_req)**2)
        return torch.exp(-err / self.cfg.rewards.front_arm_angle_sigma)

    def _reward_hind_alternation(self):
        # encourage alternating single support (one foot on, other off)
        f_rl = self._contact_mag(self.foot_index_rl) > 7.0
        f_rr = self._contact_mag(self.foot_index_rr) > 7.0
        both  = (f_rl & f_rr).float()
        none  = (~f_rl & ~f_rr).float()
        #single = (f_rl ^ f_rr).float()

        phi = (self.phi % 1.0).squeeze(-1)
        # Select which foot is expected to be on ground
        single = torch.where(phi < 0.5, f_rl.float(), f_rr.float())

        # reward exactly one in contact (and softly penalize both/none)
        # if no commands (standing still) then reward both feet
        cmd_xy = self.commands[:, :2]
        moving = torch.logical_or(
            torch.norm(cmd_xy, dim=1) >= self.cfg.commands.min_normal,
            torch.logical_or(
                torch.abs(self.base_ang_vel[:, 0]) > 0.05,
                torch.logical_or(torch.abs(self.base_lin_vel[:, 1]) > 0.05, # y
                                 torch.abs(self.base_lin_vel[:, 2]) > 0.05) # z (front)
            )
        )
        rew_standing = 1.0*both - 0.1*none - 0.1*single
        rew_moving = 1.0*single - 0.1*both - 0.1*none
        rew = torch.where(moving, rew_moving, rew_standing)

        return rew

    def _reward_hind_foot_clearance(self):
        z = self.feet_pos[:, 2:, 2]   # RL, RR z
        vxy = torch.norm(self.feet_vel[:, 2:, :2], dim=-1)
        err = torch.sum(vxy * (z - (self.foot_clearance_target + self.cfg.rewards.foot_height_offset))**2, dim=-1)
        return torch.exp(-err / self.cfg.rewards.foot_clearance_tracking_sigma)

    def _reward_com_over_support(self):
        p_rl = self.feet_pos[:, 2, :2]
        p_rr = self.feet_pos[:, 3, :2]
        com  = self.robot_com[:, :2]
        # targets
        p_mid = 0.5*(p_rl + p_rr)
        d = torch.norm(com - p_mid, dim=-1)
        return torch.exp(-(d*d) / 0.03)      # sigma ≈ 0.173 m
