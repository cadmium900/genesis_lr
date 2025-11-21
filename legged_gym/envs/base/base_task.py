import sys
import numpy as np
import torch
import time
import genesis as gs

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_device, headless):

        self.render_fps = 50
        self.last_frame_time = 0

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            assert sim_device in ["cpu", "cuda"]
            self.device = torch.device(gs.device) if sim_device == "cuda" else torch.device("cpu")
            #self.device = torch.device(sim_device)
        self.headless = headless

        self.num_envs = 1 if cfg.env.num_envs == 0 else cfg.env.num_envs
        self.num_build_envs = self.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_int)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_int)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=gs.tc_float)
        else:
            self.privileged_obs_buf = None

        self.extras = {}
        self.floating_camera = None

        self.create_sim()

    def set_camera(self, pos, lookat):
        """ Set camera position and direction
        """
        self.floating_camera.set_pose(
            pos=pos,
            lookat=lookat
        )

    def get_camera(self):
        return self.floating_camera

    # ------------- Callbacks --------------
    def _setup_camera(self):
        ''' Set camera position and direction
        '''
        self.floating_camera = self.scene.add_camera(
            res= (1280, 960),
            pos=np.array(self.cfg.viewer.pos),
            lookat=np.array(self.cfg.viewer.lookat),
            fov=30,
            GUI=True,
        )
        if self.floating_camera is None:
            print("Failed to create floating_camera")

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError