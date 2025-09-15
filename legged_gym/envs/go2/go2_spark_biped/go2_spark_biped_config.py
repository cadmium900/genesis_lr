from legged_gym.envs.base.base_config import BaseConfig

class GO2SparkBipedCfg(BaseConfig):

    class env:
        episode_length_s = 20 # episode length in seconds
        num_envs = 4096
        env_spacing = 1.0
        num_actions = 12
        # observation history
        frame_stack = 5   # policy frame stack
        c_frame_stack = 5  # critic frame stack
        num_single_obs = 57
        num_observations = int(num_single_obs * frame_stack)
        single_num_privileged_obs = 94
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        send_timeouts = True
        debug = False
        debug_viz = False

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield
        plane_length = 200.0 # [m]. plane size is 200x200x10 by default
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        curriculum = False
        friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 1 # starting curriculum state
        terrain_length = 6.0
        terrain_width = 6.0
        num_rows = 4  # number of terrain rows (levels)
        num_cols = 4  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75

    class commands:
        curriculum = True
        heading_command = True  # if true: compute ang vel command from heading error
        resampling_time = 3.  # time before command are changed[s]
        max_curriculum = 1.
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        curriculum_front_feet_thrs = 0.8
        curriculum_orientation_thrs = 0.8
        curriculum_threshold = 0.8
        class ranges:
            lin_vel_x = [0.0, 0.4]
            lin_vel_y = [-0.1, 0.1]
            ang_vel_yaw = [-0.2, 0.2]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 0.62]  # x,y,z [m]
        rot = [1.0, 0.0, -0.8, 0.0] # w, x, y, z [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        yaw_angle_range = [0., 3.14]  # min max [rad]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.5,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.5,   # [rad]

            'FL_calf_joint': -0.8,   # [rad]
            'RL_calf_joint': -0.5,    # [rad]
            'FR_calf_joint': -0.8,  # [rad]
            'RR_calf_joint': -0.5,    # [rad]
        }

    class control:
        # PD Drive parameters:
        control_type = 'P' # P: position, V: velocity, T: torques
        # control_type = 'P'
        stiffness = {'joint': 30.}   # [N*m/rad]
        damping = {'joint': 0.6}     # [N*m*s/rad]
        action_scale = 0.25  # action scale: target angle = actionScale * action + defaultAngle
        dt = 0.02  # control frequency 50Hz
        decimation = 4  # decimation: Number of control action updates @ sim DT per policy DT

    class asset:
        name = "go2" # name of the robot
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        dof_names = [        # specify the sequence of actions
            'FR_hip_joint',
            'FR_thigh_joint',
            'FR_calf_joint',
            'FL_hip_joint',
            'FL_thigh_joint',
            'FL_calf_joint',
            'RR_hip_joint',
            'RR_thigh_joint',
            'RR_calf_joint',
            'RL_hip_joint',
            'RL_thigh_joint',
            'RL_calf_joint',]
        foot_name = ["foot"]
        penalize_contacts_on = ["thigh", "calf", "hip", "base"]
        terminate_after_contacts_on = ["base"]
        links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        self_collisions = True
        fix_base_link = False

    class domain_rand:
        enable = True
        randomize_friction = enable
        friction_range = [0.2, 1.7]
        randomize_base_mass = enable
        added_mass_range = [-1., 1.]
        push_robots = enable
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_com_displacement = enable
        com_displacement_range = [-0.05, 0.05]
        randomize_ctrl_delay = False
        ctrl_delay_step_range = [0, 1]
        randomize_pd_gain = enable
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_joint_armature = enable
        joint_armature_range = [0.015, 0.025]  # [N*m*s/rad]
        randomize_joint_stiffness = enable
        joint_stiffness_range = [0.01, 0.02]
        randomize_joint_damping = enable
        joint_damping_range = [0.25, 0.3]

    class rewards:
        soft_dof_pos_limit = 0.9
        foot_height_offset = 0.022 # height of the foot coordinate origin above ground [m]
        euler_tracking_sigma = 0.5
        only_positive_rewards = True
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_torque_limit = 1.
        base_height_target = 1.
        foot_clearance_target = 0.08 # desired foot clearance above ground [m]
        foot_clearance_tracking_sigma = 0.01

        class scales:
            front_feet_off = 1.0
            hind_double_support = 0.6        # curriculum: start with this ON…
            hind_alternation = 0.8           # …then raise this and lower double_support
            com_over_support = 1.0
            tracking_lin_vel = 0.5
            orientation = 1.5

            dof_pos_limits = -10.0
            collision = -1.0
            ang_vel_xy = -0.05
            dof_vel = -5e-4
            dof_acc = -2e-7
            action_rate = -0.003
            action_smoothness = -0.03
            hip_pos = -0.9
            termination = -0.0

        class behavior_params_range:
            resampling_time = 6.0
            gait_period_range = [0.40, 0.55]
            pitch_target_range = [0.9, 1.0]

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 20.
        clip_actions = 10.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.03
            dof_vel = 0.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [2, 2, 2]       # [m]
        lookat = [0., 0, 1.]  # [m]
        rendered_envs_idx = [i for i in range(15)]  # number of environments to be rendered
        add_camera = False

    class sim:
        gravity = [0., 0. ,-9.81]  # [m/s^2]

class GO2SparkBipedCfgPPO():
    seed = 0
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        entropy_coef = 0.01
        num_mini_batches = 4
        num_learning_epochs = 5
        clip_param = 0.2
        schedule = 'adaptive' # could be adaptive, fixed
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        learning_rate = 5.e-4
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        run_name = 'spark'
        experiment_name = 'go2_spark_biped'
        save_interval = 500
        load_run = "Sep14_11-55-07_spark"
        checkpoint = -1
        max_iterations = 4000
        num_steps_per_env = 24
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        # load and resume
        resume = False
        resume_path = None # updated from load_run and chkpt


