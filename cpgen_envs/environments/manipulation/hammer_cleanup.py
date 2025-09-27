# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Contains environments for BUDS hammer place task from robosuite task zoo repo.
(https://github.com/ARISE-Initiative/robosuite-task-zoo)
"""

from copy import deepcopy
import pathlib

import numpy as np

from mimicgen.models.robosuite.objects import DrawerObject
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.objects import MujocoXMLObject
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import HammerObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.buffers import RingBuffer
from robosuite.utils.mjcf_utils import CustomMaterial, add_material, array_to_string
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)
from robosuite.utils.transform_utils import mat2quat

import cpgen_envs
from cpgen_envs.environments.manipulation.single_arm_env_mg import SingleArmEnv_MG
from cpgen_envs.environments.manipulation.real_env import RealEnvMixin
from cpgen_envs.models.arenas.square_arena import TableArenaReal


class SingleArmEnv(ManipulationEnv):
    """
    A manipulation environment intended for a single robot arm.
    """

    def _load_model(self):
        """
        Verifies correct robot model is loaded
        """
        super()._load_model()

        # Verify the correct robot has been loaded
        # assert isinstance(
        #     self.robots[0], SingleArm
        # ), "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable

        Args:
            robots (str or list of str): Robots to instantiate within this env
        """
        super()._check_robot_configuration(robots)
        if type(robots) is list:
            assert (
                len(robots) == 1
            ), "Error: Only one robot should be inputted for this task!"

    @property
    def _eef_xpos(self):
        """
        Grabs End Effector position

        Returns:
            np.array: End effector(x,y,z)
        """
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])

    @property
    def _eef_xmat(self):
        """
        End Effector orientation as a rotation matrix
        Note that this draws the orientation from the "ee" site, NOT the gripper site, since the gripper
        orientations are inconsistent!

        Returns:
            np.array: (3,3) End Effector orientation matrix
        """
        pf = self.robots[0].gripper.naming_prefix

        if self.env_configuration == "bimanual":
            return np.array(
                self.sim.data.site_xmat[
                    self.sim.model.site_name2id(pf + "right_grip_site")
                ]
            ).reshape(3, 3)
        else:
            return np.array(
                self.sim.data.site_xmat[self.sim.model.site_name2id(pf + "grip_site")]
            ).reshape(3, 3)

    @property
    def _eef_xquat(self):
        """
        End Effector orientation as a (x,y,z,w) quaternion
        Note that this draws the orientation from the "ee" site, NOT the gripper site, since the gripper
        orientations are inconsistent!

        Returns:
            np.array: (x,y,z,w) End Effector quaternion
        """
        return mat2quat(self._eef_xmat)


import robosuite.utils.transform_utils as T


class HammerPlaceEnv(SingleArmEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_latch=False,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        renderer="mujoco",
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        contact_threshold=2.0,
    ):
        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.table_full_size = table_full_size
        self.table_offset = (-0.2, 0, 0.90)
        self.table_friction = table_friction

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # ee resets
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)

        # Thresholds
        self.contact_threshold = contact_threshold

        # History observations
        self._history_force_torque = None
        self._recent_force_torque = None

        self.objects = []

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            # mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the drawer is opened

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.25], proportional to the distance between drawer handle and robot arm
            - Rotating: in [0, 0.25], proportional to angle rotated by drawer handled
              - Note that this component is only relevant if the environment is using the locked drawer version

        Note that a successfully completed task (drawer opened) will return 1.0 irregardless of whether the environment
        is using sparse or shaped rewards

        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)

        if hasattr(self, "_load_arena"):
            mujoco_arena = self._load_arena()
        else:
            mujoco_arena = TableArena(
                table_full_size=self.table_full_size,
                table_offset=self.table_offset,
                table_friction=(0.6, 0.005, 0.0001),
            )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )

        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[0.5586131746834771, 0.3, 1.2903500240372423],
            quat=[
                0.4144233167171478,
                0.3100920617580414,
                0.49641484022140503,
                0.6968992352485657,
            ],
        )

        bread = CustomMaterial(
            texture="Bread",
            tex_name="bread",
            mat_name="MatBread",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
        )

        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
        )

        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
        )

        metal = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"},
        )

        tex_attrib = {"type": "cube"}

        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}

        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="handle1_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        ingredient_size = [0.03, 0.018, 0.025]

        self.sorting_object = HammerObject(
            name="hammer",
            handle_length=(0.045, 0.05),
            handle_radius=(0.012, 0.012),
            head_density_ratio=1.0,
        )

        self.cabinet_object = CabinetObject(name="CabinetObject")
        cabinet_object = self.cabinet_object.get_obj()
        cabinet_object.set("pos", array_to_string((0.2, 0.30, 0.03)))
        mujoco_arena.table_body.append(cabinet_object)

        for obj_body in [
            self.cabinet_object,
        ]:
            for material in [lightwood, darkwood, metal, redwood, ceramic]:
                tex_element, mat_element, _, used = add_material(
                    root=obj_body.worldbody,
                    naming_prefix=obj_body.naming_prefix,
                    custom_material=deepcopy(material),
                )
                obj_body.asset.append(tex_element)
                obj_body.asset.append(mat_element)

        ingredient_size = [0.015, 0.025, 0.02]

        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-pot",
                mujoco_objects=self.sorting_object,
                x_range=[0.10, 0.18],
                y_range=[-0.20, -0.13],
                rotation=(-0.1, 0.1),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.02,
            )
        )

        mujoco_objects = [
            self.sorting_object,
        ]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=mujoco_objects,
        )
        self.objects = [
            self.sorting_object,
            self.cabinet_object,
        ]
        self.model.merge_assets(self.sorting_object)
        self.model.merge_assets(self.cabinet_object)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.object_body_ids = dict()

        self.cabinet_qpos_addrs = self.sim.model.get_joint_qpos_addr(
            self.cabinet_object.joints[0]
        )

        self.sorting_object_id = self.sim.model.body_name2id(
            self.sorting_object.root_body
        )
        self.cabinet_object_id = self.sim.model.body_name2id(
            self.cabinet_object.root_body
        )

        self.obj_body_id = {}
        for obj in self.objects:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        observables["robot0_joint_pos"]._active = True

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"
            sensors = []
            names = [s.__name__ for s in sensors]

            # NEW
            # add dummy object modality
            @sensor(modality=modality)
            def dummy(obs_cache):
                return np.zeros(1)

            sensors += [dummy]
            names += ["dummy"]
            # NEW

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        pf = self.robots[0].robot_model.naming_prefix
        modality = f"{pf}proprio"

        @sensor(modality="object")
        def world_pose_in_gripper(obs_cache):
            return np.eye(4)
            # return T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))) if\
            #     f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4)

        # sensors.append(world_pose_in_gripper)
        # names.append("world_pose_in_gripper")

        # for (i, obj) in enumerate(self.objects):
        #     obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj.name, modality="object")

        #     sensors += obj_sensors
        #     names += obj_sensor_names

        # @sensor(modality=modality)
        # def gripper_contact(obs_cache):
        #     return self._has_gripper_contact

        # @sensor(modality=modality)
        # def force_norm(obs_cache):
        #     return np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias)

        # sensors += [gripper_contact, force_norm]
        # names += [f"{pf}contact", f"{pf}eef_force_norm"]

        # for name, s in zip(names, sensors):
        #     if name == "world_pose_in_gripper":
        #         observables[name] = Observable(
        #             name=name,
        #             sensor=s,
        #             sampling_rate=self.control_freq,
        #             enabled=True,
        #             active=False,
        #         )
        #     else:
        #         observables[name] = Observable(
        #             name=name,
        #             sensor=s,
        #             sampling_rate=self.control_freq
        #         )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(
                self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw"
            )

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any(
                [
                    name not in obs_cache
                    for name in [
                        f"{obj_name}_pos",
                        f"{obj_name}_quat",
                        "world_pose_in_gripper",
                    ]
                ]
            ):
                return np.zeros(3)
            obj_pose = T.pose2mat(
                (obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"])
            )
            rel_pose = T.pose_in_A_to_pose_in_B(
                obj_pose, obs_cache["world_pose_in_gripper"]
            )
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return (
                obs_cache[f"{obj_name}_to_{pf}eef_quat"]
                if f"{obj_name}_to_{pf}eef_quat" in obs_cache
                else np.zeros(4)
            )

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [
            f"{obj_name}_pos",
            f"{obj_name}_quat",
            f"{obj_name}_to_{pf}eef_pos",
            f"{obj_name}_to_{pf}eef_quat",
        ]

        return sensors, names

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj is self.cabinet_object:
                    continue
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)
        self._history_force_torque = RingBuffer(dim=6, length=16)
        self._recent_force_torque = []

    def _check_success(self):
        """
        Check if drawer has been opened.

        Returns:
            bool: True if drawer has been opened
        """
        object_pos = self.sim.data.body_xpos[self.sorting_object_id]
        object_in_drawer = 1.0 > object_pos[2] > 0.94 and object_pos[1] > 0.22

        cabinet_closed = self.sim.data.qpos[self.cabinet_qpos_addrs] > -0.01

        return object_in_drawer and cabinet_closed

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the drawer handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def step(self, action):
        if self.action_dim == 4:
            action = np.array(action)
            action = np.concatenate((action[:3], action[-1:]), axis=-1)

        self._recent_force_torque = []
        obs, reward, done, info = super().step(action)
        info["history_ft"] = np.clip(
            np.copy(self._history_force_torque.buf), a_min=None, a_max=2
        )
        info["recent_ft"] = np.array(self._recent_force_torque)
        done = self._check_success()
        return obs, reward, done, info

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)

        # self._history_force_torque.push(np.hstack((self.robots[0].ee_force - self.ee_force_bias, self.robots[0].ee_torque - self.ee_torque_bias)))
        # self._recent_force_torque.append(np.hstack((self.robots[0].ee_force - self.ee_force_bias, self.robots[0].ee_torque - self.ee_torque_bias)))

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        # print("Cabinet joint: " , self.sim.data.qpos[self.cabinet_qpos_addrs])
        # if np.linalg.norm(self.ee_force_bias) == 0:
        #     self.ee_force_bias = self.robots[0].ee_force
        #     self.ee_torque_bias = self.robots[0].ee_torque

        return reward, done, info

    @property
    def _has_gripper_contact(self):
        """
        Determines whether the gripper is making contact with an object, as defined by the eef force surprassing
        a certain threshold defined by self.contact_threshold

        Returns:
            bool: True if contact is surpasses given threshold magnitude
        """

        return (
            np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias)
            > self.contact_threshold
        )

    def get_state_vector(self, obs):
        return np.concatenate(
            [obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]
        )


class HammerCleanup_D0(HammerPlaceEnv, SingleArmEnv_MG):
    """
    Augment BUDS hammer place task for mimicgen.
    """

    def __init__(self, robot_init_qpos=None, **kwargs):
        self.robot_init_qpos = robot_init_qpos
        HammerPlaceEnv.__init__(self, **kwargs)

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def _load_model(self):
        """
        Copied exactly from HammerPlaceEnv, but swaps out the cabinet object.
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)


        if hasattr(self, "_load_arena"):
            mujoco_arena = self._load_arena()
        else:
            mujoco_arena = TableArena(
                table_full_size=self.table_full_size,
                table_offset=self.table_offset,
                table_friction=(0.6, 0.005, 0.0001),
            )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )

        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[0.5586131746834771, 0.3, 1.2903500240372423],
            quat=[
                0.4144233167171478,
                0.3100920617580414,
                0.49641484022140503,
                0.6968992352485657,
            ],
        )

        bread = CustomMaterial(
            texture="Bread",
            tex_name="bread",
            mat_name="MatBread",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
        )

        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
        )

        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
        )

        metal = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"},
        )

        tex_attrib = {"type": "cube"}

        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}

        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="handle1_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        ingredient_size = [0.03, 0.018, 0.025]

        self.sorting_object = HammerObject(
            name="hammer",
            handle_length=(0.045, 0.05),
            handle_radius=(0.012, 0.012),
            head_density_ratio=1.0,
        )

        self.cabinet_object = DrawerObject(name="CabinetObject")
        cabinet_object = self.cabinet_object.get_obj()
        cabinet_object.set("pos", array_to_string((0.2, 0.30, 0.03)))
        mujoco_arena.table_body.append(cabinet_object)

        for obj_body in [
            self.cabinet_object,
        ]:
            for material in [lightwood, darkwood, metal, redwood, ceramic]:
                tex_element, mat_element, _, used = add_material(
                    root=obj_body.worldbody,
                    naming_prefix=obj_body.naming_prefix,
                    custom_material=deepcopy(material),
                )
                obj_body.asset.append(tex_element)
                obj_body.asset.append(mat_element)

        ingredient_size = [0.015, 0.025, 0.02]

        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-pot",
                mujoco_objects=self.sorting_object,
                x_range=[0.10, 0.18],
                y_range=[-0.20, -0.13],
                rotation=(-0.1, 0.1),
                rotation_axis="y",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.02,
            )
        )

        mujoco_objects = [
            self.sorting_object,
        ]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=mujoco_objects,
        )
        self.objects = [
            self.sorting_object,
            self.cabinet_object,
        ]
        self.model.merge_assets(self.sorting_object)
        self.model.merge_assets(self.cabinet_object)


class HammerCleanup_D1(HammerCleanup_D0):
    """
    Move object and drawer with wide initialization. Note we had to make some objects movable that were fixtures before.
    """

    def _check_success(self):
        """
        Update from superclass to have a more stringent check that's not buggy
        (e.g. there's no check in x-position before) and that supports
        different drawer (cabinet) positions.
        """
        object_pos = self.sim.data.body_xpos[self.sorting_object_id]
        # object_in_drawer = 1.0 > object_pos[2] > 0.94 and object_pos[1] > 0.22

        cabinet_closed = self.sim.data.qpos[self.cabinet_qpos_addrs] > -0.01

        # new contact-based drawer check - object in contact with bottom drawer geom
        drawer_bottom_geom = "CabinetObject_drawer_bottom"
        object_in_drawer = self.check_contact(drawer_bottom_geom, self.sorting_object)

        return object_in_drawer and cabinet_closed

    def _get_sorting_object(self):
        """
        Method that constructs object to place into drawer. Subclasses can override this method to
        construct different objects.
        """
        return HammerObject(
            name="hammer",
            handle_length=(0.045, 0.05),
            handle_radius=(0.012, 0.012),
            head_density_ratio=1.0,
        )

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        return dict(
            hammer=dict(
                x=(-0.2, 0.2),
                y=(-0.25, -0.13),
                z_rot=(0.0, 2.0 * np.pi),
                reference=self.table_offset,
                init_quat=self.sorting_object.init_quat,
                # NOTE: this rotation axis needs to be y, not z because of hammer's init_quat
                rotation_axis="y",
            ),
            drawer=dict(
                x=(0.0, 0.2),
                y=(0.2, 0.3),
                # z_rot=(0., 0.),
                z_rot=(-np.pi / 6.0, np.pi / 6.0),
                reference=self.table_offset,
            ),
        )

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds
        """
        bounds = self._get_initial_placement_bounds()
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-hammer",
                mujoco_objects=self.sorting_object,
                x_range=bounds["hammer"]["x"],
                y_range=bounds["hammer"]["y"],
                rotation=bounds["hammer"]["z_rot"],
                rotation_axis=bounds["hammer"]["rotation_axis"],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["hammer"]["reference"],
                z_offset=0.02,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-drawer",
                mujoco_objects=self.cabinet_object,
                x_range=bounds["drawer"]["x"],
                y_range=bounds["drawer"]["y"],
                rotation=bounds["drawer"]["z_rot"],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["drawer"]["reference"],
                z_offset=0.03,
            )
        )

    def _load_model(self):
        """
        Update to include drawer (cabinet) in placement initializer.
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Adjust initial robot joint configuration accordingly
        if self.robot_init_qpos is not None:
            self.robots[0].init_qpos = self.robot_init_qpos

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
            table_friction=(0.6, 0.005, 0.0001),
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )

        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[0.5586131746834771, 0.3, 1.2903500240372423],
            quat=[
                0.4144233167171478,
                0.3100920617580414,
                0.49641484022140503,
                0.6968992352485657,
            ],
        )

        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
        )

        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
        )

        metal = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"},
        )

        tex_attrib = {"type": "cube"}

        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}

        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="handle1_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.sorting_object = self._get_sorting_object()

        self.cabinet_object = DrawerObject(name="CabinetObject")

        # # old: manually set position in xml and add to mujoco arena
        # cabinet_object = self.cabinet_object.get_obj()
        # cabinet_object.set("pos", array_to_string((0.2, 0.30, 0.03)))
        # mujoco_arena.table_body.append(cabinet_object)

        for obj_body in [
            self.cabinet_object,
        ]:
            for material in [lightwood, darkwood, metal, redwood, ceramic]:
                tex_element, mat_element, _, used = add_material(
                    root=obj_body.worldbody,
                    naming_prefix=obj_body.naming_prefix,
                    custom_material=deepcopy(material),
                )
                obj_body.asset.append(tex_element)
                obj_body.asset.append(mat_element)

        self._get_placement_initializer()

        mujoco_objects = [
            self.sorting_object,
            self.cabinet_object,
        ]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=mujoco_objects,
        )
        self.objects = [
            self.sorting_object,
            self.cabinet_object,
        ]
        self.model.merge_assets(self.sorting_object)
        self.model.merge_assets(self.cabinet_object)

    def _reset_internal(self):
        """
        Update to make sure placement initializer can be used to set drawer (cabinet) pose
        even though it doesn't have a joint.
        """
        HammerPlaceEnv._reset_internal(self)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            for obj_pos, obj_quat, obj in object_placements.values():
                if obj is self.cabinet_object:
                    # object is fixture - set pose in model
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    obj_pos_to_set = np.array(obj_pos)
                    obj_pos_to_set[2] = (
                        0.905  # hardcode z-value to correspond to parent class
                    )
                    self.sim.model.body_pos[body_id] = obj_pos_to_set
                    self.sim.model.body_quat[body_id] = obj_quat
                else:
                    # object has free joint - use it to set pose
                    self.sim.data.set_joint_qpos(
                        obj.joints[0],
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )

        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)
        self._history_force_torque = RingBuffer(dim=6, length=16)
        self._recent_force_torque = []

class HammerCleanupWide(HammerCleanup_D1):
    """
    Extends HammerCleanup_D1 with random scaling of objects.
    """

    def _get_object_scale_bounds(self):
        """
        Internal function to get bounds for randomization of object scales.
        Returns a dictionary with object names mapping to their scale bounds.
        """
        return dict(
            hammer=dict(
                scale_min=0.7,
                scale_max=1.2,
            ),
            drawer=dict(
                scale_min=0.7,
                scale_max=1.2,
            ),
        )

    def _get_sorting_object(self):
        """
        Override to add random scaling to hammer object.
        """
        bounds = self._get_object_scale_bounds()["hammer"]
        scale = np.random.uniform(bounds["scale_min"], bounds["scale_max"])
        
        return HammerObject(
            name="hammer",
            handle_length=(0.045 * scale, 0.05 * scale),
            handle_radius=(0.012 * scale, 0.012 * scale),
            head_density_ratio=1.0,
        )
    def _load_model(self):
        """
        Update to include drawer (cabinet) in placement initializer.
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Adjust initial robot joint configuration accordingly
        if self.robot_init_qpos is not None:
            self.robots[0].init_qpos = self.robot_init_qpos

        if hasattr(self, "_load_arena"):
            mujoco_arena = self._load_arena()
        else:
            mujoco_arena = TableArena(
                table_full_size=self.table_full_size,
                table_offset=self.table_offset,
                table_friction=(0.6, 0.005, 0.0001),
            )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )

        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[0.5586131746834771, 0.3, 1.2903500240372423],
            quat=[
                0.4144233167171478,
                0.3100920617580414,
                0.49641484022140503,
                0.6968992352485657,
            ],
        )

        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
        )

        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
        )

        metal = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"},
        )

        tex_attrib = {"type": "cube"}

        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}

        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="handle1_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        hammer_bounds = self._get_object_scale_bounds()["hammer"]
        drawer_bounds = self._get_object_scale_bounds()["drawer"]
        hammer_scale = np.random.uniform(hammer_bounds["scale_min"], hammer_bounds["scale_max"], size=3)

        self.sorting_object = HammerObject(
            name="hammer",
            handle_length=(0.045 * 1.0000, 0.05 * 1.0000),
            handle_radius=(0.012 * 1.0000, 0.012 * 1.0000),
            head_density_ratio=1.0,
        )

        self.sorting_object.set_scale(hammer_scale)
        self.cabinet_object = DrawerObject(name="CabinetObject")
        self.cabinet_object.set_scale(hammer_scale)

        self.drawer_scale = hammer_scale
        # # old: manually set position in xml and add to mujoco arena
        # cabinet_object = self.cabinet_object.get_obj()
        # cabinet_object.set("pos", array_to_string((0.2, 0.30, 0.03)))
        # mujoco_arena.table_body.append(cabinet_object)

        for obj_body in [
            self.cabinet_object,
        ]:
            for material in [lightwood, darkwood, metal, redwood, ceramic]:
                tex_element, mat_element, _, used = add_material(
                    root=obj_body.worldbody,
                    naming_prefix=obj_body.naming_prefix,
                    custom_material=deepcopy(material),
                )
                obj_body.asset.append(tex_element)
                obj_body.asset.append(mat_element)

        self._get_placement_initializer()

        mujoco_objects = [
            self.sorting_object,
            self.cabinet_object,
        ]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=mujoco_objects,
        )
        self.objects = [
            self.sorting_object,
            self.cabinet_object,
        ]
        self.model.merge_assets(self.sorting_object)
        self.model.merge_assets(self.cabinet_object)

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        bounds = super()._get_initial_placement_bounds()
        bounds["drawer"]["y"] = (0.22, 0.32)
        return bounds


class DrawerObjectReal(MujocoXMLObject):
    """
    Custom version of cabinet object that differs from BUDs. It has manually specified top, bottom, and horizontal sites,
    a slightly different material for the handle, and changed the group for the cabinet geoms from 1 to 0 because
    robosuite v1.4 enforces that geom groups with 0 participate in physics and 1 do not.
    """
    def __init__(
            self,
            name,
            joints=None):

        path_to_cabinet_xml = pathlib.Path(cpgen_envs.__file__).parent / "models/assets/objects/drawer_real.xml"
        super().__init__(path_to_cabinet_xml,
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)

    # NOTE: had to manually set these to get placement sampler working okay
    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.065])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.065])
        
    @property
    def horizontal_radius(self):
        return 0.15


class HammerCleanupReal(RealEnvMixin, HammerCleanupWide):
    def __init__(self, **kwargs):
        self._override_init_qpos()
        kwargs = self._update_kwargs_for_real_robot(kwargs)
        self._setup_real_cameras()
        super().__init__(**kwargs)

    def _load_arena(self):
        arena = TableArenaReal(
            table_full_size=self.TABLE_FULL_SIZE,
            table_friction=self.table_friction,
            table_offset=self.TABLE_OFFSET,
            has_legs=False,
            xml=xml_path_completion("arenas/table_arena.xml"),
        )
        self._adjust_table_alignment(arena)
        return arena

    def _load_model(self):
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(self.ROBOT_BASE_POS)
        self._update_camera_in_model()

    def _get_object_scale_bounds(self):
        """
        Internal function to get bounds for randomization of object scales.
        Returns a dictionary with object names mapping to their scale bounds.
        """
        return dict(
            hammer=dict(
                scale_min=0.9,
                scale_max=1.3,
            ),
            drawer=dict(
                scale_min=0.9,
                scale_max=1.3,
            ),
        )

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        return dict(
            hammer=dict(
                x=(-0.2, 0.2),
                y=(-0.28, -0.18),
                z_rot=(0.0, 2.0 * np.pi),
                reference=self.ROBOT_BASE_POS + self.OBJECT_OFFSET_FROM_ROBOT_BASE,
                init_quat=self.sorting_object.init_quat,
                # NOTE: this rotation axis needs to be y, not z because of hammer's init_quat
                rotation_axis="y",
            ),
            drawer=dict(
                x=(0.0, 0.2),
                y=(0.25, 0.35),
                # z_rot=(0., 0.),
                z_rot=(-np.pi / 6.0, np.pi / 6.0),
                reference=self.ROBOT_BASE_POS + self.OBJECT_OFFSET_FROM_ROBOT_BASE,
            ),
        )

    def _reset_internal(self):
        """
        Update to make sure placement initializer can be used to set drawer (cabinet) pose
        even though it doesn't have a joint.
        """
        HammerPlaceEnv._reset_internal(self)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            for obj_pos, obj_quat, obj in object_placements.values():
                if obj is self.cabinet_object:
                    # object is fixture - set pose in model
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    obj_pos_to_set = np.array(obj_pos)
                    obj_pos_to_set[2] = 0.82  # UPDATE to 0.805 to correspond to real robot default values
                    self.sim.model.body_pos[body_id] = obj_pos_to_set
                    self.sim.model.body_quat[body_id] = obj_quat
                else:
                    # object has free joint - use it to set pose
                    self.sim.data.set_joint_qpos(
                        obj.joints[0],
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )

        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)
        self._history_force_torque = RingBuffer(dim=6, length=16)
        self._recent_force_torque = []

    def _load_model(self):
        """
        Update to include drawer (cabinet) in placement initializer.
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Adjust initial robot joint configuration accordingly
        if self.robot_init_qpos is not None:
            self.robots[0].init_qpos = self.robot_init_qpos

        if hasattr(self, "_load_arena"):
            mujoco_arena = self._load_arena()
        else:
            mujoco_arena = TableArena(
                table_full_size=self.table_full_size,
                table_offset=self.table_offset,
                table_friction=(0.6, 0.005, 0.0001),
            )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )

        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[0.5586131746834771, 0.3, 1.2903500240372423],
            quat=[
                0.4144233167171478,
                0.3100920617580414,
                0.49641484022140503,
                0.6968992352485657,
            ],
        )

        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
        )

        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
        )

        metal = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"},
        )

        tex_attrib = {"type": "cube"}

        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}

        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="handle1_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        hammer_bounds = self._get_object_scale_bounds()["hammer"]
        drawer_bounds = self._get_object_scale_bounds()["drawer"]
        hammer_scale = np.random.uniform(hammer_bounds["scale_min"], hammer_bounds["scale_max"], size=3)

        self.sorting_object = HammerObject(
            name="hammer",
            handle_length=(0.045 * 1.0000, 0.05 * 1.0000),
            handle_radius=(0.012 * 1.0000, 0.012 * 1.0000),
            head_density_ratio=1.0,
        )

        self.sorting_object.set_scale(hammer_scale)
        self.cabinet_object = DrawerObjectReal(name="CabinetObject")
        self.cabinet_object.set_scale(hammer_scale)

        self.drawer_scale = hammer_scale
        # # old: manually set position in xml and add to mujoco arena
        # cabinet_object = self.cabinet_object.get_obj()
        # cabinet_object.set("pos", array_to_string((0.2, 0.30, 0.03)))
        # mujoco_arena.table_body.append(cabinet_object)

        for obj_body in [
            self.cabinet_object,
        ]:
            for material in [lightwood, darkwood, metal, redwood, ceramic]:
                tex_element, mat_element, _, used = add_material(
                    root=obj_body.worldbody,
                    naming_prefix=obj_body.naming_prefix,
                    custom_material=deepcopy(material),
                )
                obj_body.asset.append(tex_element)
                obj_body.asset.append(mat_element)

        self._get_placement_initializer()

        mujoco_objects = [
            self.sorting_object,
            self.cabinet_object,
        ]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=mujoco_objects,
        )
        self.objects = [
            self.sorting_object,
            self.cabinet_object,
        ]
        self.model.merge_assets(self.sorting_object)
        self.model.merge_assets(self.cabinet_object)

        # above (except for using DrawerObjectReal) is same as HammerCleanupWide's _load_model
        self.robots[0].robot_model.set_base_xpos(self.ROBOT_BASE_POS)
        self._update_camera_in_model()
