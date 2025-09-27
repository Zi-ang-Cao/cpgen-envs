# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import pathlib
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation as R

from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.models.objects import MujocoXMLObject

from cpgen_envs.environments.manipulation.single_arm_env_mg import SingleArmEnv_MG
from cpgen_envs.environments.manipulation.real_env import convert_opencv_to_opengl

from mimicgen.models.robosuite.objects import BlenderObject
import cpgen_envs
import mimicgen


class MugHanging(SingleArmEnv_MG):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="NullMount",
        initialization_noise="default",
        table_full_size=(1.4, 1.2, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        table_offset=(0.6, 0.0, 0.82),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
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
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        *args,
        **kwargs,
    ):
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.ROBOT_BASE_POS = np.array([0, 0, 0.82])
        self.OBJECT_OFFSET_FROM_ROBOT_BASE = np.array([0.45, 0, 0])
        self.table_offset = np.array([0.6, 0.0, 0.82])
        self.use_object_obs = kwargs.get("use_object_obs", True)

        full_name = robots if isinstance(robots, str) else robots[0]
        parts = full_name.split("_")
        if len(parts) == 2:
            from robosuite.utils.robot_composition_utils import create_composite_robot
            create_composite_robot(full_name, robot=parts[0], grippers=parts[1])


        self._override_init_qpos()
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
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
            renderer=renderer,
            renderer_config=renderer_config,
            *args,
            **kwargs,
        )

    def _override_init_qpos(self):
        from robosuite.models.robots.manipulators import Panda
        def new_panda_init_qpos(_):
            return np.array([0.09162, -0.198264, -0.0199, -2.473226, -0.01307, 2.3039658, 0.8480939])
        Panda.init_qpos = property(new_panda_init_qpos)

    def _get_mug(self):
        self._shapenet_id = "3143a4ac"
        base_mjcf_path = os.path.join(mimicgen.__path__[0], "models/robosuite/assets/shapenet_core/mugs")
        mjcf_path = os.path.join(base_mjcf_path, f"{self._shapenet_id}/model.xml")
        return BlenderObject(
            name="mug",
            mjcf_path=mjcf_path,
            scale=0.8,
            solimp=(0.998, 0.998, 0.001),
            solref=(0.001, 1),
            density=100,
            friction=(1, 1, 1),
            margin=0.001,
        )

    def _get_mug_hanger(self):
        return MugHangerObject(name="mug_hanger", joints=[dict(type="free", damping="0.0005")])

    def _get_objects(self):
        self.mug = self._get_mug()
        self.mug_hanger = self._get_mug_hanger()
        return [self.mug, self.mug_hanger]

    def _get_object_scale_bounds(self):
        """
        Internal function to get bounds for randomization of object scales.
        Returns a dictionary with object names mapping to their scale bounds.
        """
        return dict(
            mug=dict(
                scale_min=1.3,
                scale_max=1.7,
            ),
        )

    def _load_model(self):
        super()._load_model()

        self.robots[0].robot_model.set_base_xpos(self.ROBOT_BASE_POS)

        mujoco_arena = TableArena(
            table_full_size=(1.4, 1.2, 0.05),
            table_friction=(1.0, 5e-3, 1e-4),
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        objects = self._get_objects()
        self._get_placement_initializer()

        # Now apply random scaling to the drawer and cleanup object
        mug_bounds = self._get_object_scale_bounds()["mug"]         
        mug_scale = np.random.uniform(mug_bounds["scale_min"], mug_bounds["scale_max"], size=3)

        # Set x-y scale to be the same for both objects
        mug_scale[0] = mug_scale[1]
        self.mug.set_scale(mug_scale)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )

        self._modify_camera_view()

    def _get_placement_initializer(self):
        bounds = self._get_initial_placement_bounds()

        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="MugSampler",
                mujoco_objects=self.mug,
                x_range=bounds["mug"]["x"],
                y_range=bounds["mug"]["y"],
                rotation=bounds["mug"]["z_rot"],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["mug"]["reference"],
                z_offset=0.01,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="HangerSampler",
                mujoco_objects=self.mug_hanger,
                x_range=bounds["hanger"]["x"],
                y_range=bounds["hanger"]["y"],
                rotation=bounds["hanger"]["z_rot"],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["hanger"]["reference"],
                z_offset=0.01,
            )
        )

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects.
        Returns a dictionary with the following structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame
        """
        return dict(
            mug=dict(
                x=(-0.1, 0.15),
                y=(-0.25, -0.1),
                z_rot=(0.0, 2.0 * np.pi),
                reference=self.ROBOT_BASE_POS + self.OBJECT_OFFSET_FROM_ROBOT_BASE,
            ),
            hanger=dict(
                x=(0.25, 0.35),  # min of 0 to visibility
                y=(0.0, 0.15),
                z_rot=(-np.pi/6 + np.pi/2, np.pi/6 + np.pi/2),
                reference=self.ROBOT_BASE_POS + self.OBJECT_OFFSET_FROM_ROBOT_BASE,
            ),
        )

    def _setup_references(self):
        SingleArmEnv_MG._setup_references(self)
        self.obj_body_id = dict(
            mug=self.sim.model.body_name2id(self.mug.root_body),
            hanger=self.sim.model.body_name2id(self.mug_hanger.root_body),
        )

    def _reset_internal(self):
        super(SingleArmEnv_MG, self)._reset_internal()

        if not self.deterministic_reset:
            placements = self.placement_initializer.sample()
            for pos, quat, obj in placements.values():
                try:
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([pos, quat]))
                except Exception as e:
                    print(f"Error setting joint qpos: {e}")
                    import ipdb; ipdb.set_trace()
                    print(f"Error setting joint qpos: {e}")

    def _check_success(self, success_thresh: float = 0.05, verbose: bool = False) -> bool:
        """
        Check if the mug is successfully placed on the hanger.
        
        The mug is considered successfully placed on top left or top right branch of hanger if:
        1. It is in contact with the branch of the hanger.
        2. It is within a threshold distance from the target position of the branch.

        Hardcoded value to tune: success_thresh = 0.04 (4 cm)
        """
        # Get target positions
        target_right = self.sim.data.geom("mug_hanger_branch_top_right").xpos
        target_left = self.sim.data.geom("mug_hanger_branch_top_left").xpos
        mug_pos = self.sim.data.body("mug_main").xpos

        # Identify geoms
        mug_geoms = [g for g in range(self.sim.model.ngeom) if "mug_g" in self.sim.model.geom_id2name(g)]
        branch_right = self.sim.model.geom_name2id("mug_hanger_branch_top_right")
        branch_left = self.sim.model.geom_name2id("mug_hanger_branch_top_left")

        # Track contact with each branch
        contact_right, contact_left = False, False
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            if (geom1 in mug_geoms and geom2 == branch_right) or (geom2 in mug_geoms and geom1 == branch_right):
                contact_right = True
            if (geom1 in mug_geoms and geom2 == branch_left) or (geom2 in mug_geoms and geom1 == branch_left):
                contact_left = True

        close_right = np.linalg.norm(mug_pos - target_right) < success_thresh
        close_left = np.linalg.norm(mug_pos - target_left) < success_thresh
        if verbose:
            print(f"close_right: {close_right}, close_left: {close_left}")
            print(f"dist left: {np.linalg.norm(mug_pos - target_left)}")
        return (close_right and contact_right) or (close_left and contact_left)


    def reward(self, action=None):
        reward = 1.0 if self._check_success() else 0.0
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def _get_vis_target_object(self):
        return self.mug

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs:
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            @sensor(modality=modality)
            def dummy(obs_cache):
                return np.zeros(1)

            observables["dummy"] = Observable(
                name="dummy",
                sensor=dummy,
                sampling_rate=self.control_freq,
            )
        return observables

    def _modify_camera_view(self):
        self.AGENTVIEW_CAM_POS_IN_ROBOT_FRAME = np.array([1.15, -0.042, 0.55])
        self.AGENTVIEW_EULER_XYZ = np.array([-135, 0, 90])
        self.agentview_camera_rot_mat_opencv = R.from_euler("xyz", self.AGENTVIEW_EULER_XYZ, degrees=True).as_matrix()
        self.agentview_camera_rot_mat_opengl = convert_opencv_to_opengl(self.agentview_camera_rot_mat_opencv)
        quat = R.from_matrix(self.agentview_camera_rot_mat_opengl).as_quat()
        agentview_quat = quat[[3, 0, 1, 2]]
        self.HAND_CAM_POS = np.array([0.064, 0.0325, 0.05])
        self.HAND_CAM_EULER_XYZ = np.array([-180, 0, 90])
        hand_quat = R.from_euler("XYZ", self.HAND_CAM_EULER_XYZ, degrees=True).as_quat()[[3, 0, 1, 2]]

        self.camera_info = {
            "agentview": {
                "pos": self.AGENTVIEW_CAM_POS_IN_ROBOT_FRAME + self.ROBOT_BASE_POS,
                "quat": agentview_quat,
                "camera_attribs": {"fovy": "42.7"},
            },
            "robot0_eye_in_hand": {
                "pos": self.HAND_CAM_POS,
                "quat": hand_quat,
                "camera_attribs": {"fovy": "42.7"},
            },
        }

        self._update_camera_in_model()

    def _update_camera_in_model(self):
        for name, info in self.camera_info.items():
            if "robot0" in name:
                cam_elem = self.robots[0].robot_model.worldbody.find(f".//camera[@name='{name}']")
                if cam_elem is not None:
                    cam_elem.set("pos", array_to_string(info["pos"]))
                    cam_elem.set("quat", array_to_string(info["quat"]))
                    for k, v in info["camera_attribs"].items():
                        cam_elem.set(k, v)
            else:
                self.model.mujoco_arena.set_camera(
                    camera_name=name,
                    pos=info["pos"],
                    quat=info["quat"],
                    camera_attribs=info["camera_attribs"],
                )



class MugHangerObject(MujocoXMLObject):
    """
    A mug hanger object built from MuJoCo XML.
    
    Data:
      - Total height: ~0.35 m (from base to branch endpoints).
      - Horizontal radius: 0.10 m (includes clearance for branches).
      
    These parameters enable correct placement sampling and collision setup.
    """
    def __init__(self, name: str, joints: Any = None):
        xml_path = pathlib.Path(cpgen_envs.__file__).parent / "models/assets/objects/mug_hanger.xml"
        super().__init__(
            str(xml_path),  # Convert Path to string for compatibility
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False
        )

    @property
    def bottom_offset(self) -> np.ndarray:
        # The lower extent of the mug hanger is at the base: z = 0.
        return np.array([0, 0, 0])

    @property
    def top_offset(self) -> np.ndarray:
        # The top reaches ~0.33736 m; here we round up to 0.35 m.
        return np.array([0, 0, 0.35])

    @property
    def horizontal_radius(self) -> float:
        # The base radius is 0.08 m, but branches extend outward.
        # Adding clearance yields a horizontal radius of ~0.10 m.
        return 0.10
