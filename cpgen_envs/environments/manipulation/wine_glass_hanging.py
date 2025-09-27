# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import pathlib
from typing import Any, Optional, Sequence, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R

from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.transform_utils import quat_multiply

from cpgen_envs.environments.manipulation.single_arm_env_mg import SingleArmEnv_MG
from cpgen_envs.environments.manipulation.real_env import convert_opencv_to_opengl

from mimicgen.models.robosuite.objects import BlenderObject
import cpgen_envs
import mimicgen



class MultiAxisUniformRandomSampler(UniformRandomSampler):
    def __init__(
        self,
        name: str,
        mujoco_objects=None,
        x_range: Tuple[float, float]=(0, 0),
        y_range: Tuple[float, float]=(0, 0),
        rotation_x_range: Optional[Sequence[float]]=None,
        rotation_y_range: Optional[Sequence[float]]=None,
        rotation_z_range: Optional[Sequence[float]]=None,
        ensure_object_boundary_in_range: bool=True,
        ensure_valid_placement: bool=True,
        reference_pos: Tuple[float, float, float]=(0, 0, 0),
        z_offset: float=0.0,
    ):
        # disable original rotation logic
        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            x_range=x_range,
            y_range=y_range,
            rotation=None,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
        )
        self.rx = rotation_x_range
        self.ry = rotation_y_range
        self.rz = rotation_z_range

    def _draw_angle(self, rng: Optional[Sequence[float]]) -> float:
        if rng is None:
            return 0
        if len(rng) == 2:
            return np.random.uniform(rng[0], rng[1])
        raise ValueError(f"rotation_*_range must be None or [min, max], got {rng}")

    def _sample_quat(self) -> np.ndarray:
        # draw half‑angles
        θx = self._draw_angle(self.rx) / 2
        θy = self._draw_angle(self.ry) / 2
        θz = self._draw_angle(self.rz) / 2

        # build per‑axis quaternions
        qx = np.array([np.cos(θx), np.sin(θx), 0.0, 0.0])
        qy = np.array([np.cos(θy), 0.0, np.sin(θy), 0.0])
        qz = np.array([np.cos(θz), 0.0, 0.0, np.sin(θz)])

        # combine in X→Y→Z order
        return quat_multiply(quat_multiply(qx, qy), qz)
    

class WineGlassHanging(SingleArmEnv_MG):
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

    def _get_wine_glass(self):
        return WineGlassObject(name="wine_glass", joints=[dict(type="free", damping="0.0005")])

    def _get_wine_glass_rack(self):
        return WineGlassRackObject(name="wine_glass_rack", joints=[dict(type="free", damping="0.0005")])

    def _get_objects(self):
        self.wine_glass = self._get_wine_glass()
        self.wine_glass_rack = self._get_wine_glass_rack()
        return [self.wine_glass, self.wine_glass_rack]

    def _get_object_scale_bounds(self):
        """
        Internal function to get bounds for randomization of object scales.
        Returns a dictionary with object names mapping to their scale bounds.
        """
        return dict(
            wine_glass=dict(
                scale_min=0.75,
                scale_max=1.0
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
        wine_glass_bounds = self._get_object_scale_bounds()["wine_glass"]         
        wine_glass_scale = np.random.uniform(wine_glass_bounds["scale_min"], wine_glass_bounds["scale_max"], size=3)

        # Set x-y scale to be the same for both objects
        wine_glass_scale[0] = wine_glass_scale[1]
        self.wine_glass.set_scale(wine_glass_scale)

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
            sampler=MultiAxisUniformRandomSampler(
                name="WineGlassSampler",
                mujoco_objects=self.wine_glass,
                x_range=bounds["wine_glass"]["x"],
                y_range=bounds["wine_glass"]["y"],
                rotation_x_range=bounds["wine_glass"]["x_rot"],
                rotation_y_range=bounds["wine_glass"]["y_rot"],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["wine_glass"]["reference"],
                z_offset=-0.06,  # because when rotated it's still high
                # hardcoded offset mightn't work for all scales
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="RackSampler",
                mujoco_objects=self.wine_glass_rack,
                x_range=bounds["rack"]["x"],
                y_range=bounds["rack"]["y"],
                rotation=bounds["rack"]["z_rot"],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["rack"]["reference"],
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
            wine_glass=dict(
                x=(-0.05, 0.1),
                y=(-0.2, -0.03),
                x_rot=(0, 2 * np.pi),
                y_rot=(np.pi/2, np.pi/2),
                reference=self.ROBOT_BASE_POS + self.OBJECT_OFFSET_FROM_ROBOT_BASE,
            ),
            rack=dict(
                x=(0.18, 0.25),  # min of 0 to visibility
                y=(0.05, 0.15),
                z_rot=(-np.pi/12 + np.pi/3, np.pi/12 + np.pi/3),
                reference=self.ROBOT_BASE_POS + self.OBJECT_OFFSET_FROM_ROBOT_BASE,
            ),
        )

    def _setup_references(self):
        SingleArmEnv_MG._setup_references(self)
        self.obj_body_id = dict(
            wine_glass=self.sim.model.body_name2id(self.wine_glass.root_body),
            rack=self.sim.model.body_name2id(self.wine_glass_rack.root_body),
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

    def _check_success(self, success_thresh: float = 0.02, verbose: bool = False) -> bool:
        """
        Check if the wine_glass is successfully placed on the rack.
        
        Conditions for success:
        1. Wine glass is in contact with any rack geom.
        2. Its local z-axis deviates less than 0.05 rad from vertical.
        3. Its xy position is within success_thresh (e.g., 4 cm) of at least one rack hole.
        
        Hardcoded tuning value: success_thresh = 0.04 (4 cm)
        """
        import numpy as np

        # 1. Wine glass pose and local z-axis (from body xmat: row-major 3x3).
        wine_pos = self.sim.data.body("wine_glass_main").xpos
        wine_rot = self.sim.data.body("wine_glass_main").xmat
        local_z = wine_rot[6:9]

        # 2. Orientation: angular error from vertical.
        vertical = np.array([0, 0, -1])  # require upside down
        dot_product = np.clip(np.dot(local_z, vertical), -1.0, 1.0)
        angle_error = np.arccos(dot_product)
        orientation_ok = angle_error < 0.2

        # 3. XY: compute each hole's world coordinates from rack's local positions.
        rack_pos = self.sim.data.body("wine_glass_rack_main").xpos
        rack_rot = np.array(self.sim.data.body("wine_glass_rack_main").xmat).reshape(3, 3)
        # Local coordinates of the rack holes from the XML:
        local_holes = [np.array([0.0000, 0.1000, 0.285]),
                    np.array([0.0866, 0.0500, 0.285]),
                    np.array([0.0866, -0.0500, 0.285]),
                    np.array([0.0000, -0.1000, 0.285]),
                    np.array([-0.0866, -0.0500, 0.285]),
                    np.array([-0.0866, 0.0500, 0.285])]

        # note: these holes would change if rack is scaled
        xy_ok = False
        min_xy_dist = np.inf
        for lh in local_holes:
            world_hole = rack_pos + rack_rot.dot(lh)
            min_xy_dist = min(min_xy_dist, np.linalg.norm(wine_pos[:2] - world_hole[:2]))
            if np.linalg.norm(wine_pos[:2] - world_hole[:2]) < success_thresh:
                xy_ok = True
                break

        # 4. Contact: check for any contact between wine glass and rack geoms.
        wine_geoms = [i for i in range(self.sim.model.ngeom)
                    if "wine_glass" in self.sim.model.geom_id2name(i)]
        rack_geoms = [i for i in range(self.sim.model.ngeom)
                    if "wine_glass_rack" in self.sim.model.geom_id2name(i)]
        contact_ok = False
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if ((contact.geom1 in wine_geoms and contact.geom2 in rack_geoms) or
                (contact.geom2 in wine_geoms and contact.geom1 in rack_geoms)):
                contact_ok = True
                break

        if verbose:
            print(f"Orientation error (rad): {angle_error:.4f}, vertical OK: {orientation_ok}")
            print(f"XY dist: {min_xy_dist}. XY proximity OK: {xy_ok}")
            print(f"Rack contact detected: {contact_ok}")

        return contact_ok and orientation_ok and xy_ok


    def reward(self, action=None):
        reward = 1.0 if self._check_success() else 0.0
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def _get_vis_target_object(self):
        return self.wine_glass

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


class WineGlassObject(MujocoXMLObject):
    """
    A wine_glass object built from MuJoCo XML.
    """
    def __init__(self, name: str, joints: Any = None):
        xml_path = pathlib.Path(cpgen_envs.__file__).parent / "models/assets/objects/wine_glass.xml"
        super().__init__(
            str(xml_path),  # Convert Path to string for compatibility
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False
        )

    @property
    def bottom_offset(self) -> np.ndarray:
        # The lower extent of the wine_glass rack is at the base: z = 0.
        return np.array([0, 0, -0.12])

    @property
    def top_offset(self) -> np.ndarray:
        # The top reaches ~0.33736 m; here we round up to 0.35 m.
        return np.array([0, 0, 0.12])

    @property
    def horizontal_radius(self) -> float:
        return 0.04

class WineGlassRackObject(MujocoXMLObject):
    """
    A wine_glass rack object built from MuJoCo XML.
    
    Data:
      - Total height: ~0.35 m (from base to branch endpoints).
      - Horizontal radius: 0.10 m (includes clearance for branches).
      
    These parameters enable correct placement sampling and collision setup.
    """
    def __init__(self, name: str, joints: Any = None):
        xml_path = pathlib.Path(cpgen_envs.__file__).parent / "models/assets/objects/wine_glass_spiral_rack.xml"
        super().__init__(
            str(xml_path),  # Convert Path to string for compatibility
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False
        )

    @property
    def bottom_offset(self) -> np.ndarray:
        # The lower extent of the wine_glass rack is at the base: z = 0.
        return np.array([0, 0, 0])

    @property
    def top_offset(self) -> np.ndarray:
        return np.array([0, 0, 0.35])

    @property
    def horizontal_radius(self) -> float:
        # The base radius is 0.08 m, but branches extend outward.
        # Adding clearance yields a horizontal radius of ~0.10 m.
        return 0.12
