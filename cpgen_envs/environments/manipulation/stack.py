from cpgen_envs.environments.manipulation.real_env import RealEnvMixin
from cpgen_envs.models.arenas.square_arena import TableArena, TableArenaReal
from mimicgen.envs.robosuite.stack import StackThree_D1 as StackThree_D1_MG
from cpgen_envs.environments.manipulation.single_arm_env_mg import SingleArmEnv_MG

import numpy as np

from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler


class StackThree_D1(StackThree_D1_MG):
    def edit_model_xml(self, xml_str):
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

class StackThreeWide(StackThree_D1):
    """
    Extends StackThree_D1 with random scaling of objects.
    """

    def _get_object_scale_bounds(self):
        """
        Internal function to get bounds for randomization of object scales.
        Returns a dictionary with object names mapping to their scale bounds.
        """
        return {
            "block1": dict(scale_min=0.6, scale_max=1.4),
            "block2": dict(scale_min=0.6, scale_max=1.4),
            "block3": dict(scale_min=0.6, scale_max=1.4),
        }

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv_MG._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        bounds = self._get_object_scale_bounds()
        scale_A = np.random.uniform(bounds["block1"]["scale_min"], bounds["block1"]["scale_max"], size=3)
        scale_B = np.random.uniform(bounds["block2"]["scale_min"], bounds["block2"]["scale_max"], size=3)
        scale_C = np.random.uniform(bounds["block3"]["scale_min"], bounds["block3"]["scale_max"], size=3)

        cube_A_base_size = [0.02, 0.02, 0.02]
        cube_B_base_size = [0.025, 0.025, 0.025]
        cube_C_base_size = [0.02, 0.02, 0.02]
        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[s * b for s, b in zip(scale_A, cube_A_base_size)],
            size_max=[s * b for s, b in zip(scale_A, cube_A_base_size)],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[s * b for s, b in zip(scale_B, cube_B_base_size)],
            size_max=[s * b for s, b in zip(scale_B, cube_B_base_size)],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        self.cubeC = BoxObject(
            name="cubeC",
            size_min=[s * b for s, b in zip(scale_C, cube_C_base_size)],
            size_max=[s * b for s, b in zip(scale_C, cube_C_base_size)],
            rgba=[1, 0, 0, 1],
            material=bluewood,
        )
        cubes = [self.cubeA, self.cubeB, self.cubeC]
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.10, 0.10],
                y_range=[-0.10, 0.10],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

def convert_opencv_to_opengl(rot_mat_opencv: np.ndarray) -> np.ndarray:
    rot_mat_opengl = rot_mat_opencv.copy()
    rot_mat_opengl[:3, 1] *= -1
    rot_mat_opengl[:3, 2] *= -1
    return rot_mat_opengl

from robosuite.utils.mjcf_utils import xml_path_completion

class StackThreeReal(RealEnvMixin, StackThreeWide):
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

    def _check_lifted(self, body_id, margin=0.04):
        # lifting is successful when the cube is above the table top by a margin
        body_pos = self.sim.data.body_xpos[body_id]
        body_height = body_pos[2]
        table_height = self.TABLE_OFFSET[2]
        body_lifted = body_height > table_height + margin
        return body_lifted

    def _check_cubeA_lifted(self):
        return self._check_lifted(self.cubeA_body_id, margin=0.04)

    def _check_cubeA_stacked(self):
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_lifted = self._check_cubeA_lifted()
        return (not grasping_cubeA) and cubeA_lifted
    
    def _check_cubeB_lifted(self):
        return self._check_lifted(self.cubeB_body_id, margin=0.04)

    def _check_cubeB_stacked(self):
        grasping_cubeB = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeB)
        cubeB_lifted = self._check_cubeB_lifted()
        return (not grasping_cubeB) and cubeB_lifted

    def _check_cubeC_lifted(self):
        return self._check_lifted(self.cubeC_body_id, margin=0.04)

    def _check_success(self):
        # Count how many cubes are stacked
        stacked_cubes = sum([
            self._check_cubeA_stacked(),
            self._check_cubeB_stacked(),
            self._check_cubeC_stacked()
        ])
        # Return True if at least 2 cubes are stacked
        return stacked_cubes >= 2

    def _check_cubeC_stacked(self):
        grasping_cubeC = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeC)
        cubeC_lifted = self._check_cubeC_lifted()
        return (not grasping_cubeC) and cubeC_lifted

    def _get_initial_placement_bounds(self):
        max_dim = 0.20
        return { 
            k : dict(
                x=(-max_dim, max_dim),
                y=(-max_dim, max_dim),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=self.OBJECT_OFFSET_FROM_ROBOT_BASE + self.ROBOT_BASE_POS,
                # assume robot base is aligned with world coordinates.
            )
            for k in ["cubeA", "cubeB", "cubeC"]
        }
