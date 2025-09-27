import pathlib
import numpy as np
from typing import List
from scipy.spatial.transform import Rotation as R
from cpgen_envs.environments.manipulation.real_env import RealEnvMixin
from robosuite.environments.manipulation.nut_assembly import (
    NutAssembly, NutAssemblySquare,
)
from robosuite.models.arenas import PegsArena
from robosuite.models.arenas.pegs_arena import PegsArena
from robosuite.models.objects import RoundNutObject, SquareNutObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils import RandomizationError
from robosuite.utils.mjcf_utils import array_to_string, find_elements, string_to_array
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)
from robosuite.utils.robot_composition_utils import create_composite_robot
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers.rethink_gripper import RethinkGripperBase

import cpgen_envs
from cpgen_envs.models.arenas.square_arena import SquareArenaReal, TableArenaReal

import xml.etree.ElementTree as ET

def __init__(self, idn=None):
    GripperModel.__init__(self, pathlib.Path(cpgen_envs.__file__).parent / "models/grippers/rethink_gripper.xml", idn=idn)

RethinkGripperBase.__init__ = __init__

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel

def __init__sawyer(self, idn=0):
    ManipulatorModel.__init__(self, pathlib.Path(cpgen_envs.__file__).parent / "models/robots/sawyer/robot.xml", idn=idn)

from robosuite.models.robots.manipulators.sawyer_robot import Sawyer
Sawyer.__init__ = __init__sawyer

from cpgen_envs.environments.manipulation.single_arm_env_mg import SingleArmEnv_MG

SquareNutObject.bottom_offset = np.array([0, 0, 0.01])


class NutAssembly_D0(NutAssembly, SingleArmEnv_MG):
    """
    Augment robosuite nut assembly task for mimicgen.
    """
    def __init__(self, **kwargs):
        assert (
            "placement_initializer" not in kwargs
        ), "this class defines its own placement initializer"

        # make placement initializer here
        nut_names = ("SquareNut", "RoundNut")

        # note: makes round nut init somewhere far off the table
        bounds = self._get_initial_placement_bounds()
        nut_x_ranges = (bounds["square_nut"]["x"], bounds["round_nut"]["x"])
        nut_y_ranges = (bounds["square_nut"]["y"], bounds["round_nut"]["y"])
        nut_z_ranges = (bounds["square_nut"]["z_rot"], bounds["round_nut"]["z_rot"])
        nut_references = (bounds["square_nut"]["reference"], bounds["round_nut"]["reference"])

        placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        for nut_name, x_range, y_range, z_range, ref in zip(
            nut_names, nut_x_ranges, nut_y_ranges, nut_z_ranges, nut_references
        ):
            placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{nut_name}Sampler",
                    x_range=x_range,
                    y_range=y_range,
                    rotation=z_range,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=ref,
                    z_offset=0.02,
                )
            )

        NutAssembly.__init__(
            self, placement_initializer=placement_initializer, **kwargs
        )

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

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
            square_nut=dict(
                x=(-0.115, -0.11),
                y=(0.11, 0.225),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
            round_nut=dict(
                x=(-0.115, -0.11),
                y=(-0.225, -0.11),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
        )


class NutAssembly_D1(NutAssembly_D0, SingleArmEnv_MG):
    """
    Augment robosuite nut assembly task for mimicgen.
    """
    def __init__(self, **kwargs):
        NutAssembly_D0.__init__(self, **kwargs)

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def _reset_internal(self):
        """
        Modify from superclass to keep sampling nut locations until there's no collision with either peg.
        """
        # SingleArmEnv._reset_internal(self)
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            success = False
            for _ in range(5000):  # 5000 retries
                # Sample from the placement initializer for all objects
                object_placements = self.placement_initializer.sample()

                # ADDED: check collision with pegs and maybe re-sample
                location_valid = True

                square_nut_data = object_placements["SquareNut"]
                round_nut_data = object_placements["RoundNut"]

                square_pos, square_quat, square_obj = square_nut_data
                round_pos, round_quat, round_obj = round_nut_data

                square_radius = 0.2
                round_radius = 0.2

                # Check collision between square and round nuts
                if (
                    np.linalg.norm((square_pos[0] - round_pos[0], square_pos[1] - round_pos[1]))
                    <= max(square_radius, round_radius)
                ):
                    location_valid = False

                if not location_valid:
                    continue

                for obj_pos, obj_quat, obj in object_placements.values():
                    horizontal_radius = obj.horizontal_radius

                    peg1_id = self.sim.model.body_name2id("peg1")
                    peg1_pos = np.array(self.sim.data.body_xpos[peg1_id])

                    peg1_horizontal_radius = self.peg1_horizontal_radius
                    if (
                        np.linalg.norm(
                            (obj_pos[0] - peg1_pos[0], obj_pos[1] - peg1_pos[1])
                        )
                        <= max(peg1_horizontal_radius, horizontal_radius)
                    ):
                        location_valid = False
                        break

                    peg2_id = self.sim.model.body_name2id("peg2")
                    peg2_pos = np.array(self.sim.data.body_xpos[peg2_id])
                    peg2_horizontal_radius = self.peg2_horizontal_radius
                    if (
                        np.linalg.norm(
                            (obj_pos[0] - peg2_pos[0], obj_pos[1] - peg2_pos[1])
                        )
                        <= max(peg2_horizontal_radius, horizontal_radius)
                    ):
                        location_valid = False
                        break

                    # check pegs collision
                    if (
                        np.linalg.norm(
                            (peg2_pos[0] - peg1_pos[0], peg2_pos[1] - peg1_pos[1])
                        )
                        <= max(peg1_horizontal_radius, peg2_horizontal_radius)
                    ):
                        location_valid = False
                        break

                if location_valid:
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

        # Move objects out of the scene depending on the mode
        nut_names = {nut.name for nut in self.nuts}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(nut_names))
            for nut_type, i in self.nut_to_id.items():
                if nut_type.lower() in self.obj_to_use.lower():
                    self.nut_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.nuts[self.nut_id].name
        if self.single_object_mode in {1, 2}:
            nut_names.remove(self.obj_to_use)
            self.clear_objects(list(nut_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.nut_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active nut, else False
                    self._observables[name].set_enabled(i == self.nut_id)
                    self._observables[name].set_active(i == self.nut_id)

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """

        # load model for table top workspace
        mujoco_arena = PegsArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        return mujoco_arena

    def _load_model(self):
        """
        Override to modify xml of pegs. This is necessary because the pegs don't have free
        joints, so we must modify the xml directly before loading the model.
        """

        # skip superclass implementation
        # SingleArmEnv._load_model(self)
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()

        # define nuts
        self.nuts = []
        nut_names = ("SquareNut", "RoundNut")

        # super class should already give us placement initializer in init
        assert self.placement_initializer is not None

        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()

        for i, (nut_cls, nut_name) in enumerate(
            zip(
                (SquareNutObject, RoundNutObject),
                nut_names,
            )
        ):
            nut = nut_cls(name=nut_name)
            self.nuts.append(nut)
            # Add this nut to the placement initializer
            if isinstance(self.placement_initializer, SequentialCompositeSampler):
                # assumes we have two samplers so we add nuts to them
                self.placement_initializer.add_objects_to_sampler(
                    sampler_name=f"{nut_name}Sampler", mujoco_objects=nut
                )
            else:
                # This is assumed to be a flat sampler, so we just add all nuts to this sampler
                self.placement_initializer.add_objects(nut)

        # get xml element corresponding to both pegs
        peg1_xml = mujoco_arena.worldbody.find("./body[@name='peg1']")
        peg2_xml = mujoco_arena.worldbody.find("./body[@name='peg2']")

        # apply randomization
        peg1_xml_pos = string_to_array(peg1_xml.get("pos"))
        peg_bounds = self._get_initial_placement_bounds()["peg0"]

        sample_x = np.random.uniform(low=peg_bounds["x"][0], high=peg_bounds["x"][1])
        sample_y = np.random.uniform(low=peg_bounds["y"][0], high=peg_bounds["y"][1])
        sample_z_rot = np.random.uniform(
            low=peg_bounds["z_rot"][0], high=peg_bounds["z_rot"][1]
        )
        peg1_xml_pos[0] = peg_bounds["reference"][0] + sample_x
        peg1_xml_pos[1] = peg_bounds["reference"][1] + sample_y
        peg1_xml_quat = np.array(
            [np.cos(sample_z_rot / 2), 0, 0, np.sin(sample_z_rot / 2)]
        )
        peg2_xml_pos = string_to_array(peg1_xml.get("pos"))
        peg2_bounds = self._get_initial_placement_bounds()["peg1"]
        sample_x = np.random.uniform(low=peg2_bounds["x"][0], high=peg2_bounds["x"][1])
        sample_y = np.random.uniform(low=peg2_bounds["y"][0], high=peg2_bounds["y"][1])
        sample_z_rot = np.random.uniform(
            low=peg2_bounds["z_rot"][0], high=peg2_bounds["z_rot"][1]
        )
        peg2_xml_pos[0] = peg2_bounds["reference"][0] + sample_x
        peg2_xml_pos[1] = peg2_bounds["reference"][1] + sample_y
        peg2_xml_quat = np.array(
            [np.cos(sample_z_rot / 2), 0, 0, np.sin(sample_z_rot / 2)]
        )

        # # move peg2 completely out of scene
        # peg2_xml_pos = string_to_array(peg1_xml.get("pos"))
        # peg2_xml_pos[0] = -10.0
        # peg2_xml_pos[1] = 0.0

        def sample_peg_position(peg_bounds, reference):
            sample_x = np.random.uniform(low=peg_bounds["x"][0], high=peg_bounds["x"][1])
            sample_y = np.random.uniform(low=peg_bounds["y"][0], high=peg_bounds["y"][1])
            sample_z_rot = np.random.uniform(
                low=peg_bounds["z_rot"][0], high=peg_bounds["z_rot"][1]
            )
            pos = np.array([reference[0] + sample_x, reference[1] + sample_y, reference[2]])
            quat = np.array([
                np.cos(sample_z_rot / 2), 0, 0, np.sin(sample_z_rot / 2)
            ])
            return pos, quat

        # get collision checking entries
        peg1_size = string_to_array(peg1_xml.find("./geom").get("size"))
        peg2_size = string_to_array(peg2_xml.find("./geom").get("size"))

        self.peg1_horizontal_radius = np.linalg.norm(peg1_size[0:2], 2) * 1.8
        self.peg2_horizontal_radius = peg2_size[0] * 1.8

        # Sample positions with collision checking
        while True:
            peg1_xml_pos, peg1_xml_quat = sample_peg_position(peg_bounds, peg_bounds["reference"])
            peg2_xml_pos, peg2_xml_quat = sample_peg_position(peg2_bounds, peg2_bounds["reference"])

            distance = np.linalg.norm(peg1_xml_pos[:2] - peg2_xml_pos[:2])
            if distance > max(self.peg1_horizontal_radius, self.peg2_horizontal_radius) + 0.05:
                break  # Positions are valid

        # set modified entry in xml
        peg1_xml.set("pos", array_to_string(peg1_xml_pos))
        peg1_xml.set("quat", array_to_string(peg1_xml_quat))
        peg2_xml.set("pos", array_to_string(peg2_xml_pos))
        peg2_xml.set("quat", array_to_string(peg2_xml_quat))


        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.nuts,
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
            square_nut=dict(
                x=(-0.1, 0.1),
                y=(-0.25, 0.25),
                z_rot=(0.0, 2.0 * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
            round_nut=dict(
                x=(-0.1, 0.1),
                y=(-0.25, 0.25),
                z_rot=(0.0, 2.0 * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
            peg0=dict(
                x=(-0.15, 0.25),
                y=(-0.2, 0.2),
                z_rot=(0.0, 0.0),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
            peg1=dict(
                x=(-0.15, 0.25),
                y=(-0.2, 0.2),
                z_rot=(0.0, 0.0),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
        )

class Square_D0(NutAssemblySquare, SingleArmEnv_MG):
    """
    Augment robosuite nut assembly square task for mimicgen.
    """

    def __init__(self, **kwargs):
        assert (
            "placement_initializer" not in kwargs
        ), "this class defines its own placement initializer"

        # make placement initializer here
        nut_names = ("SquareNut", "RoundNut")

        # note: makes round nut init somewhere far off the table
        round_nut_far_init = (-1.1, -1.0)

        bounds = self._get_initial_placement_bounds()
        nut_x_ranges = (bounds["nut"]["x"], bounds["nut"]["x"])
        nut_y_ranges = (bounds["nut"]["y"], round_nut_far_init)
        nut_z_ranges = (bounds["nut"]["z_rot"], bounds["nut"]["z_rot"])
        nut_references = (bounds["nut"]["reference"], bounds["nut"]["reference"])

        placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        for nut_name, x_range, y_range, z_range, ref in zip(
            nut_names, nut_x_ranges, nut_y_ranges, nut_z_ranges, nut_references
        ):
            placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{nut_name}Sampler",
                    x_range=x_range,
                    y_range=y_range,
                    rotation=z_range,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=ref,
                    z_offset=0.02,
                )
            )

        NutAssemblySquare.__init__(
            self, placement_initializer=placement_initializer, **kwargs
        )

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

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
            nut=dict(
                x=(-0.115, -0.11),
                y=(0.11, 0.225),
                z_rot=(0.0, 2.0 * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
        )


def generate_mujoco_xml_string(peg1_xml: ET.Element) -> str:
    """Generates a MuJoCo XML string with peg1_xml as the only body in worldbody."""
    mujoco = ET.Element("mujoco", model="custom_model")
    ET.SubElement(mujoco, "compiler", angle="degree", coordinate="local")
    ET.SubElement(mujoco, "option", timestep="0.002")

    worldbody = ET.SubElement(mujoco, "worldbody")
    body = ET.SubElement(worldbody, "body", timeconst="0.01")
    body.append(peg1_xml)
    worldbody.append(peg1_xml)

    return ET.tostring(mujoco, encoding="utf-8").decode()


def generate_mujoco_xml_string(peg1_xml: ET.Element) -> str:
    """Generates a MuJoCo XML string with peg1_xml inside a nested body structure."""
    mujoco = ET.Element("mujoco", model="custom_model")

    # Add asset definitions (placeholder)
    asset = ET.SubElement(mujoco, "asset")
    ET.SubElement(asset, "mesh", file="meshes/object.msh", name="object_mesh")
    ET.SubElement(asset, "texture", type="2d", file="../textures/object.png", rgb1="1 1 1", name="tex-object")
    ET.SubElement(asset, "material", name="object_material", reflectance="0.5",
                  texrepeat="1 1", texture="tex-object", texuniform="false")

    # Add worldbody
    worldbody = ET.SubElement(mujoco, "worldbody")

    # Outer body
    outer_body = ET.SubElement(worldbody, "body")

    # Inner body containing peg1_xml
    inner_body = ET.SubElement(outer_body, "body", name="object")
    inner_body.append(peg1_xml)

    # Add sites
    ET.SubElement(outer_body, "site", rgba="0 0 0 0", size="0.005", pos="0 0 -0.10", name="bottom_site")
    ET.SubElement(outer_body, "site", rgba="0 0 0 0", size="0.005", pos="0 0 0.03", name="top_site")
    ET.SubElement(outer_body, "site", rgba="0 0 0 0", size="0.005", pos="0.04 0.03 0", name="horizontal_radius_site")

    return ET.tostring(mujoco, encoding="utf-8").decode()

class Square_D1(Square_D0):
    """
    Specifies a different placement initializer for the pegs where it is initialized
    with a broader x-range and broader y-range.
    """

    def _get_initial_placement_bounds(self):
        return dict(
            nut=dict(
                x=(-0.115, 0.115),
                y=(-0.255, 0.255),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
            peg=dict(
                x=(-0.1, 0.3),
                y=(-0.2, 0.2),
                z_rot=(0., 0.),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
        )

    def _reset_internal(self, verbose: bool = False):
        """
        Modify from superclass to keep sampling nut locations until there's no collision with either peg.
        """
        SingleArmEnv_MG._reset_internal(self)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            success = False
            for _ in range(5000): # 5000 retries

                # Sample from the placement initializer for all objects
                object_placements = self.placement_initializer.sample()

                # ADDED: check collision with pegs and maybe re-sample
                location_valid = True
                for obj_pos, obj_quat, obj in object_placements.values():
                    horizontal_radius = obj.horizontal_radius

                    peg1_id = self.sim.model.body_name2id("peg1")
                    peg1_pos = np.array(self.sim.data.body_xpos[peg1_id])
                    peg1_horizontal_radius = self.peg1_horizontal_radius
                    if (
                        np.linalg.norm((obj_pos[0] - peg1_pos[0], obj_pos[1] - peg1_pos[1]))
                        <= peg1_horizontal_radius + horizontal_radius
                    ):
                        if verbose:
                            print(f"Collision with peg1: obj_pos {obj_pos}, peg1_pos {peg1_pos}")
                        location_valid = False
                        break

                    peg2_id = self.sim.model.body_name2id("peg2")
                    peg2_pos = np.array(self.sim.data.body_xpos[peg2_id])
                    peg2_horizontal_radius = self.peg2_horizontal_radius
                    if (
                        np.linalg.norm((obj_pos[0] - peg2_pos[0], obj_pos[1] - peg2_pos[1]))
                        <= peg2_horizontal_radius + horizontal_radius
                    ):
                        if verbose:
                            print(f"Collision with peg2: {obj_pos}, {peg2_pos}")
                        location_valid = False
                        break

                if location_valid:
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Move objects out of the scene depending on the mode
        nut_names = {nut.name for nut in self.nuts}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(nut_names))
            for nut_type, i in self.nut_to_id.items():
                if nut_type.lower() in self.obj_to_use.lower():
                    self.nut_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.nuts[self.nut_id].name
        if self.single_object_mode in {1, 2}:
            nut_names.remove(self.obj_to_use)
            self.clear_objects(list(nut_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.nut_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active nut, else False
                    self._observables[name].set_enabled(i == self.nut_id)
                    self._observables[name].set_active(i == self.nut_id)

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """

        # load model for table top workspace
        mujoco_arena = PegsArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        return mujoco_arena

    def _load_model(self):
        """
        Override to modify xml of pegs. This is necessary because the pegs don't have free
        joints, so we must modify the xml directly before loading the model.
        """

        # skip superclass implementation 
        SingleArmEnv_MG._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()

        # define nuts
        self.nuts = []
        nut_names = ("SquareNut", "RoundNut")

        # super class should already give us placement initializer in init
        assert self.placement_initializer is not None

        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()

        for i, (nut_cls, nut_name) in enumerate(zip(
                (SquareNutObject, RoundNutObject),
                nut_names,
        )):
            nut = nut_cls(name=nut_name)
            self.nuts.append(nut)
            # Add this nut to the placement initializer
            if isinstance(self.placement_initializer, SequentialCompositeSampler):
                # assumes we have two samplers so we add nuts to them
                self.placement_initializer.add_objects_to_sampler(sampler_name=f"{nut_name}Sampler", mujoco_objects=nut)
            else:
                # This is assumed to be a flat sampler, so we just add all nuts to this sampler
                self.placement_initializer.add_objects(nut)

        # get xml element corresponding to both pegs
        peg1_xml = mujoco_arena.worldbody.find("./body[@name='peg1']")
        peg2_xml = mujoco_arena.worldbody.find("./body[@name='peg2']")

        # apply randomization
        peg1_xml_pos = string_to_array(peg1_xml.get("pos"))
        peg_bounds = self._get_initial_placement_bounds()["peg"]

        sample_x = np.random.uniform(low=peg_bounds["x"][0], high=peg_bounds["x"][1])
        sample_y = np.random.uniform(low=peg_bounds["y"][0], high=peg_bounds["y"][1])
        sample_z_rot = np.random.uniform(low=peg_bounds["z_rot"][0], high=peg_bounds["z_rot"][1])
        peg1_xml_pos[0] = peg_bounds["reference"][0] + sample_x
        peg1_xml_pos[1] = peg_bounds["reference"][1] + sample_y
        peg1_xml_quat = np.array([np.cos(sample_z_rot / 2), 0, 0, np.sin(sample_z_rot / 2)])

        # move peg2 completely out of scene
        peg2_xml_pos = string_to_array(peg1_xml.get("pos"))
        peg2_xml_pos[0] = -10.
        peg2_xml_pos[1] = 0.

        # set modified entry in xml
        peg1_xml.set("pos", array_to_string(peg1_xml_pos))
        peg1_xml.set("quat", array_to_string(peg1_xml_quat))
        peg2_xml.set("pos", array_to_string(peg2_xml_pos))

        # get collision checking entries
        peg1_size = string_to_array(peg1_xml.find("./geom").get("size"))
        peg2_size = string_to_array(peg2_xml.find("./geom").get("size"))
        self.peg1_horizontal_radius = np.linalg.norm(peg1_size[0:2], 2)
        self.peg2_horizontal_radius = peg2_size[0]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.nuts,
        )

    def _setup_observables(self):
        """
        Add in peg-related observables, since the peg moves now.
        For now, just try adding peg position.
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            modality = "object"
            peg1_id = self.sim.model.body_name2id("peg1")

            @sensor(modality=modality)
            def peg_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[peg1_id])

            name = "peg1_pos"
            observables[name] = Observable(
                name=name,
                sensor=peg_pos,
                sampling_rate=self.control_freq,
                enabled=True,
                active=True,
            )

        return observables


class SquareWide(Square_D1):
    """
    Even broader range for everything, and z-rotation randomization for peg.
    """

    def _get_object_scale_bounds(self):
        return dict(
            nut=((0.8, 0.8, 0.8),
                (1.4, 1.4, 1.4)),
            peg=((0.8, 0.8, 0.8),
                (1.4, 1.4, 1.4)),
        )

    def _load_model(self):
        """
        Override to modify xml of pegs. This is necessary because the pegs don't have free
        joints, so we must modify the xml directly before loading the model.
        """
        # skip superclass implementation
        NutAssemblySquare._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()

        # define nuts
        self.nuts = []
        nut_names = ("SquareNut", "RoundNut")

        # super class should already give us placement initializer in init
        assert self.placement_initializer is not None

        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()

        for i, (nut_cls, nut_name) in enumerate(
            zip(
                (SquareNutObject, RoundNutObject),
                nut_names,
            )
        ):
            nut = nut_cls(name=nut_name)
            self.nuts.append(nut)
            # Add this nut to the placement initializer
            if isinstance(self.placement_initializer, SequentialCompositeSampler):
                # assumes we have two samplers so we add nuts to them
                self.placement_initializer.add_objects_to_sampler(
                    sampler_name=f"{nut_name}Sampler", mujoco_objects=nut
                )
            else:
                # This is assumed to be a flat sampler, so we just add all nuts to this sampler
                self.placement_initializer.add_objects(nut)

        # get xml element corresponding to both pegs
        peg1_xml = mujoco_arena.worldbody.find("./body[@name='peg1']")
        peg2_xml = mujoco_arena.worldbody.find("./body[@name='peg2']")

        # apply randomization
        peg1_xml_pos = string_to_array(peg1_xml.get("pos"))
        peg_bounds = self._get_initial_placement_bounds()["peg"]

        sample_x = np.random.uniform(low=peg_bounds["x"][0], high=peg_bounds["x"][1])
        sample_y = np.random.uniform(low=peg_bounds["y"][0], high=peg_bounds["y"][1])
        sample_z_rot = np.random.uniform(
            low=peg_bounds["z_rot"][0], high=peg_bounds["z_rot"][1]
        )
        peg1_xml_pos[0] = peg_bounds["reference"][0] + sample_x
        peg1_xml_pos[1] = peg_bounds["reference"][1] + sample_y
        peg1_xml_quat = np.array(
            [np.cos(sample_z_rot / 2), 0, 0, np.sin(sample_z_rot / 2)]
        )

        # move peg2 completely out of scene
        peg2_xml_pos = string_to_array(peg1_xml.get("pos"))
        peg2_xml_pos[0] = -10.0
        peg2_xml_pos[1] = 0.0

        # set modified entry in xml
        peg1_xml.set("pos", array_to_string(peg1_xml_pos))
        peg1_xml.set("quat", array_to_string(peg1_xml_quat))
        peg2_xml.set("pos", array_to_string(peg2_xml_pos))

        # Randomly scale the peg and nut
        scale_bounds = self._get_object_scale_bounds()
        nut_scale_min, nut_scale_max = np.array(scale_bounds["nut"])
        peg_scale_min, peg_scale_max = np.array(scale_bounds["peg"])
        valid_scale = False
        for _ in range(100):
            # Sample peg scale
            peg_scale = np.random.uniform(peg_scale_min, peg_scale_max, size=3)
            peg_scale[1] = peg_scale[0]  # Keep x and y scale equal

            # Update nut scale bounds to ensure it's larger than peg x-y scale
            nut_scale_min[0] = max(nut_scale_min[0], peg_scale[0])
            nut_scale_min[1] = max(nut_scale_min[1], peg_scale[1])

            # Sample nut scale within updated bounds
            nut_scale = np.random.uniform(nut_scale_min, nut_scale_max, size=3)

            if np.all(np.max(peg_scale[:2]) <= np.min(nut_scale[:2])):
                valid_scale = True
                break

        if not valid_scale:
            raise ValueError("Could not find valid scale for nut and peg. Try resetting bounds.")

        self.nuts[0].set_scale(nut_scale)
        mujoco_arena.set_scale(peg_scale, "peg1")

        # get collision checking entries
        peg1_size = string_to_array(peg1_xml.find("./geom").get("size"))
        peg2_size = string_to_array(peg2_xml.find("./geom").get("size"))
        self.peg1_horizontal_radius = np.linalg.norm(peg1_size[0:2], 2)
        self.peg2_horizontal_radius = peg2_size[0]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.nuts,
        )


class NutAssemblySquareReal(RealEnvMixin, SquareWide):
    def __init__(self, **kwargs):
        self._override_init_qpos()
        kwargs = self._update_kwargs_for_real_robot(kwargs)
        self._setup_real_cameras()
        super().__init__(**kwargs)

    def _load_arena(self):
        arena = SquareArenaReal(
            table_full_size=self.TABLE_FULL_SIZE,
            table_friction=self.table_friction,
            table_offset=self.TABLE_OFFSET,
            has_legs=False,
        )
        self._adjust_table_alignment(arena)
        return arena

    def _load_model(self):
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(self.ROBOT_BASE_POS)
        self._update_camera_in_model()

    def _get_object_scale_bounds(self):
        return dict(
            nut=((0.8, 0.8, 0.8),
                (1.2, 1.2, 1.2)),
            peg=((0.8, 0.8, 0.8),
                (1.2, 1.2, 1.2)),
        )

    def _get_initial_placement_bounds(self):
        return dict(
            nut=dict(
                x=(-0.1, 0.1),
                y=(-0.15, 0.15),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array(self.OBJECT_OFFSET_FROM_ROBOT_BASE) + np.array(self.ROBOT_BASE_POS),
            ),
            peg=dict(
                x=(-0.1, 0.1),
                y=(-0.15, 0.15),
                z_rot=(-np.pi / 16, np.pi / 16),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array(self.OBJECT_OFFSET_FROM_ROBOT_BASE) + np.array(self.ROBOT_BASE_POS),
            ),
        )

    def _load_model(self):
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(self.ROBOT_BASE_POS)
        self._update_camera_in_model()

        # Update visual properties of square nut
        for geom_name in self.nuts[0].visual_geoms:
            geom = self.nuts[0].tree.find(f".//geom[@name='{geom_name}']")
            if geom is None:
                continue
            if geom.attrib.get("material") is not None:
                del geom.attrib["material"]
            geom.set("rgba", array_to_string([1, 1, 1, 1]))


    def _reset_internal(self, verbose: bool = False):
        """
        Modify from superclass to keep sampling nut locations until there's no collision with either peg.
        """
        SingleArmEnv_MG._reset_internal(self)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            success = False
            for _ in range(5000): # 5000 retries

                # Sample from the placement initializer for all objects
                object_placements = self.placement_initializer.sample()

                # ADDED: check collision with pegs and maybe re-sample
                location_valid = True
                for obj_pos, obj_quat, obj in object_placements.values():
                    horizontal_radius = obj.horizontal_radius * 3/4

                    peg1_id = self.sim.model.body_name2id("peg1")
                    peg1_pos = np.array(self.sim.data.body_xpos[peg1_id])
                    peg1_horizontal_radius = self.peg1_horizontal_radius * 3/4
                    if (
                        np.linalg.norm((obj_pos[0] - peg1_pos[0], obj_pos[1] - peg1_pos[1]))
                        <= peg1_horizontal_radius + horizontal_radius
                    ):
                        if verbose:
                            print(f"Collision with peg1: obj_pos {obj_pos}, peg1_pos {peg1_pos}")
                        location_valid = False
                        break

                    peg2_id = self.sim.model.body_name2id("peg2")
                    peg2_pos = np.array(self.sim.data.body_xpos[peg2_id])
                    peg2_horizontal_radius = self.peg2_horizontal_radius
                    if (
                        np.linalg.norm((obj_pos[0] - peg2_pos[0], obj_pos[1] - peg2_pos[1]))
                        <= peg2_horizontal_radius + horizontal_radius
                    ):
                        if verbose:
                            print(f"Collision with peg2: {obj_pos}, {peg2_pos}")
                        location_valid = False
                        break

                if location_valid:
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Move objects out of the scene depending on the mode
        nut_names = {nut.name for nut in self.nuts}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(nut_names))
            for nut_type, i in self.nut_to_id.items():
                if nut_type.lower() in self.obj_to_use.lower():
                    self.nut_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.nuts[self.nut_id].name
        if self.single_object_mode in {1, 2}:
            nut_names.remove(self.obj_to_use)
            self.clear_objects(list(nut_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.nut_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active nut, else False
                    self._observables[name].set_enabled(i == self.nut_id)
                    self._observables[name].set_active(i == self.nut_id)


class NutAssemblySquareRealPegCloseCam(NutAssemblySquareReal):
    def _get_initial_placement_bounds(self):
        return dict(
            nut=dict(
                x=(-0.1, 0.1),
                y=(-0.1, 0.1),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array(self.OBJECT_OFFSET_FROM_ROBOT_BASE) + np.array(self.ROBOT_BASE_POS),
            ),
            peg=dict(
                x=(0.15, 0.25),
                y=(-0.1, 0.1),
                z_rot=(-np.pi / 16, np.pi / 16),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array(self.OBJECT_OFFSET_FROM_ROBOT_BASE) + np.array(self.ROBOT_BASE_POS),
            ),
        )
