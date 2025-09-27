import numpy as np
import pathlib

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import array_to_string, string_to_array, xml_path_completion

import cpgen_envs


class TableArena(Arena):
    """
    Workspace that contains an empty table.


    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
        has_legs (bool): whether the table has legs or not
        xml (str): xml file to load arena
    """

    def __init__(
        self,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.8),
        has_legs=True,
        xml="arenas/table_arena.xml",
    ):
        super().__init__(xml)  # Removed xml_path_completion

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction
        self.table_offset = table_offset
        self.center_pos = self.bottom_pos + np.array([0, 0, -self.table_half_size[2]]) + self.table_offset

        self.table_body = self.worldbody.find("./body[@name='table']")
        self.table_collision = self.table_body.find("./geom[@name='table_collision']")
        self.table_visual = self.table_body.find("./geom[@name='table_visual']")
        self.table_top = self.table_body.find("./site[@name='table_top']")

        self.has_legs = has_legs
        self.table_legs_visual = [
            self.table_body.find("./geom[@name='table_leg1_visual']"),
            self.table_body.find("./geom[@name='table_leg2_visual']"),
            self.table_body.find("./geom[@name='table_leg3_visual']"),
            self.table_body.find("./geom[@name='table_leg4_visual']"),
        ]

        self.configure_location()

    def configure_location(self):
        """Configures correct locations for this arena"""
        self.floor.set("pos", array_to_string(self.bottom_pos))

        self.table_body.set("pos", array_to_string(self.center_pos))
        self.table_collision.set("size", array_to_string(self.table_half_size))
        self.table_collision.set("friction", array_to_string(self.table_friction))
        self.table_visual.set("size", array_to_string(self.table_half_size))

        self.table_top.set("pos", array_to_string(np.array([0, 0, self.table_half_size[2]])))

        # If we're not using legs, set their size to 0
        if not self.has_legs:
            for leg in self.table_legs_visual:
                leg.set("rgba", array_to_string([1, 0, 0, 0]))
                leg.set("size", array_to_string([0.0001, 0.0001]))
        else:
            # Otherwise, set leg locations appropriately
            delta_x = [0.1, -0.1, -0.1, 0.1]
            delta_y = [0.1, 0.1, -0.1, -0.1]
            for leg, dx, dy in zip(self.table_legs_visual, delta_x, delta_y):
                # If x-length of table is less than a certain length, place leg in the middle between ends
                # Otherwise we place it near the edge
                x = 0
                if self.table_half_size[0] > abs(dx * 2.0):
                    x += np.sign(dx) * self.table_half_size[0] - dx
                # Repeat the same process for y
                y = 0
                if self.table_half_size[1] > abs(dy * 2.0):
                    y += np.sign(dy) * self.table_half_size[1] - dy
                # Get z value
                z = (self.table_offset[2] - self.table_half_size[2]) / 2.0
                # Set leg position
                leg.set("pos", array_to_string([x, y, -z]))
                # Set leg size
                leg.set("size", array_to_string([0.025, z]))

    @property
    def table_top_abs(self):
        """
        Grabs the absolute position of table top

        Returns:
            np.array: (x,y,z) table position
        """
        return string_to_array(self.floor.get("pos")) + self.table_offset

class TableArenaReal(TableArena):
    def __init__(
        self,
        table_full_size=(0.45, 0.69, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0),
        has_legs=False,
        xml=xml_path_completion("arenas/table_arena.xml"),
    ):
        super().__init__(
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=table_offset,
            has_legs=has_legs,
            xml=xml,
        )

        # update colors to match real world objects
        if self.table_visual.attrib.get("material") is not None:
            del self.table_visual.attrib["material"]
        self.table_visual.set("rgba", array_to_string([0, 0, 0, 1]))


class SquareArenaReal(TableArena):
    """
    Workspace that contains a tabletop with two fixed pegs.

    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
    """

    def __init__(
        self,
        table_full_size=(0.45, 0.69, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0),
        has_legs=False,
    ):
        super().__init__(
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=table_offset,
            xml=str(pathlib.Path(cpgen_envs.__file__).parent / "models/assets/arenas/square_arena_real.xml"),
            has_legs=has_legs,
        )

        self.peg1_base_visual = self.worldbody.find("./body[@name='peg1']/geom[@name='peg1_base_visual']")
        self.peg1_stick_visual = self.worldbody.find("./body[@name='peg1']/geom[@name='peg1_stick_visual']")
        # Get references to peg bodies
        self.peg1_body = self.worldbody.find("./body[@name='peg1']")
        self.peg2_body = self.worldbody.find("./body[@name='peg2']")

        # update colors to match real world objects
        if self.table_visual.attrib.get("material") is not None:
            del self.table_visual.attrib["material"]
        self.table_visual.set("rgba", array_to_string([0, 0, 0, 1]))

        if self.peg1_base_visual.attrib.get("material") is not None:
            del self.peg1_base_visual.attrib["material"]
        self.peg1_base_visual.set("rgba", array_to_string([1, 1, 1, 1]))

        if self.peg1_stick_visual.attrib.get("material") is not None:
            del self.peg1_stick_visual.attrib["material"]
        self.peg1_stick_visual.set("rgba", array_to_string([1, 1, 1, 1]))
