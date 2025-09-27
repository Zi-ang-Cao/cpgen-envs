import numpy as np

from robosuite.utils.mjcf_utils import array_to_string, string_to_array, xml_path_completion
from scipy.spatial.transform import Rotation as R
from robosuite.utils.robot_composition_utils import create_composite_robot

import numpy as np


from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

def convert_opencv_to_opengl(rot_mat_opencv: np.ndarray) -> np.ndarray:
    rot_mat_opengl = rot_mat_opencv.copy()
    rot_mat_opengl[:3, 1] *= -1
    rot_mat_opengl[:3, 2] *= -1
    return rot_mat_opengl


class RealEnvMixin:
    AGENTVIEW_CAM_POS_IN_ROBOT_FRAME = np.array([1.15, -0.042, 0.55])
    AGENTVIEW_EULER_XYZ = np.array([-135, 0, 90])
    HAND_CAM_POS = np.array([0.064, 0.0325, 0.05])
    HAND_CAM_EULER_XYZ = np.array([-180, 0, 90])
    TABLE_FULL_SIZE = np.array([1.5, 1.4, 0.05])
    TABLE_OFFSET = np.array([0.6, 0, 0.82])
    TABLE_LEFT_ALIGN_WIDTH = 0.884
    ROBOT_BASE_POS = np.array([0, 0, 0.82])
    OBJECT_OFFSET_FROM_ROBOT_BASE = np.array([0.45, 0, 0])

    def _override_init_qpos(self):
        from robosuite.models.robots.manipulators import Panda
        def new_panda_init_qpos(_):
            return np.array([0.09162, -0.198264, -0.0199, -2.473226, -0.01307, 2.3039658, 0.8480939])
        Panda.init_qpos = property(new_panda_init_qpos)

    def _update_kwargs_for_real_robot(self, kwargs: dict) -> dict:
        if "robots" in kwargs:
            full_name = kwargs["robots"] if isinstance(kwargs["robots"], str) else kwargs["robots"][0]
            parts = full_name.split("_")
            if len(parts) == 2:
                create_composite_robot(full_name, robot=parts[0], grippers=parts[1])
        kwargs["base_types"] = "NullMount"
        kwargs["table_full_size"] = self.TABLE_FULL_SIZE
        return kwargs

    def _setup_real_cameras(self):
        self.agentview_camera_rot_mat_opencv = R.from_euler("xyz", self.AGENTVIEW_EULER_XYZ, degrees=True).as_matrix()
        self.agentview_camera_rot_mat_opengl = convert_opencv_to_opengl(self.agentview_camera_rot_mat_opencv)
        quat = R.from_matrix(self.agentview_camera_rot_mat_opengl).as_quat()
        agentview_quat = quat[[3, 0, 1, 2]]

        hand_quat = R.from_euler("XYZ", self.HAND_CAM_EULER_XYZ, degrees=True).as_quat()[[3, 0, 1, 2]]

        self.camera_info = {
            "agentview": {
                "pos": np.array(self.AGENTVIEW_CAM_POS_IN_ROBOT_FRAME) + np.array(self.ROBOT_BASE_POS),
                "quat": agentview_quat,
                "camera_attribs": {"fovy": "42.7"},
            },
            "robot0_eye_in_hand": {
                "pos": np.array(self.HAND_CAM_POS),
                "quat": hand_quat,
                "camera_attribs": {"fovy": "42.7"},
            },
        }

    def _adjust_table_alignment(self, arena):
        from robosuite.utils.mjcf_utils import array_to_string, string_to_array
        curr_table_pos = string_to_array(arena.table_visual.get("pos"))
        curr_table_pos[1] += (self.TABLE_FULL_SIZE[1] - self.TABLE_LEFT_ALIGN_WIDTH) / 2
        arena.table_visual.set("pos", array_to_string(curr_table_pos))
        arena.bottom_pos = np.array([0, 0, -arena.table_half_size[2]])
        arena.floor.set("pos", array_to_string(arena.bottom_pos))
        arena.set_origin([0, 0, 0])

    def _update_camera_in_model(self):
        for name, info in self.camera_info.items():
            if "robot0" in name:
                cam_elem = self.robots[0].robot_model.worldbody.find(f".//camera[@name='{name}']")
                if cam_elem is None:
                    continue
                cam_elem.set("pos", array_to_string(info['pos']))
                cam_elem.set("quat", array_to_string(info['quat']))
                for key, val in info.get("camera_attribs", {}).items():
                    cam_elem.set(key, val)
            else:
                self.model.mujoco_arena.set_camera(
                    camera_name=name,
                    pos=info['pos'],
                    quat=info['quat'],
                    camera_attribs=info.get('camera_attribs', {}),
                )
