"""Retarget human fingertips to the real RH56F1 hand on the xArm flange.

Solves for a flange pose in the object frame and the six actuated joints, using the vendor
RH56F1 URDF rather than the OmniDex Inspire model. Coupled joints use the multipliers measured
in demofungrasp_policy/RH56F1_HAND_RESULT.md. Object contact is approximated by a radial
profile around the object z axis, so this only suits roughly axisymmetric objects.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

DEFAULT_URDF = Path(__file__).resolve().parents[1] / "third_party/robot_models/rh56f1/RH56F1_R.urdf"
ACTUATED = ("right_thumb_1_joint", "right_thumb_2_joint", "right_index_1_joint",
            "right_middle_1_joint", "right_ring_1_joint", "right_little_1_joint")
UPPER = np.array([2.091, 0.440, 1.30, 1.476, 1.476, 1.476])  # measured travel, RH56F1_HAND_RESULT.md
COUPLED = {"right_thumb_3_joint": ("right_thumb_2_joint", 1.1425),
           "right_thumb_4_joint": ("right_thumb_2_joint", 0.858),
           **{f"right_{f}_2_joint": (f"right_{f}_1_joint", 1.1169) for f in ("index", "middle", "ring", "little")}}
TIPS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip")
CHAINS = (("right_thumb_1", "right_thumb_2", "right_thumb_3", "right_thumb_4", "thumb_tip"),
          *((f"right_{f}_1", f"right_{f}_2", f"{f}_tip") for f in ("index", "middle", "ring", "little")))
ADAPTER_Z = 0.0235  # link7 to hand base_link, no rotation (RH56F1_URDF_HANDOFF.md)
PALM = np.array([[x, y, z] for x in (-0.02, 0.021) for y in (-0.04, 0.04) for z in (0.06, 0.10, 0.14, 0.165)])
RAW_PER_RAD = 1800 / np.pi


class HandModel:
    def __init__(self, urdf: Path = DEFAULT_URDF):
        root = ET.parse(urdf).getroot()
        self.joints = {}
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler("xyz", list(map(float, origin.get("rpy", "0 0 0").split()))).as_matrix()
            transform[:3, 3] = list(map(float, origin.get("xyz", "0 0 0").split()))
            axis = joint.find("axis")
            self.joints[joint.find("child").get("link")] = (
                joint.get("name"), joint.find("parent").get("link"), joint.get("type"), transform,
                None if axis is None else np.array(list(map(float, axis.get("xyz").split()))))

    def link(self, name: str, q: np.ndarray) -> np.ndarray:
        """Link origin in the flange frame for actuated joints q, ordered as ACTUATED."""
        values = dict(zip(ACTUATED, q))
        chain = []
        while name in self.joints:
            chain.append(self.joints[name])
            name = self.joints[name][1]
        transform = np.eye(4)
        transform[2, 3] = ADAPTER_Z
        for joint_name, _, joint_type, origin, axis in reversed(chain):
            transform = transform @ origin
            if joint_type == "revolute":
                angle = values.get(joint_name)
                if angle is None and joint_name in COUPLED:
                    source, ratio = COUPLED[joint_name]
                    angle = values[source] * ratio
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(axis * (angle or 0.0)).as_matrix()
                transform = transform @ motion
        return transform[:3, 3]

    def tips(self, q: np.ndarray) -> np.ndarray:
        return np.array([self.link(name, q) for name in TIPS])

    def surface_points(self, q: np.ndarray) -> np.ndarray:
        points = []
        for chain in CHAINS:
            nodes = [self.link(name, q) for name in chain]
            for a, b in zip(nodes[:-1], nodes[1:]):
                points += [a + t * (b - a) for t in (0.0, 0.5, 1.0)]
        return np.vstack([np.array(points), PALM])


def radial_profile(vertices: np.ndarray, bins: int = 40):
    """Largest radius around the z axis per height bin."""
    edges = np.linspace(vertices[:, 2].min(), vertices[:, 2].max(), bins + 1)
    radius = np.linalg.norm(vertices[:, :2], axis=1)
    index = np.clip(np.digitize(vertices[:, 2], edges) - 1, 0, bins - 1)
    return edges, np.array([radius[index == i].max() if np.any(index == i) else 0.0 for i in range(bins)])


def penetration(points: np.ndarray, profile) -> np.ndarray:
    edges, radius = profile
    index = np.digitize(points[:, 2], edges) - 1
    inside = (index >= 0) & (index < len(radius))
    depth = np.zeros(len(points))
    depth[inside] = np.maximum(radius[index[inside]] - np.linalg.norm(points[inside, :2], axis=1), 0.0)
    return depth


def to_transform(pose: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


@dataclass
class Result:
    T_object_flange: np.ndarray
    joints: np.ndarray
    tips: np.ndarray
    tip_error: np.ndarray
    max_penetration: float


PALM_FINGERS = ("index", "middle", "ring", "little")
# Palm prior weights per metre along the human palm axes (x palm normal, y lateral, z finger
# direction) and per radian of rotation about those axes. The palm normal offset is the
# loosest: the robot hand is thicker and its fingers have different lengths, so it may stand
# off further. Rotation about the lateral axis (wrist flexion, i.e. finger elevation) is weak
# because it mostly reflects the human forearm posture, which the arm may not be able to copy.
PALM_POSITION_WEIGHTS = np.array([0.6, 1.5, 1.0])
PALM_ROTATION_WEIGHTS = np.array([0.12, 0.03, 0.12])


def palm_in_flange(model: HandModel) -> np.ndarray:
    """Palm frame of the open hand in the flange frame, defined like grasp_interface.palm_frame.

    Origin at the mean of the four finger base joints, +z along the flange axis (the finger
    direction of the open hand), +y from the little to the index finger base.
    """
    from sample.grasp_interface import frame_from_axes

    q = np.zeros(len(ACTUATED))
    bases = {name: model.link(f"right_{name}_1", q) for name in PALM_FINGERS}
    return frame_from_axes(np.mean(list(bases.values()), axis=0), bases["index"] - bases["little"],
                           np.array([0.0, 0.0, 1.0]))


def palm_prior_residual(flange: np.ndarray, flange_palm: np.ndarray, human_palm: np.ndarray,
                        scale: float = 1.0) -> np.ndarray:
    """Weighted robot-to-human palm deviation in the human palm frame (6 terms)."""
    palm = flange @ flange_palm
    offset = human_palm[:3, :3].T @ (palm[:3, 3] - human_palm[:3, 3])
    rotation = Rotation.from_matrix(human_palm[:3, :3].T @ palm[:3, :3]).as_rotvec()
    return scale * np.r_[PALM_POSITION_WEIGHTS * offset, PALM_ROTATION_WEIGHTS * rotation]


def retarget(model: HandModel, human_tips: np.ndarray, initial_flange: np.ndarray, initial_joints: np.ndarray,
             profile=None, tip_weights=(2.0, 1.0, 1.0, 1.0, 1.0), penetration_weight: float = 5.0,
             pose_weight: float = 0.02, palm_prior=None, palm_prior_scale: float = 1.0) -> Result:
    """human_tips are ordered thumb to little, in the object frame, in metres.

    ``palm_prior`` is an optional (T_object_palm_human, T_flange_palm_robot) pair that keeps the
    robot palm near the human palm while the fingertips are matched.
    """
    human_tips = np.asarray(human_tips, dtype=float)
    weights = np.asarray(tip_weights, dtype=float)[:, None]
    start = np.concatenate([initial_flange[:3, 3], Rotation.from_matrix(initial_flange[:3, :3]).as_rotvec(),
                            np.clip(initial_joints, 0.0, UPPER)])

    def world(transform, points):
        return points @ transform[:3, :3].T + transform[:3, 3]

    def residual(x):
        transform, q = to_transform(x[:6]), x[6:]
        terms = [(weights * (world(transform, model.tips(q)) - human_tips)).ravel(), pose_weight * (x[6:] - start[6:])]
        if profile is not None:
            terms.append(penetration_weight * penetration(world(transform, model.surface_points(q)), profile))
        if palm_prior is not None:
            human_palm, flange_palm = palm_prior
            terms.append(palm_prior_residual(transform, flange_palm, human_palm, palm_prior_scale))
        return np.concatenate(terms)

    lower = np.concatenate([np.full(6, -np.inf), np.zeros(6)])
    upper = np.concatenate([np.full(6, np.inf), UPPER])
    solution = least_squares(residual, np.clip(start, lower + 1e-6, upper - 1e-6), bounds=(lower, upper), max_nfev=200, diff_step=1e-4).x
    transform, q = to_transform(solution[:6]), solution[6:]
    tips = world(transform, model.tips(q))
    depth = 0.0 if profile is None else float(penetration(world(transform, model.surface_points(q)), profile).max())
    return Result(transform, q, tips, np.linalg.norm(tips - human_tips, axis=1), depth)


def preshape_joints(q_open: np.ndarray, amount) -> np.ndarray:
    """Open the flexion joints of ``q_open`` toward zero by ``amount`` in [0, 1].

    Thumb rotation (index 0) is kept so the thumb stays in the same opposition plane while the
    fingers open; zero flexion is the fully open hand.
    """
    amount = np.clip(np.asarray(amount, dtype=float), 0.0, 1.0)
    q = np.array(q_open, dtype=float) * np.ones(np.shape(amount) + (len(ACTUATED),))
    q[..., 1:] *= (1.0 - amount)[..., None]
    return q


def joints_to_raw(q: np.ndarray) -> np.ndarray:
    """Order out: little, ring, middle, index, thumb_bend, thumb_rot."""
    thumb_rot, thumb_bend, index, middle, ring, little = q
    return np.rint([1742 - little * RAW_PER_RAD, 1742 - ring * RAW_PER_RAD, 1742 - middle * RAW_PER_RAD,
                    1742 - index * RAW_PER_RAD, 1350 - thumb_bend * RAW_PER_RAD, 1800 - thumb_rot * RAW_PER_RAD]).astype(int)
