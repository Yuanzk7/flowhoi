"""Robot-agnostic grasp description extracted from a MANO grasp trajectory.

The description is what a robot retargeting step consumes instead of raw MANO joints:

* ``T_object_palm``: the final palm frame in the object frame. Origin at the mean of the
  four finger MCP joints; +z from the wrist to the middle MCP (finger direction), +y from
  the pinky MCP to the index MCP, +x = y x z (palm normal, palmar side for a right hand).
* ``contact_fingers``: digits within the contact radius at the final frame.
* ``approach_dir_object``: unit direction of palm travel over the last frames.
* ``remaining_m`` / ``preshape``: remaining palm path length and normalized grip aperture
  for every frame, so a robot can replay the preshape against its own distance to the goal.

A robot hand defines the same palm frame from its own finger base joints (see
``rh56f1_retarget.palm_in_flange``), so aligning the two frames gives the initial wrist pose.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP, RING_MCP = 0, 1, 4, 7, 10
THUMB_TIP, INDEX_TIP, MIDDLE_TIP = 16, 17, 18
APPROACH_WINDOW = 15
MIN_TRAVEL_M = .003
MIN_APERTURE_RANGE_M = .005
FALLBACK_CONTACT_RADIUS_M = .01


def frame_from_axes(origin, lateral, forward):
    """Right-handed frame with +z along ``forward`` and +y along ``lateral`` made orthogonal."""
    z = np.asarray(forward, dtype=float)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.asarray(lateral, dtype=float)
    y = y - np.sum(y * z, axis=-1, keepdims=True) * z
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    pose = np.zeros(z.shape[:-1] + (4, 4))
    pose[..., :3, :3] = np.stack((np.cross(y, z), y, z), axis=-1)
    pose[..., :3, 3] = origin
    pose[..., 3, 3] = 1
    if not np.isfinite(pose).all():
        raise ValueError('Degenerate palm frame')
    return pose


def palm_frame(joints):
    """Palm frame from MANO joints shaped [..., 21, 3] (smplx order with fingertips appended)."""
    joints = np.asarray(joints, dtype=float)
    origin = joints[..., [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP], :].mean(axis=-2)
    return frame_from_axes(origin, joints[..., INDEX_MCP, :] - joints[..., PINKY_MCP, :],
                           joints[..., MIDDLE_MCP, :] - joints[..., WRIST, :])


def palm_deviation(reference, pose):
    """Offset of ``pose`` expressed in the ``reference`` frame axes, and the rotation angle in degrees."""
    offset = reference[:3, :3].T @ (pose[:3, 3] - reference[:3, 3])
    angle = np.rad2deg(Rotation.from_matrix(reference[:3, :3].T @ pose[:3, :3]).magnitude())
    return offset, float(angle)


def approach_direction(origins, window=APPROACH_WINDOW):
    """Unit travel direction of the palm into the grasp, or None if the palm barely moves."""
    for start in (max(0, len(origins) - 1 - window), 0):
        travel = origins[-1] - origins[start]
        length = np.linalg.norm(travel)
        if length >= MIN_TRAVEL_M:
            return travel / length
    return None


def preshape_profile(origins, joints):
    """Remaining palm path length and normalized aperture (1 = widest, 0 = final grasp)."""
    steps = np.linalg.norm(np.diff(origins, axis=0), axis=1)
    remaining = np.r_[np.cumsum(steps[::-1])[::-1], 0.]
    aperture = np.linalg.norm(joints[:, THUMB_TIP] - joints[:, [INDEX_TIP, MIDDLE_TIP]].mean(axis=1), axis=1)
    span = aperture.max() - aperture[-1]
    if span < MIN_APERTURE_RANGE_M:
        preshape = np.zeros(len(aperture))
    else:
        preshape = np.clip((aperture - aperture[-1]) / span, 0., 1.)
    return remaining, aperture, preshape


def preshape_at(distance, remaining, preshape):
    """Preshape for a remaining distance; beyond the recorded path it holds the first frame's value."""
    return np.interp(distance, remaining[::-1], preshape[::-1])


def contact_set(finger_distance, radius):
    return tuple(name for name, value in zip(FINGERS, finger_distance) if value <= radius)


def extract_spec(joints, finger_distance=None, contact_radius=None, joint_distance=None):
    """Build the description from object-frame MANO joints shaped [T, 21, 3].

    ``finger_distance`` (5,) is the closest surface distance of each digit at the final
    frame. Without it, fingertip joint distances (``joint_distance`` [21]) are used with a
    looser radius.
    """
    joints = np.asarray(joints, dtype=float)
    if joints.ndim != 3 or joints.shape[1:] != (21, 3) or len(joints) < 2 or not np.isfinite(joints).all():
        raise ValueError('Expected finite MANO joints shaped [T >= 2, 21, 3]')
    palms = palm_frame(joints)
    origins = palms[:, :3, 3]
    direction = approach_direction(origins)
    remaining, aperture, preshape = preshape_profile(origins, joints)
    if finger_distance is not None and contact_radius is not None:
        fingers, radius, source = contact_set(finger_distance, contact_radius), contact_radius, 'finger surface'
    elif joint_distance is not None:
        tips = np.abs(np.asarray(joint_distance)[THUMB_TIP:THUMB_TIP + 5])
        fingers, radius, source = contact_set(tips, FALLBACK_CONTACT_RADIUS_M), FALLBACK_CONTACT_RADIUS_M, 'fingertip joint'
        finger_distance = tips
    else:
        fingers, radius, source, finger_distance = (), np.nan, 'unavailable', np.full(5, np.nan)
    final = palms[-1]
    return dict(
        T_object_palm=final, palm_trajectory=palms,
        contact_fingers=np.array(fingers, dtype='<U6'), finger_distance_m=np.asarray(finger_distance, dtype=float),
        contact_radius_m=float(radius), contact_source=source,
        fingertips_object_m=joints[-1, THUMB_TIP:THUMB_TIP + 5],
        approach_valid=direction is not None,
        approach_dir_object=np.zeros(3) if direction is None else direction,
        approach_dir_palm=np.zeros(3) if direction is None else final[:3, :3].T @ direction,
        remaining_m=remaining, aperture_m=aperture, preshape=preshape,
    )


def object_frame_joints(joints, object_pose):
    """Express scene-frame joints [T, 21, 3] in the object frame given translation + 6D rotation [T, 9]."""
    from sample.grasp_geometry import rotation_matrix

    rotation = rotation_matrix(object_pose[:, 3:])
    return np.einsum('tij,tki->tkj', rotation, joints - object_pose[:, None, :3])


def spec_from_grasp_archive(path, frame=None):
    """Description from a select_grasp ``right_hand_grasp.npz``, truncated at ``frame``."""
    with np.load(path, allow_pickle=False) as data:
        if tuple(data['fingertip_order']) != FINGERS:
            raise ValueError(f'{path}: unexpected fingertip order')
        frames = data['frame_index']
        stop = len(frames) if frame is None else int(np.flatnonzero(frames == frame)[0]) + 1
        joints = object_frame_joints(data['joints'][:stop], data['object_pose'][:stop])
        finger = data['finger_distance_m'] if 'finger_distance_m' in data else None
        radius = float(data['contact_radius_m']) if 'contact_radius_m' in data else None
        joint_distance = data['joint_signed_distance'][stop - 1]
    return extract_spec(joints, finger, radius, joint_distance)


def save_spec(path, spec, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **spec, **extra)


def load_spec(path):
    with np.load(path, allow_pickle=False) as data:
        spec = {key: data[key] for key in data.files}
    spec['contact_fingers'] = tuple(str(name) for name in spec['contact_fingers'])
    spec['approach_valid'] = bool(spec['approach_valid'])
    return spec
