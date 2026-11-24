"""MANO forward kinematics and mesh distances for grasp evaluation."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh

from data_loaders.humanml.utils.data_utils import REPRESENTATION_IDCS
from utils.rotation_conversions import rotation_6d_to_matrix


FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
REFLECT_X = np.diag([-1, 1, 1]).astype(np.float32)


def feature(motion, name):
    start, stop = REPRESENTATION_IDCS[name]
    return motion[..., start:stop]


def rotation_matrix(values):
    matrices = rotation_6d_to_matrix(torch.as_tensor(values, dtype=torch.float32)).numpy()
    if not np.allclose(np.linalg.det(matrices), 1.0, atol=1e-4, rtol=0):
        raise ValueError('Invalid 6D rotation in the motion or object pose')
    return matrices


@dataclass
class HandMotion:
    wrist: np.ndarray
    rotation: np.ndarray
    pose: np.ndarray
    joints: np.ndarray
    vertices: np.ndarray
    predicted_distance: np.ndarray


class ManoHand:
    def __init__(self, model_dir, side, template=None):
        import smplx
        from smplx.vertex_ids import vertex_ids

        path = model_dir / f'MANO_{side.upper()}.pkl'
        if not path.is_file():
            raise FileNotFoundError(f'MANO model not found: {path}')
        kwargs = {} if template is None else {'v_template': template}
        self.model = smplx.MANO(str(path), is_rhand=side == 'right', use_pca=False,
                               flat_hand_mean=True, num_pca_comps=24, **kwargs).eval()
        self.components = np.asarray(self.model.np_hand_components, dtype=np.float32)
        self.tips = np.array([vertex_ids['mano'][finger] for finger in FINGERS])
        dominant_joint = self.model.lbs_weights.numpy().argmax(axis=-1)
        parents = self.model.parents.numpy()
        self.finger_vertices = []
        for tip in self.tips:
            joint = dominant_joint[tip]
            chain = []
            while joint > 0:
                chain.append(joint)
                joint = parents[joint]
            self.finger_vertices.append(np.flatnonzero(np.isin(dominant_joint, chain)))

    @torch.inference_mode()
    def fk(self, wrist, rotation, pose, predicted_distance):
        orient = Rotation.from_matrix(rotation).as_rotvec().astype(np.float32)
        count = len(wrist)
        result = self.model(global_orient=torch.from_numpy(orient),
                            hand_pose=torch.from_numpy(pose),
                            transl=torch.zeros(count, 3),
                            betas=torch.zeros(count, self.model.num_betas))
        offset = wrist[:, None] - result.joints[:, :1].numpy()
        vertices = result.vertices.numpy() + offset
        joints = np.concatenate((result.joints.numpy() + offset, vertices[:, self.tips]), axis=1)
        return HandMotion(wrist, rotation, pose, joints, vertices, predicted_distance)


def decode_hand(motion, hands, side, mirrored=False):
    """Decode one hand, undoing the X reflection and swapping sides if needed."""
    source = {'left': 'right', 'right': 'left'}[side] if mirrored else side
    wrist = feature(motion, f'pos_{source}')
    rotation = rotation_matrix(feature(motion, f'global_orient_{source[0]}'))
    pose = feature(motion, f'pca_{source}') @ hands[source].components
    if mirrored:
        wrist = wrist @ REFLECT_X
        rotation = REFLECT_X @ rotation @ REFLECT_X
        # Axis-angle vectors are axial vectors. Reflect each local joint rotation.
        pose = (pose.reshape(-1, 15, 3) @ -REFLECT_X).reshape(-1, 45)
    return hands[side].fk(wrist, rotation, pose, feature(motion, f'sdf_{source}'))


def object_coordinates(points, object_pose):
    translation, rotation = object_pose
    local = points.reshape(len(points), -1, 3)
    return ((local - translation[:, None]) @ rotation).reshape(points.shape)


class MeshDistance:
    """Exact triangle distances with positive exterior and negative interior."""

    def __init__(self, mesh, threads=0):
        try:
            import open3d as o3d
        except ImportError as error:
            raise RuntimeError('Grasp evaluation requires Open3D. Install it with python -m pip install open3d.') from error
        if not mesh.is_watertight:
            mesh = mesh.copy()
            mesh.merge_vertices(merge_tex=True, merge_norm=True)
        if not mesh.is_watertight:
            raise ValueError('The object mesh must be watertight to evaluate penetration')
        self.bounds = mesh.bounds
        self.threads = threads
        self.o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=threads)
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
                                 o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))

    def signed(self, points):
        t = self.o3d.core.Tensor(np.ascontiguousarray(points, dtype=np.float32))
        return self.scene.compute_signed_distance(t, nthreads=self.threads, nsamples=3).numpy()

    def minimum(self, points):
        """Exact minimum signed distance using the box to discard distant queries."""
        flat = points.reshape(-1, 3)
        first, last = self.bounds
        excess = np.maximum(np.maximum(first - flat, flat - last), 0)
        lower = np.linalg.norm(excess, axis=-1)
        index = np.argmin(lower)
        incumbent = self.signed(flat[index:index + 1])[0]
        near = lower <= max(0, incumbent) + 1e-6
        return float(min(incumbent, self.signed(flat[near]).min()))


def load_mesh(path):
    mesh = trimesh.load(path, force='mesh', process=False)
    if (not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces)
            or not np.isfinite(mesh.vertices).all()):
        raise ValueError(f'{path}: expected a nonempty triangle mesh with finite vertices')
    return mesh
