"""Prepare a metric object mesh and the BPS used by FlowHOI."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch
import trimesh


def load_object_basis(path):
    """Read the published 1024 object basis points in their original order."""
    data = torch.load(path, map_location='cpu', weights_only=True)
    basis = data.get('obj') if isinstance(data, dict) else None
    if not isinstance(basis, torch.Tensor) or basis.shape != (1, 1024, 3):
        raise ValueError(f'{path}: expected an obj tensor shaped (1, 1024, 3)')
    points = basis[0].float().numpy()
    if not np.isfinite(points).all():
        raise ValueError(f'{path}: object basis has nonfinite coordinates')
    return points


def encode_object_bps(vertices, basis):
    """Return nearest vertex minus basis point, as float32 [1, 1024, 3]."""
    vertices = np.asarray(vertices, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)
    _, indices = cKDTree(vertices).query(basis)
    return (vertices[indices] - basis)[None]


def prepare_mesh(mesh, *, meters_per_unit, longest_extent_m, rotation_deg, center, mirror_x):
    """Rotate into the object frame, set scale and origin, and optionally reflect X."""
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError('Expected a nonempty triangle mesh')
    vertices = np.asarray(mesh.vertices)
    if not np.isfinite(vertices).all():
        raise ValueError('Mesh has nonfinite vertices')

    rotation = Rotation.from_euler('xyz', rotation_deg, degrees=True).as_matrix()
    rotated = vertices @ rotation.T
    low, high = rotated.min(axis=0), rotated.max(axis=0)
    extent = (high - low).max()
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError('Mesh needs a finite, positive extent')

    scale = meters_per_unit if longest_extent_m is None else longest_extent_m / extent
    origin = (low + high) / 2 if center == 'bbox' else np.zeros(3)
    reflection = np.diag([-1, 1, 1]) if mirror_x else np.eye(3)
    transform = np.eye(4)
    transform[:3, :3] = scale * (reflection @ rotation)
    transform[:3, 3] = -scale * (reflection @ origin)

    # PLY stores float32 vertices. Encode the coordinates that will be saved.
    with np.errstate(over='ignore', under='ignore'):
        points = (vertices @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)
    if not np.isfinite(points).all() or not np.any(np.ptp(points.astype(np.float64), axis=0) > 0):
        raise ValueError('Prepared mesh coordinates are invalid at float32 precision')

    prepared = mesh.copy()
    prepared.vertices = points
    if mirror_x:
        prepared.faces = np.ascontiguousarray(prepared.faces[:, ::-1])
    return prepared, transform, scale


def positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('expected a finite, positive number')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', required=True, type=Path, help='Input triangle mesh (OBJ, PLY, GLB, etc.)')
    parser.add_argument('--basis', type=Path, default=Path('dataset/GRAB_HANDS/bps.pt'),
                        help='FlowHOI basis checkpoint (default: dataset/GRAB_HANDS/bps.pt)')
    parser.add_argument('--output-dir', required=True, type=Path, help='Directory for the mesh, BPS and transform')
    scale_options = parser.add_mutually_exclusive_group(required=True)
    scale_options.add_argument('--meters-per-unit', type=positive_float, help='Use 1 for a mesh already in metres')
    scale_options.add_argument('--longest-extent-m', type=positive_float,
                               help='Set the longest axis-aligned extent after rotation to this size in metres')
    parser.add_argument('--rotate-xyz-deg', type=float, nargs=3, default=(0, 0, 0), metavar=('RX', 'RY', 'RZ'),
                        help='Extrinsic X, Y, Z rotations in degrees to place the object upright (+Z)')
    parser.add_argument('--center', choices=('bbox', 'keep'), default='bbox',
                        help='Move the bounding box center to the origin, or keep the input origin')
    parser.add_argument('--mirror-x', action='store_true', help='Reflect X in the final object frame as in GRAB')
    args = parser.parse_args()
    if not np.isfinite(args.rotate_xyz_deg).all():
        parser.error('--rotate-xyz-deg must contain finite values')

    output = args.output_dir.absolute()
    try:
        basis = load_object_basis(args.basis)
        mesh = trimesh.load(args.mesh, force='mesh', process=False)
        prepared, transform, scale = prepare_mesh(
            mesh, meters_per_unit=args.meters_per_unit, longest_extent_m=args.longest_extent_m,
            rotation_deg=args.rotate_xyz_deg, center=args.center, mirror_x=args.mirror_x)
        bps = encode_object_bps(prepared.vertices, basis)
        if not np.isfinite(bps).all():
            raise ValueError('BPS has nonfinite values')

        metadata = {
            'source_mesh': str(args.mesh.resolve()),
            'basis': str(args.basis.resolve()),
            'input_to_object': transform.tolist(),
            'meters_per_unit': scale,
            'rotate_xyz_deg': list(args.rotate_xyz_deg),
            'center': args.center,
            'mirror_x': args.mirror_x,
            'extents_m': prepared.extents.tolist(),
        }

        output.parent.mkdir(parents=True, exist_ok=True)
        # Publish the mesh, transform and BPS together. Rename fails for an
        # existing nonempty directory, so a previous export stays intact.
        with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
            bundle = Path(temporary) / 'result'
            bundle.mkdir()
            prepared.export(bundle / 'object_mesh.ply')
            np.save(bundle / 'object_bps.npy', bps, allow_pickle=False)
            (bundle / 'mesh_transform.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
            bundle.rename(output)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    print(f'Saved {bps.shape} {bps.dtype} BPS and mesh to {output}')
    print(f'Object XYZ extents in metres: {prepared.extents.round(5)}')


if __name__ == '__main__':
    main()
