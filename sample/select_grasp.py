"""Select a right-hand Stage 1 trajectory using MANO and physical mesh distances."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation

from sample.grasp_geometry import (FINGERS, REFLECT_X, ManoHand, MeshDistance, decode_hand,
                                   feature, load_mesh, object_coordinates, rotation_matrix)
from sample.grasp_interface import save_spec, spec_from_grasp_archive
from sample.sample_grasp import GRAB_DATA, GRASP_FRAMES, HAND_CHANNELS, save_npz
from utils.mesh_to_bps import encode_object_bps, load_object_basis


GRAB_REFERENCES = (('M000582', 'mug', 's6'), ('M000543', 'train', 's7'))
IDENTITY_POSE = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)


def pose_arrays(values):
    return values[:, :3], rotation_matrix(values[:, 3:])


def finger_distances(distance, vertices, hand_model, window):
    surface = np.abs(distance.signed(vertices[-window:]))
    return np.stack([surface[:, indices].min(axis=1) for indices in hand_model.finger_vertices], axis=-1)


def calibrate(cli):
    """Measure both GRAB references using their fitted MANO templates and object poses."""
    refs = []
    for seq_id, object_name, subject in GRAB_REFERENCES:
        motion = np.load(cli.grab_data / 'representation_grasp' / f'{seq_id}.npy', allow_pickle=False)
        motion = motion[:GRASP_FRAMES].astype(np.float32)
        poses = pose_arrays(feature(motion, 'object_pose'))
        distance = MeshDistance(load_mesh(cli.grab_assets / 'contact_meshes' / f'{object_name}.ply'), cli.threads)
        ref = {'sequence': seq_id, 'object': object_name}
        for side in ('left', 'right'):
            template = load_mesh(cli.grab_assets / 'female' / f'{subject}_{side[0]}hand.ply').vertices
            model = ManoHand(cli.mano_dir, side, template=template)
            hand = decode_hand(motion, {side: model}, side)
            vertices = object_coordinates(hand.vertices, poses)
            joints = distance.signed(object_coordinates(hand.joints, poses))
            measured = {
                'min_signed_surface_distance_m': distance.minimum(vertices),
                'joint_distance_mae_m': float(np.mean(np.abs(joints - hand.predicted_distance))),
                'negative_measured_joint_distances': int((joints < 0).sum()),
                'negative_stored_joint_distances': int((hand.predicted_distance < 0).sum()),
            }
            if side == 'left':
                pads = finger_distances(distance, vertices, model, cli.contact_frames)
                measured['finger_distances_m'] = pads.tolist()
                measured['contact_radius_m'] = float(np.sort(pads, axis=-1)[:, cli.min_contact_fingers - 1].max())
            ref[side] = measured
        refs.append(ref)
    return refs


def get_thresholds(cli):
    fixed = (cli.contact_distance_m, cli.max_penetration_m, cli.left_clearance_m)
    refs = calibrate(cli) if None in fixed else []
    # Use the tighter observed grasp to set the contact radius. A loose
    # reference can be grasping through contact on just one digit or the palm.
    contact = cli.contact_distance_m
    if contact is None:
        contact = np.ceil(min(row['left']['contact_radius_m'] for row in refs) * 1000) / 1000
    penetration = cli.max_penetration_m
    if penetration is None:
        depth = max(max(0, -row['left']['min_signed_surface_distance_m']) for row in refs)
        penetration = np.ceil(depth * 1000) / 1000
    clearance = cli.left_clearance_m
    if clearance is None:
        clearance = min(0.05, min(row['right']['min_signed_surface_distance_m'] for row in refs))
    return {'contact_distance_m': float(contact), 'max_penetration_m': float(penetration),
            'left_clearance_m': float(clearance), 'contact_frames': cli.contact_frames,
            'min_contact_fingers': cli.min_contact_fingers, 'table_tolerance_m': cli.table_tolerance_m}, refs


def measure(right, left, hand_model, distance, poses, thresholds, table_z):
    rv = object_coordinates(right.vertices, poses)
    lv = object_coordinates(left.vertices, poses)
    right_joints = distance.signed(object_coordinates(right.joints, poses))
    fingers = finger_distances(distance, rv, hand_model, thresholds['contact_frames'])
    counts = (fingers <= thresholds['contact_distance_m']).sum(axis=-1)
    contact_needed = float(np.sort(fingers, axis=-1)[:, thresholds['min_contact_fingers'] - 1].max())
    depth = max(0.0, -distance.minimum(rv))
    clearance = distance.minimum(lv)
    table_clearance = None
    if table_z is not None:
        lowest_vertex = min(right.vertices[..., 2].min(), left.vertices[..., 2].min())
        table_clearance = float(lowest_vertex - table_z)
    checks = {
        'right_contact': int(counts.min()) >= thresholds['min_contact_fingers'],
        'right_penetration': depth <= thresholds['max_penetration_m'],
        'left_clearance': clearance >= thresholds['left_clearance_m'],
        'table_clearance': table_clearance is None or table_clearance >= -thresholds['table_tolerance_m'],
    }
    return {
        'passed': all(checks.values()), 'checks': checks,
        'right_contact_counts': counts.tolist(), 'right_finger_distances_m': fingers.tolist(),
        'right_required_contact_radius_m': contact_needed,
        'right_max_penetration_m': depth, 'left_min_signed_distance_m': clearance,
        'table_clearance_m': table_clearance,
        'right_prediction_distance_mae_m': float(np.mean(np.abs(right.predicted_distance - right_joints))),
    }, right_joints


def diversity(wrists, rotations):
    if len(wrists) < 2:
        return {'wrist_position_std_m': 0.0, 'median_wrist_rotation_delta_deg': 0.0}
    # Compare to a fixed sample so the diagnostic is linear in sample count.
    delta = rotations[0].T @ rotations[1:]
    return {'wrist_position_std_m': float(np.linalg.norm(wrists.std(axis=0))),
            'median_wrist_rotation_delta_deg': float(np.median(np.rad2deg(Rotation.from_matrix(delta).magnitude())))}


def rank(candidate):
    return (min(candidate['right_contact_counts']), -candidate['right_max_penetration_m'],
            -candidate['right_required_contact_radius_m'], candidate['left_min_signed_distance_m'])


def read_archive(path, mesh):
    with np.load(path, allow_pickle=False) as archive:
        motion = archive['motion']
        if (motion.ndim != 3 or motion.shape[1:] != (GRASP_FRAMES, HAND_CHANNELS)
                or len(motion) == 0 or not np.isfinite(motion).all()):
            raise ValueError(f'{path}: expected finite Stage 1 motion [N, 50, 108]')
        mirror = bool(archive['input_mirrored'].item())
        basis = load_object_basis(Path(archive['basis_path'].item()))
        vertices = mesh.vertices @ REFLECT_X if mirror else mesh.vertices
        expected = encode_object_bps(vertices, basis)
        bps = archive['object_bps']
        if bps.shape != (len(motion), 1, 1024, 3) or not np.allclose(bps, expected, atol=2e-8, rtol=0):
            raise ValueError(f'{path}: BPS and object mesh disagree')
        seeds = archive['seeds']
        if seeds.shape != (len(motion),):
            raise ValueError(f'{path}: missing seed for one or more motions')
    return motion.astype(np.float32), mirror, seeds


def evaluate(cli):
    mesh = load_mesh(cli.mesh)
    distance = MeshDistance(mesh, cli.threads)
    hands = {side: ManoHand(cli.mano_dir, side) for side in ('left', 'right')}
    object_pose = IDENTITY_POSE if cli.object_pose is None else np.load(cli.object_pose, allow_pickle=False)
    if object_pose.shape == (9,):
        object_pose = np.repeat(object_pose[None], GRASP_FRAMES, axis=0)
    if object_pose.shape != (GRASP_FRAMES, 9) or not np.isfinite(object_pose).all():
        raise ValueError('--object-pose must be a finite (9,) or (50, 9) array of translation + 6D rotation')
    poses = pose_arrays(object_pose)
    table_z = cli.table_z
    if cli.object_on_table:
        translation, rotation = poses
        table_z = float((mesh.vertices @ rotation[0, 2] + translation[0, 2]).min())

    thresholds, references = get_thresholds(cli)
    records, variation = [], []
    best, best_hand, best_joint_distances = None, None, None
    for path in cli.candidates:
        motions, mirrored, seeds = read_archive(path, mesh)
        wrists, rotations = [], []
        for index, motion in enumerate(motions):
            right = decode_hand(motion, hands, 'right', mirrored)
            left = decode_hand(motion, hands, 'left', mirrored)
            wrists.append(right.wrist[-1])
            rotations.append(right.rotation[-1])
            record, joints = measure(right, left, hands['right'], distance, poses, thresholds, table_z)
            record.update(source=str(path.resolve()), index=index, seed=int(seeds[index]), mirrored=mirrored)
            records.append(record)
            if record['passed'] and (best is None or rank(record) > rank(best)):
                best, best_hand, best_joint_distances = record, right, joints
        variation.append(dict(source=str(path.resolve()), **diversity(np.stack(wrists), np.stack(rotations))))

    report = dict(mesh=str(cli.mesh.resolve()), object_pose=object_pose.tolist(),
                  table_z_m=table_z, thresholds=thresholds,
                  references=references, diversity=variation, selected=best, candidates=records)
    output = cli.output_dir.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / 'selection'
        bundle.mkdir()
        (bundle / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        if best is not None:
            save_npz(bundle / 'right_hand_grasp.npz', wrist_position=best_hand.wrist,
                     wrist_rotation=best_hand.rotation, hand_pose_axis_angle=best_hand.pose,
                     joints=best_hand.joints, fingertips=best_hand.joints[:, 16:], fingertip_order=FINGERS,
                     vertices=best_hand.vertices, faces=hands['right'].model.faces,
                     joint_signed_distance=best_joint_distances, predicted_joint_distance=best_hand.predicted_distance,
                     object_pose=object_pose, frame_index=np.arange(GRASP_FRAMES),
                     object_mesh=str(cli.mesh.resolve()), source_archive=best['source'], source_index=best['index'],
                     seed=best['seed'], mirrored=best['mirrored'],
                     finger_distance_m=np.asarray(best['right_finger_distances_m'])[-1],
                     contact_radius_m=thresholds['contact_distance_m'])
            # The robot-agnostic description a retargeting step consumes (see grasp_interface).
            save_spec(bundle / 'grasp_spec.npz', spec_from_grasp_archive(bundle / 'right_hand_grasp.npz'),
                      source_grasp='right_hand_grasp.npz')
        bundle.rename(output)
    count = sum(row['passed'] for row in records)
    print(f'{count}/{len(records)} candidates passed. Evaluation report: {output / "report.json"}')
    if best is not None:
        print(f'Selected seed {best["seed"]}: {output / "right_hand_grasp.npz"}')
    return best is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, nargs='+', required=True, help='One or more sample_grasp archives')
    parser.add_argument('--mesh', type=Path, required=True, help='Original prepared object_mesh.ply in metres')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for report and selected trajectory')
    parser.add_argument('--object-pose', type=Path, help='Object translation + 6D rotation as .npy, default identity')
    parser.add_argument('--mano-dir', type=Path, default=Path('assets/smplx/mano'))
    parser.add_argument('--grab-data', type=Path, default=GRAB_DATA)
    parser.add_argument('--grab-assets', type=Path, default=Path('assets'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--contact-frames', type=int, default=3)
    parser.add_argument('--min-contact-fingers', type=int, default=2)
    parser.add_argument('--contact-distance-m', type=float)
    parser.add_argument('--max-penetration-m', type=float)
    parser.add_argument('--left-clearance-m', type=float)
    table = parser.add_mutually_exclusive_group()
    table.add_argument('--table-z', type=float, help='Measured table height in the motion frame, metres')
    table.add_argument('--object-on-table', action='store_true', help='Use the object bottom at frame 0 as the table')
    parser.add_argument('--table-tolerance-m', type=float, default=0.001)
    cli = parser.parse_args()
    if not 1 <= cli.contact_frames <= GRASP_FRAMES or not 1 <= cli.min_contact_fingers <= len(FINGERS):
        parser.error('Contact frames must be 1–50 and contact fingers must be 1–5')
    lengths = (cli.contact_distance_m, cli.max_penetration_m, cli.left_clearance_m, cli.table_tolerance_m)
    if cli.threads < 0 or any(value is not None and (not np.isfinite(value) or value < 0) for value in lengths):
        parser.error('Threads and metric thresholds must be finite and nonnegative')
    if cli.table_z is not None and not np.isfinite(cli.table_z):
        parser.error('--table-z must be finite')
    try:
        selected = evaluate(cli)
    except (KeyError, ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    if not selected:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
