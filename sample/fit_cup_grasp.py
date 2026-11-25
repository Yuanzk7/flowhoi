"""Fit a right-hand side grasp to an upright cup and retarget a Stage 1 approach."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import torch
from torch.nn import functional as F
import trimesh

from sample.grasp_geometry import ManoHand, MeshDistance, decode_hand, feature, load_mesh
from sample.grasp_interface import palm_deviation, palm_frame
from sample.sample_grasp import GRAB_DATA, GRASP_FRAMES, HAND_CHANNELS, save_npz
from sample.select_grasp import get_thresholds, measure, read_archive
from utils.mesh_to_bps import encode_object_bps, load_object_basis
from utils.rotation_conversions import quaternion_to_matrix


MIN_OPPOSITION_DEG = 120
MODEL_YAW_DELTAS_DEG = (0, 20, -20, 40, -40, 65, -65, 90, -90, 135, -135, 180)
METHODS = {'model': 'FlowHOI trajectory moved to the fitted cup grasp',
           'synth': 'Geometry-fitted cup approach (synthesized)'}


def unit(vector):
    return vector / np.linalg.norm(vector)


def smoothstep(t):
    return t * t * (3 - 2 * t)


def as_pca(hand, model):
    return hand.pose @ np.linalg.pinv(model.components)


def cup_contacts(distance, yaw, height):
    """Trace from outside the cup so the targets lie on its outer wall."""
    lower, upper = distance.bounds
    center = (lower + upper) / 2
    center[2] = height
    angles = yaw + np.deg2rad([0, 105, 137, 164, 190])
    directions = np.stack((np.cos(angles), np.sin(angles), np.zeros(len(angles))), axis=-1)
    origins = center + directions * np.linalg.norm(upper - lower)
    origins[:, 2] += [0, .011, .003, -.006, -.015]
    rays = np.column_stack((origins, -directions)).astype(np.float32)
    hits = distance.scene.cast_rays(distance.o3d.core.Tensor(rays), nthreads=distance.threads)['t_hit'].numpy()
    if not np.isfinite(hits).all():
        raise ValueError('Cup wall is missing at one or more grasp target heights')
    wall = origins - directions * hits[:, None]
    return (wall + .00045 * directions).astype(np.float32)


def align_hand(hand, targets):
    """Place the hand's thumb and finger span across the cup wall."""
    tips = hand.joints[0, 16:]
    span = unit(tips[2] - tips[0])
    forward = (tips[0] + tips[2]) / 2 - hand.wrist[0]
    forward = unit(forward - (forward @ span) * span)
    source = np.stack((span, forward, np.cross(span, forward)), axis=1)
    across = targets[2] - targets[0]
    across[2] = 0
    across = unit(across)
    into_cup = np.cross([0, 0, 1], across)
    destination = np.stack((across, into_cup, np.cross(across, into_cup)), axis=1)
    adjustment = destination @ source.T
    rotation = adjustment @ hand.rotation[0]
    grip_center = (tips[0] + tips[2]) / 2 - hand.wrist[0]
    wrist = (targets[0] + targets[2]) / 2 - adjustment @ grip_center
    return wrist.astype(np.float32), rotation.astype(np.float32)


class PoseFit:
    def __init__(self, hand, distance, envelope, table_z):
        self.hand = hand
        self.distance = distance
        self.envelope = envelope
        self.table_z = table_z
        self.hand.model.requires_grad_(False)
        self.components = torch.from_numpy(hand.components)
        self.weights = torch.tensor([1.5, .65, 1.6, .25, .1])
        self.pads = []
        points = hand.model.v_template.detach().numpy()
        for indices, tip in zip(hand.finger_vertices, hand.tips):
            # Use distal surfaces to make thumb opposition measurable.
            closest = np.argsort(np.linalg.norm(points[indices] - points[tip], axis=-1))[:40]
            self.pads.append(torch.from_numpy(indices[closest]))

    def vertices(self, x, wrist, rotation, pca):
        offset = x[3:6] * .65
        angle = torch.linalg.vector_norm(offset)
        quat = torch.cat((torch.cos(angle / 2).reshape(1), .5 * torch.sinc(angle / (2 * torch.pi)) * offset))
        r = quaternion_to_matrix(quat) @ rotation
        p = pca + x[6:] * .75
        w = wrist + x[:3] * .05
        result = self.hand.model(hand_pose=(p @ self.components)[None])
        verts = (result.vertices[0] - result.joints[0, 0]) @ r.T + w
        return verts, w, r, p

    @staticmethod
    def signed_with_gradient(vertices, distance):
        points = vertices.detach().numpy()
        closest = distance.scene.compute_closest_points(
            distance.o3d.core.Tensor(points), nthreads=distance.threads)
        signed = distance.signed(points)
        delta = points - closest['points'].numpy()
        length = np.linalg.norm(delta, axis=-1, keepdims=True)
        normals = delta * np.sign(signed[:, None]) / np.maximum(length, 1e-8)
        normals[length[:, 0] < 1e-8] = closest['primitive_normals'].numpy()[length[:, 0] < 1e-8]
        return torch.from_numpy(signed) + ((vertices - vertices.detach()) * torch.from_numpy(normals)).sum(-1)

    def fit(self, hand, pca, targets, steps):
        wrist, rotation = align_hand(hand, targets)
        w, r, p, desired = [torch.as_tensor(item, dtype=torch.float32) for item in (wrist, rotation, pca, targets)]

        def objective(values):
            x = torch.tensor(values, dtype=torch.float32, requires_grad=True)
            vertices, *_ = self.vertices(x, w, r, p)
            d = self.signed_with_gradient(vertices, self.distance)
            exterior = self.signed_with_gradient(vertices, self.envelope)
            contacts = []
            for group, target in zip(self.pads, desired):
                location = ((vertices[group] - target) * 270).square().sum(-1)
                surface = ((d[group] - .0004) * 1000).square()
                contacts.append((.22 * surface + location).topk(3, largest=False).values.mean())
            penetration = F.relu((-d - .0004) * 1000)
            interior = F.relu((-exterior - .0006) * 1000)
            floor = F.relu((self.table_z + .002 - vertices[:, 2]) * 1000)
            loss = (torch.stack(contacts) * self.weights).sum()
            loss += 2 * penetration.topk(24).values.square().mean() + 2 * penetration.max().square()
            loss += 4 * interior.topk(24).values.square().mean() + 6 * interior.max().square()
            loss += 2 * floor.max().square() + .11 * x[6:].square().sum() + .09 * x[3:6].square().sum()
            loss.backward()
            return float(loss.detach()), x.grad.detach().numpy().astype(np.float64)

        result = minimize(objective, np.zeros(30), method='L-BFGS-B', jac=True,
                          bounds=[(-1.7, 1.7)] * 3 + [(-1.4, 1.4)] * 3 + [(-2., 2.)] * 24,
                          options={'maxiter': steps, 'ftol': 2e-8})
        with torch.no_grad():
            _, w, r, p = self.vertices(torch.tensor(result.x, dtype=torch.float32), w, r, p)
        return w.numpy(), r.numpy(), p.numpy()


def lift_to_table(wrist, rotation, pca, model, table_z, predicted):
    hand = model.fk(wrist, rotation, pca @ model.components, predicted)
    lift = max(0, table_z + .003 - hand.vertices[..., 2].min())
    wrist = wrist.copy()
    wrist[:, 2] += lift
    return wrist


def put_hand(motion, side, wrist, rotation, pca, predicted):
    feature(motion, f'pos_{side}')[:] = wrist
    feature(motion, f'global_orient_{side[0]}')[:] = rotation[:, :2].reshape(-1, 6)
    feature(motion, f'pca_{side}')[:] = pca
    feature(motion, f'sdf_{side}')[:] = predicted


def approach(source, hands, goal, table_z, stand_off, center):
    right, left = source['right'], source['left']
    goal_wrist, goal_rot, goal_pca = goal
    frames = len(right.wrist)
    start, stop = 31, frames - 5
    direction = unit(np.r_[goal_wrist[:2] - center[:2], 0])
    ready = goal_wrist + direction * stand_off
    initial = right.wrist[0].copy()
    initial[2] = max(initial[2], ready[2])

    # Keep the approaching wrist outside the cup even if it starts on the far side.
    a = np.clip(np.arange(frames) / start, 0, 1)
    a = smoothstep(a)
    r0, r1 = np.linalg.norm(initial[:2] - center[:2]), np.linalg.norm(ready[:2] - center[:2])
    theta_0 = np.arctan2(*(initial[:2] - center[:2])[::-1])
    theta_1 = np.arctan2(*(ready[:2] - center[:2])[::-1])
    theta = theta_0 + a * ((theta_1 - theta_0 + np.pi) % (2 * np.pi) - np.pi)
    radius = r0 + (r1 - r0) * a
    wrist = np.column_stack((center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta),
                             initial[2] + (ready[2] - initial[2]) * a))
    close = smoothstep(np.clip((np.arange(frames) - start) / (stop - start), 0, 1))
    wrist += close[:, None] * (goal_wrist - ready)
    wrist = wrist.astype(np.float32)
    blend = smoothstep(np.clip(np.arange(frames) / (stop - 7), 0, 1))
    source_pca = as_pca(right, hands['right'])
    pca = (source_pca * (1 - blend[:, None]) + goal_pca * blend[:, None]).astype(np.float32)
    relative = Rotation.from_matrix(right.rotation.transpose(0, 2, 1) @ goal_rot).as_rotvec()
    rotation = (right.rotation @ Rotation.from_rotvec(relative * blend[:, None]).as_matrix()).astype(np.float32)

    # A constant raise preserves the original idle hand motion above the table.
    left_pca = as_pca(left, hands['left'])
    left_wrist = lift_to_table(left.wrist, left.rotation, left_pca, hands['left'], table_z, left.predicted_distance)
    # Guarantee the source approach clears the table while preserving the fitted endpoint.
    proposed = hands['right'].fk(wrist, rotation, pca @ hands['right'].components, right.predicted_distance)
    lift = np.maximum(table_z + .002 - proposed.vertices[..., 2].min(axis=1), 0)
    wrist[:, 2] += lift
    motion = np.zeros((frames, HAND_CHANNELS), dtype=np.float32)
    put_hand(motion, 'right', wrist, rotation, pca, right.predicted_distance)
    put_hand(motion, 'left', left_wrist, left.rotation, left_pca, left.predicted_distance)
    return motion


def follow_model(source, hands, goal, table_z):
    """Move the model's own right-hand trajectory rigidly so it ends at the fitted grasp.

    The palm-relative approach direction and the timing of the hand opening and closing are
    FlowHOI's; only the final pose is corrected, with the finger shape blended in over time.
    """
    right, left = source['right'], source['left']
    goal_wrist, goal_rot, goal_pca = goal
    frames = len(right.wrist)
    correction = goal_rot @ right.rotation[-1].T
    wrist = ((right.wrist - right.wrist[-1]) @ correction.T + goal_wrist).astype(np.float32)
    rotation = (correction @ right.rotation).astype(np.float32)
    source_pca = as_pca(right, hands['right'])
    ramp = smoothstep(np.linspace(0, 1, frames))
    pca = (source_pca + ramp[:, None] * (goal_pca - source_pca[-1])).astype(np.float32)

    left_pca = as_pca(left, hands['left'])
    left_wrist = lift_to_table(left.wrist, left.rotation, left_pca, hands['left'], table_z, left.predicted_distance)
    proposed = hands['right'].fk(wrist, rotation, pca @ hands['right'].components, right.predicted_distance)
    wrist[:, 2] += np.maximum(table_z + .002 - proposed.vertices[..., 2].min(axis=1), 0)
    motion = np.zeros((frames, HAND_CHANNELS), dtype=np.float32)
    put_hand(motion, 'right', wrist, rotation, pca, right.predicted_distance)
    put_hand(motion, 'left', left_wrist, left.rotation, left_pca, left.predicted_distance)
    return motion


def start_yaw(offset, model_yaw):
    """Thumb contact azimuth for a fitting start: near the model's own grasp first, or the
    original golden-angle spread when ``model_yaw`` is None."""
    golden = np.pi * (3 - np.sqrt(5))
    if model_yaw is None:
        return offset * golden
    return model_yaw + np.deg2rad(MODEL_YAW_DELTAS_DEG[offset % len(MODEL_YAW_DELTAS_DEG)]) \
        + (offset // len(MODEL_YAW_DELTAS_DEG)) * golden


def opposed_contact_angles(vertices, hand, distance, center, radius):
    """Find the most opposed simultaneous thumb and finger contacts in each frame."""
    signed = distance.signed(vertices)
    angles = []
    fingers = np.concatenate(hand.finger_vertices[1:])
    for points, row in zip(vertices, signed):
        radial = points[:, :2] - center[:2]
        length = np.linalg.norm(radial, axis=1)
        valid = (np.abs(row) <= radius) & (length > 1e-5)
        radial[valid] /= length[valid, None]
        thumb = radial[hand.finger_vertices[0][valid[hand.finger_vertices[0]]]]
        others = radial[fingers[valid[fingers]]]
        cosine = (thumb @ others.T).min() if len(thumb) and len(others) else 1.
        angles.append(float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))))
    return angles


def outer_wall_intrusion(vertices, distance, center):
    """Measure how far hand vertices enter the radial envelope below the rim."""
    lower, upper = distance.bounds
    points = vertices.reshape(-1, 3)
    points = points[(points[:, 2] >= lower[2]) & (points[:, 2] <= upper[2])]
    if not len(points):
        return 0.0
    radial = points[:, :2] - center[:2]
    length = np.linalg.norm(radial, axis=1)
    reach = np.linalg.norm(upper - lower)
    near = length < reach
    if not near.any():
        return 0.0
    points, radial, length = points[near], radial[near], length[near]
    direction = np.column_stack((radial, np.zeros(len(radial))))
    direction[length < 1e-9] = [1, 0, 0]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    origins = np.column_stack((np.broadcast_to(center[:2], radial.shape), points[:, 2])) + reach * direction
    rays = np.column_stack((origins, -direction)).astype(np.float32)
    hits = distance.scene.cast_rays(distance.o3d.core.Tensor(rays), nthreads=distance.threads)['t_hit'].numpy()
    return float(max(0, np.max(reach - hits - length)))


def sealed_cup_envelope(mesh):
    """Extend the closed cup volume above its rim to keep fingers outside the opening."""
    hull = mesh.convex_hull
    rim = hull.vertices[hull.vertices[:, 2] > mesh.bounds[1, 2] - mesh.extents[2] * .12].copy()
    rim[:, 2] += mesh.extents[2] * .44
    return trimesh.convex.convex_hull(np.concatenate((hull.vertices, rim)))


def contact_variants(goal, model, distance, center, radius):
    """Recover contacts just outside tolerance with small local pose adjustments."""
    wrist, rotation, pca = goal

    def angle(pose):
        w, r, p = pose
        hand = model.fk(w[None], r[None], (p @ model.components)[None], np.zeros((1, 21), dtype=np.float32))
        return opposed_contact_angles(hand.vertices, model, distance, center, radius)[0]

    initial = angle(goal)
    print(f'Endpoint thumb opposition {initial:.1f} deg', flush=True)
    if initial >= MIN_OPPOSITION_DEG:
        yield goal
        return
    if initial < 105:
        return
    # Small shape, orientation and wrist adjustments refine the contact patch
    # without changing the grasp family or loosening the geometric checks.
    rng = np.random.default_rng(92)
    yielded = 0
    for _ in range(120):
        delta = rng.uniform(-1, 1, 7)
        w = (wrist + delta[:3] * .0009).astype(np.float32)
        r = (Rotation.from_rotvec(delta[3:6] * np.deg2rad(1.5)).as_matrix() @ rotation).astype(np.float32)
        p = (pca * (1 + delta[6] * .018)).astype(np.float32)
        pose = (w, r, p)
        if angle(pose) >= MIN_OPPOSITION_DEG:
            yield pose
            yielded += 1
            if yielded == 4:
                return


def fit_grasp(cli):
    mesh = load_mesh(cli.mesh)
    field = MeshDistance(mesh, cli.threads)
    envelope = MeshDistance(sealed_cup_envelope(mesh), cli.threads)
    motions, mirrored, seeds = read_archive(cli.candidates, mesh)
    if not 0 <= cli.index < len(motions):
        raise ValueError('Candidate index is out of range')
    with np.load(cli.candidates, allow_pickle=False) as archive:
        basis_path = Path(archive['basis_path'].item())
        text = archive['text'][cli.index]
    hands = {side: ManoHand(cli.mano_dir, side) for side in ('left', 'right')}
    source = {side: decode_hand(motions[cli.index], hands, side, mirrored) for side in hands}
    raw = decode_hand(motions[cli.index, -1:], hands, 'right', mirrored)
    raw_pca = as_pca(raw, hands['right'])[0]
    center = mesh.bounds.mean(axis=0)
    table_z = float(mesh.bounds[0, 2]) if cli.table_z is None else cli.table_z
    identity = (np.zeros((GRASP_FRAMES, 3), dtype=np.float32),
                np.repeat(np.eye(3, dtype=np.float32)[None], GRASP_FRAMES, axis=0))
    thresholds, _ = get_thresholds(cli)
    solver = PoseFit(hands['right'], field, envelope, table_z)
    model_palm = palm_frame(raw.joints[0])
    thumb = raw.joints[0, 16] - center
    model_yaw = float(np.arctan2(thumb[1], thumb[0])) if cli.yaw_source == 'model' else None
    if model_yaw is not None:
        print(f'Starting from the model thumb azimuth {np.rad2deg(model_yaw):.0f} deg', flush=True)

    def motions(method, proposed):
        if method == 'model':
            yield 'model trajectory', follow_model(source, hands, proposed, table_z)
        else:
            for stand_off in (.065, .095):
                yield f'stand-off {stand_off:.3f} m', approach(source, hands, proposed, table_z, stand_off, center)

    def attempt(method, proposed, yaw):
        for label, motion in motions(method, proposed):
            right = decode_hand(motion, hands, 'right')
            left = decode_hand(motion, hands, 'left')
            metrics, _ = measure(right, left, hands['right'], field, identity, thresholds, table_z)
            angles = opposed_contact_angles(right.vertices[-cli.contact_frames:], hands['right'], field, center,
                                            thresholds['contact_distance_m'])
            intrusion = outer_wall_intrusion(right.vertices, field, center)
            offset_m, rotation_deg = palm_deviation(model_palm, palm_frame(right.joints[-1]))
            metrics.update(thumb_opposition_deg=angles, outer_wall_intrusion_m=intrusion,
                           palm_offset_from_model_m=offset_m.tolist(), palm_rotation_from_model_deg=rotation_deg,
                           start_yaw_deg=float(np.rad2deg(yaw)), yaw_source=cli.yaw_source, approach_method=method)
            print(f'{label}: contacts {metrics["right_contact_counts"]}, '
                  f'minimum thumb opposition {min(angles):.1f} deg, '
                  f'max penetration {metrics["right_max_penetration_m"] * 1000:.2f} mm, '
                  f'outer wall intrusion {intrusion * 1000:.2f} mm, '
                  f'palm vs model {np.linalg.norm(offset_m) * 1000:.0f} mm / {rotation_deg:.0f} deg', flush=True)
            if not metrics['passed'] or min(angles) < MIN_OPPOSITION_DEG or intrusion > thresholds['max_penetration_m']:
                continue
            bps = encode_object_bps(mesh.vertices, load_object_basis(basis_path))
            save_npz(cli.output, motion=motion[None], text=[text], seeds=[seeds[cli.index]],
                     input_mirrored=False, basis_path=str(basis_path), object_bps=bps[None],
                     mesh_path=str(cli.mesh.resolve()), source_archive=str(cli.candidates.resolve()),
                     source_index=cli.index, trajectory_method=METHODS[method],
                     metrics_json=json.dumps(metrics))
            print(f'Saved verified cup grasp ({METHODS[method]}) to {cli.output}')
            return True
        return False

    # Fitting is the expensive part, so fitted endpoints are kept for the synthesized fallback.
    fitted = []
    first = 'model' if cli.approach == 'model' else 'synth'
    for offset in range(cli.starts):
        yaw = start_yaw(offset, model_yaw)
        height = center[2] - mesh.extents[2] * (.15 if offset % 4 < 2 else .22)
        pose = raw_pca * (.75, .55, .9)[offset % 3]
        prior = hands['right'].fk(raw.wrist, raw.rotation, (pose @ hands['right'].components)[None],
                                 raw.predicted_distance)
        targets = cup_contacts(field, yaw, height)
        goal = solver.fit(prior, pose, targets, cli.steps)
        print(f'Fit {offset + 1}/{cli.starts}', flush=True)
        for proposed in contact_variants(goal, hands['right'], field, center, thresholds['contact_distance_m']):
            if attempt(first, proposed, yaw):
                return
            fitted.append((proposed, yaw))
    if first == 'model':
        print('No moved FlowHOI trajectory passed; trying synthesized approaches', flush=True)
        for proposed, yaw in fitted:
            if attempt('synth', proposed, yaw):
                return
    raise ValueError('No fitted cup approach met the contact, clearance, and penetration criteria')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--mesh', type=Path, required=True, help='Prepared upright cup mesh in metres')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mano-dir', type=Path, default=Path('assets/smplx/mano'))
    parser.add_argument('--grab-data', type=Path, default=GRAB_DATA)
    parser.add_argument('--grab-assets', type=Path, default=Path('assets'))
    parser.add_argument('--table-z', type=float, help='Table height in metres; defaults to the cup base')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--starts', type=int, default=12)
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--yaw-source', choices=('model', 'legacy'), default='model',
                        help="model: start fitting at the generated grasp's thumb azimuth. legacy: golden-angle spread")
    parser.add_argument('--approach', choices=('model', 'synth'), default='model',
                        help='model: keep the generated trajectory, moved to end at the fitted grasp, and fall back '
                             'to synthesized approaches. synth: synthesized approaches only, as before')
    cli = parser.parse_args()
    cli.contact_frames = 3
    cli.min_contact_fingers = 2
    cli.table_tolerance_m = .001
    cli.contact_distance_m = cli.max_penetration_m = cli.left_clearance_m = None
    if min(cli.threads, cli.starts, cli.steps) < 1 or (cli.table_z is not None and not np.isfinite(cli.table_z)):
        parser.error('Expected positive thread and search limits and a finite table height')
    if cli.output.suffix != '.npz' or cli.output.resolve() == cli.candidates.resolve():
        parser.error('--output must be an .npz path separate from the input archive')
    try:
        fit_grasp(cli)
    except (KeyError, ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
