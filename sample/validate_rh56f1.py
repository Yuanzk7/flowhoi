"""Retarget a MANO grasp to RH56F1 and test an actuated lift in MuJoCo."""

import argparse
import importlib.util
import json
import shutil
import sys
from itertools import product
from pathlib import Path
from struct import pack
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import trimesh

from sample import rh56f1_retarget


DEFAULT_MJCF = rh56f1_retarget.DEFAULT_URDF.with_name('right.xml')
FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')
FPS = 30
DT = 1 / 600
LIFT_HEIGHT = .075
MIN_HEIGHT = .050
HOLD_SECONDS = 2.


def load_retarget(path=None):
    if path is None:
        return rh56f1_retarget
    spec = importlib.util.spec_from_file_location('rh56f1_reference', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def text(values):
    return ' '.join(f'{float(value):.10g}' for value in values)


def transform(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def rotation6d(values):
    a, b = np.asarray(values, dtype=float).reshape(2, 3)
    a = a / np.linalg.norm(a)
    b = b - (a @ b) * a
    b = b / np.linalg.norm(b)
    rotation = np.stack((a, b, np.cross(a, b)))
    if not np.isfinite(rotation).all():
        raise ValueError('The object has an invalid 6D rotation')
    return rotation


def read_grasp(path, frame):
    with np.load(path, allow_pickle=False) as data:
        indices = np.flatnonzero(data['frame_index'] == frame)
        if len(indices) != 1 or tuple(data['fingertip_order']) != ('thumb', 'index', 'middle', 'ring', 'pinky'):
            raise ValueError('Expected the selected frame and MANO fingertip order in a right_hand_grasp archive')
        i = int(indices[0])
        pose, fingers, joints = (data[name][i] for name in ('object_pose', 'fingertips', 'joints'))
        mesh = Path(str(data['object_mesh'].item()))
    obj = transform(rotation6d(pose[3:]), pose[:3])
    fingers = (fingers - obj[:3, 3]) @ obj[:3, :3]
    joints = (joints - obj[:3, 3]) @ obj[:3, :3]
    if (fingers.shape != (5, 3) or joints.shape != (21, 3)
            or not np.isfinite(fingers).all() or not np.isfinite(joints).all()):
        raise ValueError('Expected finite MANO fingertips and joints')
    return obj, fingers, joints, mesh


def write_msh(path, mesh):
    with path.open('wb') as stream:
        stream.write(pack('<4i', len(mesh.vertices), 0, 0, len(mesh.faces)))
        np.asarray(mesh.vertices, dtype='<f4').tofile(stream)
        np.asarray(mesh.faces, dtype='<i4').tofile(stream)


def joint_name(name):
    return name.replace('_little_', '_pinky_')


def build_scene(directory, source, mesh, obj, robot, mass, friction):
    """Adapt the installed RH56F1 model to a fixed-orientation, actuated lift rig."""
    xml = ET.parse(source).getroot()
    xml.set('model', 'RH56F1 cup lift')
    for key in ('option', 'default', 'visual', 'equality', 'actuator', 'contact', 'keyframe', 'sensor'):
        for element in xml.findall(key):
            xml.remove(element)
    compiler = xml.find('compiler')
    meshdir = source.parent / compiler.get('meshdir', '')
    compiler.attrib.pop('meshdir', None)
    compiler.set('fusestatic', 'false')
    ET.SubElement(xml, 'option', timestep=str(DT), integrator='implicitfast', cone='elliptic',
                  iterations='60', gravity='0 0 -9.81')
    default = ET.SubElement(xml, 'default')
    ET.SubElement(default, 'geom', contype='0', conaffinity='0', solref='.008 1', solimp='.94 .99 .001')
    ET.SubElement(default, 'joint', damping='.045', armature='.00035')
    visual = ET.SubElement(xml, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='720', fovy='39')
    ET.SubElement(visual, 'headlight', ambient='.5 .5 .5', diffuse='.7 .7 .7', specular='.16 .16 .16')

    assets = xml.find('asset')
    (directory / 'hand_meshes').mkdir()
    for element in list(assets):
        if element.tag != 'mesh':
            assets.remove(element)
            continue
        original = meshdir / element.get('file')
        copied = directory / 'hand_meshes' / original.name
        if not copied.exists():
            shutil.copyfile(original, copied)
        element.set('file', f'hand_meshes/{original.name}')
    license_path = source.parent / 'LICENSE'
    if not license_path.is_file():
        license_path = source.parent / 'urdf_src' / 'LICENSE'
    if license_path.is_file():
        shutil.copyfile(license_path, directory / 'RH56F1_LICENSE.txt')
    ET.SubElement(assets, 'texture', type='skybox', builtin='gradient', rgb1='.23 .28 .34', rgb2='.11 .14 .19',
                  width='64', height='512')
    ET.SubElement(assets, 'texture', name='grid', type='2d', builtin='checker', mark='edge',
                  rgb1='.27 .32 .37', rgb2='.3 .35 .40', markrgb='.43 .48 .52', width='256', height='256')
    ET.SubElement(assets, 'material', name='table_grid', texture='grid', texrepeat='6 6')
    write_msh(directory / 'cup_visual.msh', mesh)
    # The convex exterior is used for side contacts, with the full open cup shown in the replay.
    hull = mesh.convex_hull
    write_msh(directory / 'cup_collision.msh', hull)
    for name in ('cup_visual', 'cup_collision'):
        ET.SubElement(assets, 'mesh', name=name, file=f'{name}.msh', smoothnormal='true')

    world = xml.find('worldbody')
    hand = world.find("body[@name='right_hand_base']")
    if hand is None:
        raise ValueError('The RH56F1 MJCF must contain right_hand_base')
    world.remove(hand)
    for element in list(world):
        world.remove(element)
    for element in list(hand):
        if element.tag in ('joint', 'camera'):
            hand.remove(element)
    for body in hand.iter('body'):
        body.set('gravcomp', '1')
    for element in hand.iter('geom'):
        if element.get('type') == 'mesh':
            element.attrib.update(contype='1', conaffinity='0', condim='4',
                                  friction=f'{friction} .004 .0002')
        else:
            element.attrib.update(contype='0', conaffinity='0')
    ranges = dict(zip(map(joint_name, robot.ACTUATED), robot.UPPER))
    ranges.update({joint_name(follower): ranges[joint_name(driver)] * ratio
                   for follower, (driver, ratio) in robot.COUPLED.items()})
    for element in hand.iter('joint'):
        name = element.get('name')
        if name in ranges:
            element.set('range', text((0, ranges[name])))
    for site in hand.iter('site'):
        site.set('group', '3')
    carriage = ET.SubElement(world, 'body', name='lift_carriage', gravcomp='1')
    ET.SubElement(carriage, 'inertial', pos='0 0 0', mass='.04', diaginertia='.00002 .00002 .00002')
    ET.SubElement(carriage, 'joint', name='wrist_lift', type='slide', axis='0 0 1', range='-.01 .13', damping='4')
    carriage.append(hand)

    table_z = float((mesh.vertices @ obj[2, :3] + obj[2, 3]).min())
    ET.SubElement(world, 'light', pos='.1 -.4 1.5', dir='0 0 -1', diffuse='.58 .57 .55', castshadow='false')
    ET.SubElement(world, 'geom', name='table', type='box', size='.45 .45 .025', material='table_grid',
                  pos=text((obj[0, 3], obj[1, 3], table_z - .025)), contype='4', conaffinity='3',
                  condim='3', friction='.48 .002 .0001', priority='1')
    cup = ET.SubElement(world, 'body', name='cup', pos=text(obj[:3, 3]),
                        quat=text(Rotation.from_matrix(obj[:3, :3]).as_quat(scalar_first=True)))
    ET.SubElement(cup, 'freejoint', name='cup_free')
    ET.SubElement(cup, 'geom', name='cup_render', type='mesh', mesh='cup_visual', mass='0', rgba='.31 .82 .68 1')
    ET.SubElement(cup, 'geom', name='cup_contact', type='mesh', mesh='cup_collision', mass=str(mass), group='3',
                  contype='2', conaffinity='1', condim='4', friction=f'{friction} .004 .0002')

    equality = ET.SubElement(xml, 'equality')
    for follower, (driver, ratio) in robot.COUPLED.items():
        ET.SubElement(equality, 'joint', joint1=joint_name(follower), joint2=joint_name(driver),
                      polycoef=text((0, ratio, 0, 0, 0)), solref='.003 1', solimp='.98 .999 .0002')
    actuators = ET.SubElement(xml, 'actuator')
    ET.SubElement(actuators, 'position', name='lift_servo', joint='wrist_lift', kp='1800', kv='55',
                  ctrlrange='0 .1', forcerange='-32 32')
    for name, upper in zip(robot.ACTUATED, robot.UPPER):
        ET.SubElement(actuators, 'position', name=joint_name(name) + '_servo', joint=joint_name(name), kp='5',
                      kv='.075', ctrlrange=text((0, upper)), forcerange='-.26 .26')
    ET.SubElement(xml, 'statistic', center=text(obj[:3, 3] + np.array([0, 0, .05])), extent='.45')
    return xml, hand, hull, table_z


def hand_fingers(model):
    fingers = {}
    for geom in np.flatnonzero(model.geom_contype == 1):
        name = model.geom(int(geom)).name
        fingers[int(geom)] = next((finger for finger in FINGERS if finger in name), 'palm')
    return fingers


def cup_contact_forces(model, data, cup_geom, fingers):
    active = {}
    force = np.zeros(6)
    for index, contact in enumerate(data.contact):
        first, second = map(int, contact.geom)
        other = second if first == cup_geom else first if second == cup_geom else -1
        if other in fingers:
            mujoco.mj_contactForce(model, data, index, force)
            finger = fingers[other]
            active[finger] = active.get(finger, 0.) + float(force[0])
    return active


class LiftTrial:
    def __init__(self, model, robot, object_pose, hull, table_z):
        self.model = model
        self.robot = robot
        self.object_pose = object_pose
        self.hull_points = np.asarray(hull.vertices)
        self.hull_normals = np.asarray(hull.face_normals)
        self.hull_offsets = np.einsum('ij,ij->i', self.hull_normals, hull.triangles[:, 0])
        self.table_z = table_z
        self.data = mujoco.MjData(model)
        self.hand = model.body('right_hand_base').id
        self.cup = model.body('cup').id
        self.cup_geom = model.geom('cup_contact').id
        self.table_geom = model.geom('table').id
        self.driven = np.array([model.joint(joint_name(name)).qposadr.item() for name in robot.ACTUATED])
        self.followers = [(model.joint(joint_name(name)).qposadr.item(), robot.ACTUATED.index(driver), ratio)
                          for name, (driver, ratio) in robot.COUPLED.items()]
        self.tip_bodies = [model.body(name).id for name in robot.TIPS]
        self.finger = hand_fingers(model)
        self.hand_geoms = list(self.finger)
        self.distal = {name: [i for i in self.hand_geoms if self.finger[i] == name and any(
                       suffix in model.geom(i).name for suffix in ('tip', 'sensor', 'thumb_4', name + '_2'))]
                       for name in FINGERS}

    def set_flange(self, flange):
        world = self.object_pose @ flange
        self.model.body_pos[self.hand] = world[:3, 3] + world[:3, 2] * self.robot.ADAPTER_Z
        self.model.body_quat[self.hand] = Rotation.from_matrix(world[:3, :3]).as_quat(scalar_first=True)

    def set_joints(self, qpos, q):
        qpos[self.driven] = q
        for index, driver, ratio in self.followers:
            qpos[index] = q[driver] * ratio

    def static(self, flange, q):
        self.set_flange(flange)
        self.data.qpos[:] = self.model.qpos0
        self.set_joints(self.data.qpos, q)
        mujoco.mj_kinematics(self.model, self.data)

    def distances(self, geoms, target, cutoff=.13):
        return np.array([mujoco.mj_geomDistance(self.model, self.data, geom, target, cutoff, None) for geom in geoms])

    def fingertip_positions(self):
        return (self.data.xpos[self.tip_bodies] - self.object_pose[:3, 3]) @ self.object_pose[:3, :3]

    def contact_patch(self, finger):
        segments = np.zeros((len(self.distal[finger]), 6))
        distances = np.array([mujoco.mj_geomDistance(self.model, self.data, geom, self.cup_geom, .13, segments[i])
                              for i, geom in enumerate(self.distal[finger])])
        i = distances.argmin()
        return distances[i], (segments[i, 3:] - self.object_pose[:3, 3]) @ self.object_pose[:3, :3]

    def contacts(self):
        active = cup_contact_forces(self.model, self.data, self.cup_geom, self.finger)
        table = any(self.table_geom in contact.geom and any(int(g) in self.finger for g in contact.geom)
                    for contact in self.data.contact)
        return active, table


def seed_flange(joints):
    wrist = joints[0]
    z = joints[4] - wrist
    z /= np.linalg.norm(z)
    y = joints[1] - joints[7]
    y -= (y @ z) * z
    y /= np.linalg.norm(y)
    return transform(np.column_stack((np.cross(y, z), y, z)), wrist - .09 * z)


def patch_angle(thumb, opposed):
    a, b = thumb[:2], opposed[:2]
    return float(np.dot(a, b) / max(1e-6, np.linalg.norm(a) * np.linalg.norm(b)))


def refine_grasp(trial, flange, q, tips, opposed, grip_height, targets=None):
    base_r, base_p = flange[:3, :3].copy(), flange[:3, 3].copy()
    scale = np.array([.025] * 3 + [.45] * 3 + [.55] * 6)

    def state(x):
        change = x * scale
        rotation = Rotation.from_rotvec(change[3:6]).as_matrix() @ base_r
        return transform(rotation, base_p + change[:3]), q + change[6:]

    def residual(x):
        target, q_hand = state(x)
        trial.static(target, q_hand)
        distance = trial.distances(trial.hand_geoms, trial.cup_geom)
        floor = trial.distances(trial.hand_geoms, trial.table_geom)
        thumb_d, thumb = trial.contact_patch('thumb')
        other_d, other = trial.contact_patch(opposed)
        tip_weight = np.array([.26, .10, .10, .10, .10])
        tip_weight[FINGERS.index(opposed)] = .24
        return np.r_[
            ((trial.fingertip_positions() - tips) * tip_weight[:, None]).ravel(),
            6. * np.array([thumb_d + .00035, other_d + .00035]),
            4. * np.minimum(distance + .00065, 0),
            5. * np.minimum(floor - .003, 0),
            .08 * max(0., patch_angle(thumb, other) + .965),
            .45 * (np.array([thumb[2], other[2]]) - grip_height),
            [] if targets is None else (.65 * (np.stack((thumb, other)) - targets)).ravel(),
            x[3:6] * .0015,
        ]

    def jacobian(x):
        base = residual(x)
        eps = .0006
        return np.column_stack([(residual(x + gap * eps) - base) / eps for gap in np.eye(len(x))])

    bounds = (np.r_[[-2.] * 6, -q / scale[6:]], np.r_[[2.] * 6, (trial.robot.UPPER - q) / scale[6:]])
    result = least_squares(residual, np.zeros(12), jac=jacobian, bounds=bounds,
                           max_nfev=140, ftol=2e-6)
    return state(result.x)


def smooth(value):
    s = np.clip(value, 0., 1.)
    return s * s * (3 - 2 * s)


def simulate(trial, flange, initial, closed, opposed, record=False):
    model, data = trial.model, trial.data
    trial.set_flange(flange)
    mujoco.mj_resetData(model, data)
    trial.set_joints(data.qpos, initial)
    data.ctrl[1:] = initial
    mujoco.mj_forward(model, data)
    lift_start, hold_start = 1.35, 2.60
    end = hold_start + HOLD_SECONDS
    samples = []
    hold = []
    table_collision = False
    maximum_effort = np.zeros(6)
    close_contacts = {}
    close_opposed_steps = 0
    peak = -.1
    hold_relative = None
    total_steps = round(end / DT)
    frame_steps = round(1 / (FPS * DT))
    for step in range(total_steps):
        t = step * DT
        replay_qpos = data.qpos.copy() if record and step % frame_steps == 0 else None
        data.ctrl[0] = LIFT_HEIGHT * smooth((t - lift_start) / (hold_start - lift_start))
        data.ctrl[1:] = initial + (closed - initial) * smooth((t - .22) / .83)
        mujoco.mj_step(model, data)
        rotation = data.xmat[trial.cup].reshape(3, 3)
        position = data.xpos[trial.cup].copy()
        height = float((trial.hull_points @ rotation[2] + position[2]).min() - trial.table_z)
        tilt = float(np.rad2deg(np.arccos(np.clip(rotation[2, 2], -1, 1))))
        contact, hit_table = trial.contacts()
        maximum_effort = np.maximum(maximum_effort, np.abs(data.actuator_force[1:]))
        peak = max(height, peak)
        table_collision |= hit_table
        touching = contact.get('thumb', 0) > .015 and contact.get(opposed, 0) > .015
        if 1.10 <= t < lift_start:
            for name, force in contact.items():
                close_contacts[name] = max(force, close_contacts.get(name, 0.))
            close_opposed_steps += int(touching)
        phase = 'Close' if t < lift_start else 'Lift' if t < hold_start else 'Hold'
        if t >= hold_start:
            relative = position - data.xpos[trial.hand]
            if hold_relative is None:
                hold_relative = relative
            slip = float(np.linalg.norm(relative - hold_relative))
            hold.append((height, tilt, touching, slip))
        if replay_qpos is not None:
            samples.append((replay_qpos, t, height, phase, contact.get('thumb', 0.), contact.get(opposed, 0.)))

    held = np.asarray(hold)
    checks = {
        'opposed_contacts_before_lift': close_opposed_steps * DT >= .05,
        'hold_above_5_cm': bool(np.min(held[:, 0]) >= MIN_HEIGHT),
        'opposed_contacts_while_holding': bool(np.mean(held[:, 2]) >= .95),
        'hold_slip_below_10_mm': bool(np.max(held[:, 3]) <= .01),
        'cup_upright': bool(np.max(held[:, 1]) <= 20.),
        'hand_clear_of_table': not table_collision,
    }
    stats = dict(passed=all(checks.values()), checks=checks, minimum_hold_height_m=float(held[:, 0].min()),
                 peak_height_m=peak, hold_seconds=HOLD_SECONDS, max_hold_tilt_deg=float(held[:, 1].max()),
                 max_hold_slip_m=float(held[:, 3].max()), opposed_hold_fraction=float(held[:, 2].mean()),
                 opposed_contact_time_before_lift_s=close_opposed_steps * DT,
                 contact_force_during_close_n=close_contacts, max_actuator_effort_nm=maximum_effort.tolist())
    return stats, samples


def candidate_controls(trial, flange, q, opposed):
    other_geoms = [i for i, geom in enumerate(trial.hand_geoms) if trial.finger[geom] not in ('thumb', opposed)]

    def clearance(q_open, gap):
        trial.static(flange, q_open)
        cup = trial.distances(trial.hand_geoms, trial.cup_geom)
        table = trial.distances(trial.hand_geoms, trial.table_geom)
        patch = np.array([trial.contact_patch(name)[0] for name in ('thumb', opposed)])
        return np.r_[np.minimum(cup - gap, 0) * 8, np.minimum(table - .0015, 0) * 9,
                     2 * (patch - gap), .006 * (q_open - q)]

    def preload(q_command, depth):
        trial.static(flange, q_command)
        cup = trial.distances(trial.hand_geoms, trial.cup_geom)
        table = trial.distances(trial.hand_geoms, trial.table_geom)
        patch = np.array([trial.contact_patch(name)[0] for name in ('thumb', opposed)])
        return np.r_[8 * (patch + depth), 4 * np.minimum(cup[other_geoms] + .00035, 0),
                     8 * np.minimum(table - .0015, 0), .006 * (q_command - q)]

    # The thumb rotates as well as flexes. Solve for clearance at the same
    # flange pose instead of assuming that subtracting every joint angle opens it.
    for gap, delta in ((.0025, -.05), (.0012, .05), (.004, -.10)):
        initial = least_squares(clearance, np.clip(q + delta, 0, trial.robot.UPPER),
                                args=(gap,), bounds=(0, trial.robot.UPPER),
                                diff_step=3e-5, max_nfev=100).x
        trial.static(flange, initial)
        closest = trial.distances(trial.hand_geoms, trial.cup_geom).min()
        floor = trial.distances(trial.hand_geoms, trial.table_geom).min()
        print(f'  Open pose clearances: cup {closest * 1000:.2f} mm, table {floor * 1000:.2f} mm', flush=True)
        if closest < .0005 or floor < .0005:
            continue
        direction = q - initial
        for depth in (.0018, .0034, .0009, .005):
            push = np.clip(direction * 2., -.22, .22)
            command = least_squares(preload, np.clip(q + push, 0, trial.robot.UPPER),
                                    args=(depth,), bounds=(0, trial.robot.UPPER),
                                    diff_step=3e-5, max_nfev=80).x
            yield initial, command


def opposing_seeds(trial, retarget, tips, finger, height):
    index = FINGERS.index(finger)
    hand_model = trial.robot.HandModel()
    base_rotation = retarget.T_object_flange[:3, :3]
    horizontal = tips[0] - tips[index]
    horizontal[2] = 0.
    horizontal /= np.linalg.norm(horizontal)
    origin = np.array([0., 0., height])
    offsets = trial.hull_offsets - trial.hull_normals @ origin
    seeds = []
    for angle in (0., -25., 25., -50., 50.):
        axis = Rotation.from_euler('z', angle, degrees=True).apply(horizontal)
        radii = []
        for side in (axis, -axis):
            projection = trial.hull_normals @ side
            valid = projection > 1e-6
            radii.append(np.min(offsets[valid] / projection[valid]))
        targets = np.stack((origin + axis * radii[0], origin - axis * radii[1]))
        for rotate, bend, flex in product((.45, .85, 1.25, 1.65), (.01, .16, .32), (.15, .40, .65, .90)):
            q = retarget.joints.copy()
            q[0], q[1], q[index + 1] = rotate, bend, flex
            points = hand_model.tips(q)[[0, index]]
            chord = base_rotation @ (points[0] - points[1])
            span = np.linalg.norm(chord)
            if not sum(radii) - .008 < span < sum(radii) + .035:
                continue
            align, _ = Rotation.align_vectors(axis, chord / span)
            for roll in (-65., -35., -10., 15., 40., 65.):
                rotation = Rotation.from_rotvec(axis * np.deg2rad(roll)).as_matrix() @ align.as_matrix() @ base_rotation
                flange = transform(rotation, targets.mean(axis=0) - rotation @ points.mean(axis=0))
                trial.static(flange, q)
                cup = trial.distances(trial.hand_geoms, trial.cup_geom)
                floor = trial.distances(trial.hand_geoms, trial.table_geom)
                thumb_d, thumb = trial.contact_patch('thumb')
                finger_d, patch = trial.contact_patch(finger)
                score = (abs(thumb_d) + abs(finger_d) + 3 * max(0., -.0005 - cup.min())
                         + 3 * max(0., .002 - floor.min())
                         + np.linalg.norm(np.stack((thumb, patch)) - targets))
                seeds.append((score, flange, q, targets))
    seeds.sort(key=lambda seed: seed[0])
    return seeds[:4]


def solve(trial, tips, joints):
    robot = trial.robot
    first = seed_flange(joints)
    joint_seed = np.array([1.2, .20, .7, .7, .7, .7])
    profile = robot.radial_profile(trial.hull_points)
    retarget = robot.retarget(robot.HandModel(), tips, first, joint_seed, profile=profile, pose_weight=.015)
    print(f'URDF seed mean fingertip error {np.mean(retarget.tip_error) * 1000:.1f} mm', flush=True)

    best = None
    def candidates():
        for finger in ('middle', 'index', 'ring'):
            grasp_z = float(np.mean(tips[[0, FINGERS.index(finger)], 2]) + .018)
            for _, flange, q, targets in opposing_seeds(trial, retarget, tips, finger, grasp_z):
                yield finger, flange, q, grasp_z, targets
            for yaw in (0., -24., 24.):
                rotate = Rotation.from_euler('z', yaw, degrees=True).as_matrix()
                pose = retarget.T_object_flange
                flange = transform(rotate @ pose[:3, :3], rotate @ pose[:3, 3])
                yield finger, flange, retarget.joints, grasp_z, None

    for index, (finger, flange, q, grasp_z, targets) in enumerate(candidates()):
        flange, q = refine_grasp(trial, flange, q, tips, finger, grasp_z, targets)
        trial.static(flange, q)
        thumb_d, thumb = trial.contact_patch('thumb')
        other_d, other = trial.contact_patch(finger)
        angle = np.rad2deg(np.arccos(np.clip(patch_angle(thumb, other), -1, 1)))
        table_clearance = float(trial.distances(trial.hand_geoms, trial.table_geom).min())
        print(f'Fit {index + 1}, thumb + {finger}: distances {thumb_d * 1000:.1f}, {other_d * 1000:.1f} mm, '
              f'opposition {angle:.0f} deg, table clearance {table_clearance * 1000:.1f} mm', flush=True)
        if max(abs(thumb_d), abs(other_d)) > .003 or angle < 120 or table_clearance < .0005:
            continue
        for initial, command in candidate_controls(trial, flange, q, finger):
            result, _ = simulate(trial, flange, initial, command, finger)
            print(f'  Lift peak {result["peak_height_m"] * 100:.1f} cm, '
                  f'hold minimum {result["minimum_hold_height_m"] * 100:.1f} cm, '
                  f'opposed close {result["opposed_contact_time_before_lift_s"]:.2f} s, '
                  f'passed {result["passed"]}', flush=True)
            quality = (result['passed'], result['minimum_hold_height_m'], result['opposed_hold_fraction'])
            if best is None or quality > best[0]:
                best = quality, flange, q, initial, command, finger
            if result['passed']:
                return best[1:]
    if best is None:
        raise ValueError('No RH56F1 pose satisfied static opposition, clearance and opening checks')
    return best[1:]


def run(cli):
    robot = load_retarget(cli.retarget_module)
    obj, tips, joints, source_mesh = read_grasp(cli.grasp, cli.frame)
    mesh = trimesh.load(cli.mesh or source_mesh, force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError('Expected a nonempty metric cup mesh')
    cli.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cli.output_dir.parent) as staging:
        bundle = Path(staging) / 'result'
        bundle.mkdir()
        xml, hand, hull, table_z = build_scene(bundle, cli.robot_mjcf, mesh, obj, robot, cli.cup_mass, cli.friction)
        scene = bundle / 'scene.xml'
        ET.ElementTree(xml).write(scene, encoding='utf-8')
        trial = LiftTrial(mujoco.MjModel.from_xml_path(str(scene)), robot, obj, hull, table_z)

        # Confirm that the MJCF and the existing URDF module use the same hand frame and couplings.
        sample_q = .37 * robot.UPPER
        trial.static(np.eye(4), sample_q)
        fk_error = float(np.linalg.norm(trial.fingertip_positions() - robot.HandModel().tips(sample_q), axis=-1).max())
        if fk_error >= .001:
            raise ValueError(f'RH56F1 MJCF and URDF tip positions disagree by {fk_error * 1000:.2f} mm')

        flange, pose, initial, command, finger = solve(trial, tips, joints)
        trial.static(flange, pose)
        tip_errors = np.linalg.norm(trial.fingertip_positions() - tips, axis=-1)
        thumb_d, thumb = trial.contact_patch('thumb')
        other_d, other = trial.contact_patch(finger)
        table_clearance = float(trial.distances(trial.hand_geoms, trial.table_geom).min())
        trial.static(flange, initial)
        opening_clearance = float(trial.distances(trial.hand_geoms, trial.cup_geom).min())
        trial.static(flange, pose)
        np.savez_compressed(bundle / f'retarget_frame{cli.frame}.npz', source_frame=cli.frame, T_object_flange=flange,
                            T_scene_object=obj, T_scene_flange=obj @ flange, q_hand_rad=pose,
                            q_open_rad=initial, q_close_command_rad=command, joint_names=robot.ACTUATED,
                            human_tips_object_m=tips, robot_tips_object_m=trial.fingertip_positions(),
                            tip_error_m=tip_errors, contact_fingers=('thumb', finger),
                            contact_points_object_m=np.stack((thumb, other)))
        stats, samples = simulate(trial, flange, initial, command, finger, record=True)
        stats.update(grasp_frame=cli.frame, source=str(cli.grasp.resolve()),
                     model='RH56F1 with actuated wrist height rig', cup_mass_kg=cli.cup_mass,
                     sliding_friction=cli.friction, contact_fingers=['thumb', finger],
                     static_contact_distance_m=[float(thumb_d), float(other_d)],
                     static_opposition_deg=float(np.rad2deg(np.arccos(np.clip(patch_angle(thumb, other), -1, 1)))),
                     static_mean_tip_error_m=float(tip_errors.mean()), static_table_clearance_m=table_clearance,
                     opening_cup_clearance_m=opening_clearance, urdf_mjcf_tip_error_m=fk_error,
                     requested_wrist_lift_m=LIFT_HEIGHT, physics_timestep_s=DT)
        hand.set('pos', text(trial.model.body_pos[trial.hand]))
        hand.set('quat', text(trial.model.body_quat[trial.hand]))
        initial_state = trial.model.qpos0.copy()
        trial.set_joints(initial_state, initial)
        keys = ET.SubElement(xml, 'keyframe')
        ET.SubElement(keys, 'key', name='open_start', qpos=text(initial_state), ctrl=text(np.r_[0, initial]))
        ET.indent(xml)
        ET.ElementTree(xml).write(scene, encoding='utf-8')
        qpos, time, height, phase, thumb_force, other_force = zip(*samples)
        np.savez_compressed(bundle / 'lift_replay.npz', qpos=qpos, fps=FPS, time_s=time, cup_height_m=height,
                            phase=phase, thumb_force_n=thumb_force, opposing_force_n=other_force)
        (bundle / 'lift_metrics.json').write_text(json.dumps(stats, indent=2) + '\n')
        shutil.copyfile(Path(__file__).resolve().parents[1] / 'visualize/replay_rh56f1_lift.py', bundle / 'replay.py')
        bundle.rename(cli.output_dir)
    print(f'Results at {cli.output_dir} | hold {stats["minimum_hold_height_m"] * 100:.2f} cm | passed {stats["passed"]}')
    return stats['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grasp', type=Path, required=True, help='right_hand_grasp.npz from select_grasp')
    parser.add_argument('--mesh', type=Path, help='Override the cup mesh path in the archive')
    parser.add_argument('--frame', type=int, default=49)
    parser.add_argument('--robot-mjcf', type=Path, default=DEFAULT_MJCF, help='Override the bundled RH56F1 MJCF')
    parser.add_argument('--retarget-module', type=Path, help='Override the bundled RH56F1 retarget module')
    parser.add_argument('--cup-mass', type=float, default=.06, help='Simulated cup mass in kg')
    parser.add_argument('--friction', type=float, default=.65, help='Simulated cup and finger sliding friction')
    parser.add_argument('--output-dir', type=Path, required=True)
    cli = parser.parse_args()
    if not np.isfinite([cli.cup_mass, cli.friction]).all() or cli.cup_mass <= 0 or cli.friction <= 0:
        parser.error('Cup mass and friction must be finite and positive')
    if cli.output_dir.exists():
        parser.error('Use a new output directory')
    try:
        passed = run(cli)
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
