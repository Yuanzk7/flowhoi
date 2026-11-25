"""Plan and simulate an xArm7 approach, RH56F1 grasp, and contact-driven cup lift."""

import argparse
import json
import shutil
import subprocess
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from sample.grasp_interface import preshape_at
from sample.rh56f1_retarget import preshape_joints
from sample.validate_rh56f1 import (
    FPS, HOLD_SECONDS, LIFT_HEIGHT, MIN_HEIGHT, cup_contact_forces,
    hand_fingers, joint_name, smooth, text, transform,
)


ARM_NAMES = tuple(f'joint{i}' for i in range(1, 8))
DEFAULT_XARM_ROOT = Path(__file__).resolve().parents[1] / 'third_party/robot_models/xarm7'
HOME = (0., -.45, 0., 1.1, 0., .9, 0.)
APPROACH_LENGTH = .08
COLLISION_MARGIN = .0005
AZIMUTHS = (0, 20, -20, 35, -35, 50, -50)


def expand_urdf(root, xacro, work):
    package = work / 'xarm_description'
    description = root / 'xarm_description'
    for name in ('urdf', 'config'):
        shutil.copytree(description / name, package / name)
    for path in (package / 'urdf').rglob('*.xacro'):
        content = path.read_text().replace('$(find xarm_description)', str(package))
        path.write_text(content.replace('$(find xarm_controller)', str(root / 'xarm_controller')))
    result = subprocess.run([xacro, str(package / 'urdf/xarm_device.urdf.xacro'), 'dof:=7',
                             'robot_type:=xarm', 'limited:=true', 'mesh_suffix:=stl',
                             'add_gripper:=false'], check=True, capture_output=True, text=True)
    path = work / 'xarm7.urdf'
    path.write_text(result.stdout.replace('package://xarm_description', str(description)))
    return path


def arm_urdf_path(root, urdf, xacro, work):
    if urdf is not None:
        return urdf.resolve()
    expanded = root / 'xarm7.urdf'
    return expanded if expanded.is_file() else expand_urdf(root, xacro, work)


def add_arm_model_arguments(parser):
    parser.add_argument('--xarm-root', type=Path, default=DEFAULT_XARM_ROOT,
                        help='Override the bundled xArm7 model directory with a model bundle or xArm ROS 2 checkout')
    parser.add_argument('--xacro', default='xacro', help='Xacro executable for an unexpanded xArm ROS 2 checkout')
    parser.add_argument('--arm-urdf', type=Path, help='Override the expanded xArm7 URDF')


def add_flowhoi_motion_arguments(parser):
    parser.add_argument('--approach', choices=('flowhoi', 'flange'), default='flowhoi',
                        help='flowhoi: retreat against the FlowHOI palm travel direction first, with the flange '
                             'axis as fallback. flange: only the flange axis, as before')
    parser.add_argument('--preshape', choices=('flowhoi', 'open'), default='flowhoi',
                        help='flowhoi: open the fingers by the FlowHOI aperture profile against the remaining '
                             'distance, falling back to the open hand. open: hold the open hand, as before')
    parser.add_argument('--object-symmetry', choices=('none', 'z'), default='none',
                        help='z: the object is symmetric about its +z axis (a plain cup), so the grasp and its '
                             'approach may be rotated about that axis to suit the arm. none: use the grasp as fitted')


def numbers(value):
    return np.array([float(item) for item in value.split()])


def origin(element):
    node = element.find('origin')
    if node is None:
        return np.eye(4)
    rotation = Rotation.from_euler('xyz', numbers(node.get('rpy', '0 0 0'))).as_matrix()
    return transform(rotation, numbers(node.get('xyz', '0 0 0')))


def mjcf_pose(matrix):
    return dict(pos=text(matrix[:3, 3]),
                quat=text(Rotation.from_matrix(matrix[:3, :3]).as_quat(scalar_first=True)))


def hand_mount_in_flange(hand, flange_pose):
    hand_pose = transform(Rotation.from_quat(numbers(hand.get('quat')), scalar_first=True).as_matrix(),
                          numbers(hand.get('pos')))
    return np.linalg.solve(flange_pose, hand_pose)


def load_grasp(directory, frame):
    source = json.loads((directory / 'lift_metrics.json').read_text())
    if not source['passed'] or source['grasp_frame'] != frame:
        raise ValueError('Use the frame from a passing RH56F1 lift bundle')
    with np.load(directory / f'retarget_frame{frame}.npz', allow_pickle=False) as data:
        grasp = {key: data[key] for key in data.files}
    if int(grasp['source_frame']) != frame or tuple(grasp['contact_fingers']) != tuple(source['contact_fingers']):
        raise ValueError('The retarget archive and lift metrics refer to different grasps')
    return grasp, source


def configure_arm(scene, output, path, root, base, hand):
    robot = ET.parse(path).getroot()
    links = {link.get('name'): link for link in robot.findall('link')}
    joints = {joint.get('name'): joint for joint in robot.findall('joint')}
    bodies = {}
    assets = scene.find('asset')
    world = scene.find('worldbody')
    actuators = scene.find('actuator')
    copied = {}
    arm_assets = output / 'arm_meshes'
    arm_assets.mkdir()

    def geometry(body, element, name, collision):
        mesh = element.find('geometry/mesh')
        filename = mesh.get('filename')
        if filename.startswith('package://xarm_description/'):
            source = root / 'xarm_description' / filename.removeprefix('package://xarm_description/')
        else:
            source = path.parent / filename
        scale = mesh.get('scale', '1 1 1')
        if (source, scale) not in copied:
            target = arm_assets / f'{len(copied):02d}_{source.name}'
            shutil.copyfile(source, target)
            copied[source, scale] = target.stem
            ET.SubElement(assets, 'mesh', name=target.stem, file=f'arm_meshes/{target.name}', scale=scale)
        geom = dict(name=name, type='mesh', mesh=copied[source, scale], mass='0', **mjcf_pose(origin(element)))
        if collision:
            geom.update(contype='8', conaffinity='9', group='3')
        else:
            geom.update(rgba='.80 .82 .86 1', group='1')
        ET.SubElement(body, 'geom', **geom)

    bodies['link_base'] = ET.SubElement(world, 'body', name='link_base', **mjcf_pose(base))
    for name in ARM_NAMES:
        joint = joints[name]
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        body = ET.SubElement(bodies[parent], 'body', name=child, gravcomp='1', **mjcf_pose(origin(joint)))
        bodies[child] = body
        inertial = links[child].find('inertial')
        pose = origin(inertial)
        mass = inertial.find('mass').get('value')
        inertia = inertial.find('inertia')
        xx, yy, zz, xy, xz, yz = (float(inertia.get('i' + entry)) for entry in ('xx', 'yy', 'zz', 'xy', 'xz', 'yz'))
        matrix = pose[:3, :3] @ np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) @ pose[:3, :3].T
        tensor = matrix[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]
        ET.SubElement(body, 'inertial', pos=text(pose[:3, 3]), mass=mass, fullinertia=text(tensor))
        limit = joint.find('limit')
        bounds = f'{limit.get("lower")} {limit.get("upper")}'
        effort = float(limit.get('effort'))
        i = ARM_NAMES.index(name)
        ET.SubElement(body, 'joint', name=name, axis=joint.find('axis').get('xyz'), range=bounds,
                      damping=joint.find('dynamics').get('damping'), armature='.012')
        kp = (1600, 1900, 1300, 1600, 1000, 1100, 800)[i]
        kv = (64, 72, 42, 53, 30, 26, 16)[i]
        ET.SubElement(actuators, 'position', name=f'xarm_{name}_servo', joint=name,
                      ctrlrange=bounds, forcerange=text((-effort, effort)), kp=str(kp), kv=str(kv))

    for name, body in bodies.items():
        for tag in ('visual', 'collision'):
            for i, element in enumerate(links[name].findall(tag)):
                geometry(body, element, f'xarm_{name}_{tag}_{i}', tag == 'collision')
    flange_body = bodies['link7']
    flange_body.append(hand)
    mount = numbers(hand.get('pos'))
    if np.linalg.norm(mount) > .001:
        ET.SubElement(flange_body, 'geom', name='rh56f1_mount', type='capsule', size='.027',
                      fromto=text(np.r_[0., 0., 0., mount]), mass='0', contype='8',
                      conaffinity='9', rgba='.26 .29 .34 1')

    srdf_path = root / 'xarm7.srdf'
    if not srdf_path.is_file():
        srdf_path = root / 'xarm_moveit_config/srdf/_xarm7_macro.srdf.xacro'
    srdf = ET.parse(srdf_path)
    contact = ET.SubElement(scene, 'contact')
    excluded = set()
    for node in srdf.iter('disable_collisions'):
        pair = tuple(node.get(f'link{i}').replace('${prefix}', '') for i in (1, 2))
        if all(name in bodies for name in pair) and frozenset(pair) not in excluded:
            ET.SubElement(contact, 'exclude', body1=pair[0], body2=pair[1])
            excluded.add(frozenset(pair))
    license_path = root / 'LICENSE'
    if license_path.is_file():
        shutil.copyfile(license_path, output / 'XARM_ROS2_LICENSE.txt')


def build_scene(directory, source, grasp, urdf, root, base_xy=None, *, measured=None):
    scene = ET.parse(source / 'scene.xml').getroot()
    scene.set('model', 'xArm7 + RH56F1 cup lift')
    for asset in scene.findall('asset/mesh'):
        file = Path(asset.get('file'))
        (directory / file.parent).mkdir(exist_ok=True, parents=True)
        shutil.copyfile(source / file, directory / file)
    hand_license = source / 'RH56F1_LICENSE.txt'
    if hand_license.is_file():
        shutil.copyfile(hand_license, directory / hand_license.name)

    world = scene.find('worldbody')
    carriage = world.find("body[@name='lift_carriage']")
    hand = carriage.find("body[@name='right_hand_base']")
    if measured is None:
        target = grasp['T_scene_flange']
        mount = hand_mount_in_flange(hand, target)
    else:
        mount = measured['T_flange_hand']
    hand.attrib.update(mjcf_pose(mount))
    world.remove(carriage)
    actuators = scene.find('actuator')
    actuators.remove(actuators.find("position[@name='lift_servo']"))
    scene.remove(scene.find('keyframe'))

    table = world.find("geom[@name='table']")
    cup = world.find("body[@name='cup']")
    for geom in (table, cup.find("geom[@name='cup_contact']")):
        geom.set('conaffinity', str(int(geom.get('conaffinity')) | 8))
    if measured is not None:
        table_pose = measured['T_base_table'].copy()
        half_size = measured['table_size_m'] / 2
        table_pose[:3, 3] -= half_size[2] * table_pose[:3, 2]
        table.attrib.update(size=text(half_size), **mjcf_pose(table_pose))
        cup.attrib.update(mjcf_pose(measured['T_base_object']))
        base = np.eye(4)
        configure_arm(scene, directory, urdf, root, base, hand)
        return scene, base

    table_pos, table_size = numbers(table.get('pos')), numbers(table.get('size'))
    table_z = table_pos[2] + table_size[2]
    if base_xy is None:
        base_xy = table_pos[:2] + [table_size[0] + .13, -.19]
    direction = target[:2, 3] - base_xy
    yaw = np.arctan2(direction[1], direction[0])
    base = transform(Rotation.from_euler('z', yaw).as_matrix(), np.r_[base_xy, table_z])
    configure_arm(scene, directory, urdf, root, base, hand)
    center = table_pos + .5 * (base[:3, 3] - table_pos) + [0, 0, .26]
    scene.find('statistic').attrib.update(center=text(center), extent='.85')
    world.find('light').set('castshadow', 'true')
    decor = dict(contype='0', conaffinity='0', group='1')
    ET.SubElement(world, 'geom', name='arm_mounting_pedestal', type='box', size='.076 .092 .23',
                  pos=text(np.r_[base_xy, table_z - .23]), rgba='.34 .39 .47 1', **decor)
    for x in (-.37, .37):
        for y in (-.37, .37):
            position = table_pos + [x, y, -.23]
            ET.SubElement(world, 'geom', type='box', size='.025 .025 .207', pos=text(position),
                          rgba='.29 .35 .42 1', **decor)
    return scene, base


class ArmTrial:
    def __init__(self, model, grasp, xml):
        self.model = model
        self.data = mujoco.MjData(model)
        self.flange = model.body('link7').id
        self.hand = model.body('right_hand_base').id
        self.arm_ids = np.array([model.joint(name).id for name in ARM_NAMES])
        self.arm_q = model.jnt_qposadr[self.arm_ids]
        self.arm_dof = model.jnt_dofadr[self.arm_ids]
        self.arm_ctrl = np.array([model.actuator(f'xarm_{name}_servo').id for name in ARM_NAMES])
        self.bounds = model.jnt_range[self.arm_ids].T
        hand_names = [joint_name(name) for name in grasp['joint_names']]
        self.hand_ctrl = np.array([model.actuator(f'{name}_servo').id for name in hand_names])
        self.hand_q = np.array([model.joint(name).qposadr.item() for name in hand_names])
        self.fingers = hand_fingers(model)
        self.arm_geoms = set(np.flatnonzero(model.geom_contype == 8).tolist())
        self.cup_geom = model.geom('cup_contact').id
        self.table = model.geom('table').id
        self.cup = model.body('cup').id
        self.cup_q = model.joint('cup_free').qposadr.item()
        mesh = model.geom_dataid[self.cup_geom]
        start = model.mesh_vertadr[mesh]
        self.cup_points = model.mesh_vert[start:start + model.mesh_vertnum[mesh]].astype(float)
        self.table_z = model.geom_pos[self.table, 2] + model.geom_size[self.table, 2]
        self.open_angles = grasp['q_open_rad']
        self.close_angles = grasp['q_close_command_rad']
        self.followers = []
        for eq in xml.findall('equality/joint'):
            follower = model.joint(eq.get('joint1')).qposadr.item()
            driver = model.joint(eq.get('joint2')).qposadr.item()
            self.followers.append((follower, driver, numbers(eq.get('polycoef'))[1]))
        self.set_hand(self.open_angles)
        self.open_qpos = self.data.qpos.copy()

        excluded = {frozenset((model.body(node.get('body1')).id, model.body(node.get('body2')).id))
                    for node in xml.findall('contact/exclude')}
        self.self_pairs = []
        for first, second in combinations(sorted(self.arm_geoms | self.fingers.keys()), 2):
            if first in self.fingers and second in self.fingers:
                continue
            a, b = model.body_weldid[model.geom_bodyid[[first, second]]]
            if (a == b or frozenset((a, b)) in excluded
                    or model.body_weldid[model.body_parentid[a]] == b
                    or model.body_weldid[model.body_parentid[b]] == a):
                continue
            self.self_pairs.append((first, second))
        self.self_pairs = set(self.self_pairs)
        self.table_pairs = [(g, self.table) for g in sorted(self.arm_geoms | self.fingers.keys())]
        self.base_pairs = (sorted(self.self_pairs) + self.table_pairs
                           + [(g, self.cup_geom) for g in sorted(self.arm_geoms)])
        self.approach_pairs = self.base_pairs + [(g, self.cup_geom) for g in self.fingers]

    def set_hand(self, angles):
        self.data.qpos[self.hand_q] = angles
        for follower, driver, ratio in self.followers:
            self.data.qpos[follower] = ratio * self.data.qpos[driver]

    def set_arm(self, q):
        self.data.qpos[self.arm_q] = q
        mujoco.mj_kinematics(self.model, self.data)

    def flange_pose(self):
        return self.data.xpos[self.flange], self.data.xmat[self.flange].reshape(3, 3)

    def solve_ik(self, goal, seed):
        goal_r = goal[:3, :3]
        low, high = self.bounds
        initial = np.clip(seed, low + .001, high - .001)

        def error(q):
            self.set_arm(q)
            position, rotation = self.flange_pose()
            angle = Rotation.from_matrix(goal_r @ rotation.T).as_rotvec()
            return np.r_[position - goal[:3, 3], .15 * angle, .00025 * (q - initial)]

        result = least_squares(error, initial, bounds=(low + .001, high - .001),
                               max_nfev=85, xtol=5e-8, ftol=1e-7, gtol=1e-9)
        residual = error(result.x)
        if np.linalg.norm(residual[:3]) > .00025 or np.linalg.norm(residual[3:6]) > .0006:
            return None
        return result.x

    def linear_ik(self, start, goal, seed):
        points = [seed]
        count = int(np.ceil(np.linalg.norm(goal[:3, 3] - start[:3, 3]) / .005))
        for fraction in np.linspace(0., 1., count + 1)[1:]:
            target = start + (goal - start) * fraction
            q = self.solve_ik(target, points[-1])
            if q is None:
                return None
            points.append(q)
        return CubicSpline(np.linspace(0., 1., len(points)), points)

    def clearance(self, path, approach, samples, *, pairs=None, apply_state=None, cutoff=.10):
        if pairs is None:
            pairs = self.approach_pairs if approach else self.base_pairs
        closest = cutoff, None
        positions = np.linspace(0., 1., samples)
        saved = self.data.qpos.copy() if apply_state is not None else None
        try:
            for fraction, q in zip(positions, path(positions)):
                if np.any(q < self.bounds[0]) or np.any(q > self.bounds[1]):
                    return -np.inf, 'joint limits'
                if apply_state is not None:
                    apply_state(fraction)
                self.set_arm(q)
                for first, second in pairs:
                    distance = mujoco.mj_geomDistance(self.model, self.data, first, second, cutoff, None)
                    if distance < closest[0]:
                        closest = distance, (first, second)
        finally:
            if saved is not None:
                self.data.qpos[:] = saved
                mujoco.mj_kinematics(self.model, self.data)
        distance, pair = closest
        return distance, None if pair is None else [self.model.geom(i).name for i in pair]


def flowhoi_approach(grasp, frame_rotation_source):
    """FlowHOI approach direction from a retarget archive, in the frame of ``frame_rotation_source``."""
    if 'approach_valid' not in grasp or not bool(grasp['approach_valid']):
        return None
    return np.asarray(frame_rotation_source)[:3, :3] @ grasp['approach_dir_object']


def flowhoi_preshape(grasp):
    """Hand angles as a function of remaining flange distance, or None for an older archive."""
    if 'preshape' not in grasp:
        return None
    remaining, amount, q_open = grasp['preshape_remaining_m'], grasp['preshape'], grasp['q_open_rad']
    return lambda distance: preshape_joints(q_open, preshape_at(distance, remaining, amount))


def about_object_axis(degrees):
    """Rotation about the object +z axis through its origin, as a 4x4 transform."""
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('z', degrees, degrees=True).as_matrix()
    return pose


def symmetry_yaws(symmetry):
    """Grasp rotations about the object axis to try, nearest to the fitted grasp first."""
    if symmetry == 'none':
        return (0.,)
    return (0., 30., -30., 60., -60., 90., -90., 120., -120., 150., -150., 180.)


def plan_over_symmetry(attempt, symmetry):
    """Call ``attempt(yaw)`` over the allowed object-axis rotations until one plans.

    For an object symmetric about its +z axis (a plain cup on a table), rotating the whole
    grasp, approach included, leaves the grasp physically the same while letting a fixed
    arm reach it from its own side.
    """
    failure = None
    for yaw in symmetry_yaws(symmetry):
        try:
            paths, targets, clearances, choice = attempt(yaw)
        except PlanningFailure as error:
            if failure is None or (error.grasp_ik_found and not failure.grasp_ik_found):
                failure = error
            if symmetry != 'none':
                print(f'Object-axis rotation {yaw:+.0f} deg: {error}', flush=True)
            continue
        choice['grasp_axis_rotation_deg'] = yaw
        return paths, targets, clearances, choice
    raise failure


class PlanningFailure(ValueError):
    def __init__(self, grasp_ik_found, best_clearances):
        super().__init__('No collision-free start, pregrasp, grasp and lift path at the required margin')
        self.grasp_ik_found = grasp_ik_found
        self.best_clearances = best_clearances


def flange_position(trial, q):
    """Flange position for arm angles ``q`` without disturbing the current state."""
    saved = trial.data.qpos.copy()
    trial.set_arm(q)
    position = trial.flange_pose()[0].copy()
    trial.data.qpos[:] = saved
    mujoco.mj_kinematics(trial.model, trial.data)
    return position


def preshape_states(trial, preshape, distance_to_pregrasp):
    """Hand-state setters for path fractions of the transit and approach, or (None, None).

    The remaining distance is approximated as the straight approach length plus the
    straight-line distance from the transit start to the pregrasp.
    """
    if preshape is None:
        return None, None
    return (lambda u: trial.set_hand(preshape(APPROACH_LENGTH + (1 - u) * distance_to_pregrasp)),
            lambda u: trial.set_hand(preshape(APPROACH_LENGTH * (1 - u))))


def pregrasp_candidates(goal, vertical, approach_dir=None):
    """Pregrasp poses ordered by preference, as (pose, source, degrees from the FlowHOI approach).

    With a FlowHOI approach direction the hand retreats against its direction of travel,
    then against nearby directions rotated about the vertical; the flange-axis retreats used
    before are kept as fallbacks.
    """
    reference = None if approach_dir is None else -np.asarray(approach_dir) / np.linalg.norm(approach_dir)
    directions = []
    if reference is not None:
        directions += [('flowhoi', Rotation.from_rotvec(np.deg2rad(a) * vertical).apply(reference)) for a in AZIMUTHS]
    directions += [('flange_axis', Rotation.from_rotvec(np.deg2rad(a) * vertical).apply(-goal[:3, 2])) for a in AZIMUTHS]
    candidates = []
    for source, backward in directions:
        pregrasp = goal.copy()
        pregrasp[:3, 3] += APPROACH_LENGTH * backward
        deviation = None if reference is None else float(np.rad2deg(np.arccos(np.clip(backward @ reference, -1, 1))))
        candidates.append((pregrasp, source, deviation))
    return candidates


def plan(trial, goal, home, *, margin=COLLISION_MARGIN, up=(0., 0., 1.), check_path=None,
         approach_dir=None, preshape=None):
    """Plan home -> pregrasp -> grasp -> lift.

    ``approach_dir`` is the FlowHOI palm travel direction in the planning frame. ``preshape``
    maps the remaining flange distance to the goal onto hand joint angles; when given, transit
    and approach clearances are checked with that hand state, falling back to the open hand.
    Returns paths, targets, clearances and a dict describing the choices made.
    """
    best = {}
    vertical = np.asarray(up)

    def evaluate(phase, path, samples, state=None):
        if check_path is None:
            result = trial.clearance(path, phase != 'lift', samples, apply_state=state)
        else:
            result = check_path(phase, path, samples, state)
        if phase not in best or result[0] > best[phase][0]:
            best[phase] = result
        return result

    home_flange = flange_position(trial, home)

    lift = goal.copy()
    lift[:3, 3] += LIFT_HEIGHT * vertical
    pregrasps = pregrasp_candidates(goal, vertical, approach_dir)
    modes = ('flowhoi', 'open') if preshape is not None else ('open',)
    seeds = [home, np.zeros(7), [0, .7, 0, 1.7, 0, -.45, 0],
             [0, -.1, 0, 1.2, 0, 1.5, 1.5], [0, .4, 0, .8, 0, .5, -1.5]]
    seeds.extend(np.random.default_rng(19).uniform(trial.bounds[0], trial.bounds[1], (22, 7)))
    attempted = []
    for seed in seeds:
        grasp_q = trial.solve_ik(goal, seed)
        if grasp_q is None or any(np.linalg.norm(grasp_q - q) < .12 for q in attempted):
            continue
        attempted.append(grasp_q)
        if check_path is not None:
            def stationary(u):
                return np.broadcast_to(grasp_q, (*np.shape(u), grasp_q.size))

            if evaluate('close', stationary, 33)[0] < margin:
                continue
        lift_path = trial.linear_ik(goal, lift, grasp_q)
        if lift_path is None:
            continue
        lift_clearance = evaluate('lift', lift_path, 65)
        if lift_clearance[0] < margin:
            continue
        for index, (pregrasp, source, deviation) in enumerate(pregrasps):
            retreat = trial.linear_ik(goal, pregrasp, grasp_q)
            if retreat is None:
                continue

            def approach(u):
                return retreat(1 - np.asarray(u))

            start_q = approach(0.)

            def transit(u):
                return home + np.asarray(u)[..., None] * (start_q - home)

            transit_state, approach_state = preshape_states(trial, preshape,
                                                            np.linalg.norm(pregrasp[:3, 3] - home_flange))
            for mode in modes:
                states = (transit_state, approach_state) if mode == 'flowhoi' else (None, None)
                approach_clearance = evaluate('approach', approach, 65, states[1])
                if approach_clearance[0] < margin:
                    continue
                transit_clearance = evaluate('transit', transit, 97, states[0])
                clearances = (transit_clearance, approach_clearance, lift_clearance)
                print(f'Pregrasp {index} ({source}, hand {mode}) clearance (transit, approach, lift): '
                      + ', '.join(f'{d * 1000:.1f} mm {pair}' for d, pair in clearances), flush=True)
                if transit_clearance[0] >= margin:
                    info = dict(approach_source=source, approach_deviation_deg=deviation,
                                pregrasp_index=index, hand_mode=mode)
                    return (transit, approach, lift_path), (pregrasp, goal, lift), clearances, info
    raise PlanningFailure(bool(attempted), best)


def simulate(trial, paths, targets, home, opposed, preshape=None):
    """Run the planned motion. ``preshape`` maps flange distance to the goal onto hand angles
    until closing starts; without it the hand stays open as before."""
    model, data = trial.model, trial.data
    transit, approach, lift = paths
    _, goal, _ = targets
    dt = model.opt.timestep
    mujoco.mj_resetData(model, data)
    data.qpos[:] = trial.open_qpos
    data.qpos[trial.arm_q] = home
    data.ctrl[trial.arm_ctrl] = home
    data.ctrl[trial.hand_ctrl] = trial.open_angles
    mujoco.mj_forward(model, data)

    def approach_hand():
        if preshape is None:
            return trial.open_angles
        return preshape(float(np.linalg.norm(data.xpos[trial.flange] - goal[:3, 3])))

    if preshape is not None:
        trial.set_hand(approach_hand())
        data.ctrl[trial.hand_ctrl] = approach_hand()
        mujoco.mj_forward(model, data)
    initial_cup = data.xpos[trial.cup].copy()
    transit_duration = np.ceil(FPS * max(2.4, 2 * np.max(np.abs(transit(1.) - home)))) / FPS
    transit_step, approach_step, close_step, lift_step, hold_step = np.cumsum([
        round(duration / dt) for duration in (.40, transit_duration, 1.65, 1.35, 1.25)
    ])
    times = dict(zip(('Transit', 'Approach', 'Close', 'Lift', 'Hold'),
                     (step * dt for step in (transit_step, approach_step, close_step, lift_step, hold_step))))
    hold, frames = [], []
    close_opposed = 0
    unexpected = set()
    preclose_displacement = 0.
    grasp_position_error = grasp_rotation_error = 0.
    max_joint_speed = np.zeros(7)
    max_joint_error = np.zeros(7)
    max_limit_violation = 0.
    held_reference = None
    peak = 0.
    steps = hold_step + round(HOLD_SECONDS / dt)
    every = round(1 / (FPS * dt))
    bounds = trial.bounds

    for step in range(steps):
        t = step * dt
        frame = data.qpos.copy() if step % every == 0 else None
        phase = 'Home'
        arm_goal, position_target = home, None
        if step >= hold_step:
            phase, arm_goal = 'Hold', lift(1.)
        elif step >= lift_step:
            phase = 'Lift'
            arm_goal = lift(smooth((t - times['Lift']) / (times['Hold'] - times['Lift'])))
        elif step >= close_step:
            phase, arm_goal = 'Close', approach(1.)
        elif step >= approach_step:
            phase = 'Approach'
            u = smooth((t - times['Approach']) / (times['Close'] - times['Approach'] - .25))
            arm_goal = approach(u)
            position_target = targets[0][:3, 3] + u * (goal[:3, 3] - targets[0][:3, 3])
        elif step >= transit_step:
            phase = 'Transit'
            arm_goal = transit(smooth((t - times['Transit']) / (times['Approach'] - times['Transit'])))

        data.ctrl[trial.arm_ctrl] = arm_goal
        if step < close_step:
            hand_goal = approach_hand()
        else:
            fraction = smooth((t - times['Close'] - .22) / .83)
            hand_goal = trial.open_angles + fraction * (trial.close_angles - trial.open_angles)
        data.ctrl[trial.hand_ctrl] = hand_goal
        mujoco.mj_step(model, data)
        max_joint_speed = np.maximum(max_joint_speed, np.abs(data.qvel[trial.arm_dof]))
        arm_q = data.qpos[trial.arm_q]
        max_joint_error = np.maximum(max_joint_error, np.abs(arm_q - arm_goal))
        max_limit_violation = max(max_limit_violation, float(np.max(np.maximum(bounds[0] - arm_q, arm_q - bounds[1]))))

        forces = cup_contact_forces(model, data, trial.cup_geom, trial.fingers)
        touching = forces.get('thumb', 0.) > .015 and forces.get(opposed, 0.) > .015
        for contact in data.contact:
            a, b = sorted(map(int, contact.geom))
            pair = {a, b}
            is_robot = a in trial.fingers or b in trial.fingers or bool(pair & trial.arm_geoms)
            if ((trial.table in pair and is_robot) or
                (trial.cup_geom in pair and bool(pair & trial.arm_geoms)) or
                (trial.cup_geom in pair and step < close_step and is_robot) or
                (a, b) in trial.self_pairs):
                unexpected.add((model.geom(a).name, model.geom(b).name))

        position = data.xpos[trial.cup]
        rotation = data.xmat[trial.cup].reshape(3, 3)
        geom_z = data.geom_xmat[trial.cup_geom].reshape(3, 3)[2]
        height = float(np.min(trial.cup_points @ geom_z + data.geom_xpos[trial.cup_geom, 2]) - trial.table_z)
        peak = max(height, peak)
        if step < close_step:
            preclose_displacement = max(preclose_displacement, float(np.linalg.norm(position - initial_cup)))
        if close_step - round(.10 / dt) <= step < close_step:
            flange_p, flange_r = trial.flange_pose()
            grasp_position_error = max(grasp_position_error, float(np.linalg.norm(flange_p - position_target)))
            angle = Rotation.from_matrix(goal[:3, :3] @ flange_r.T).magnitude()
            grasp_rotation_error = max(grasp_rotation_error, float(np.rad2deg(angle)))
        if lift_step - round(.25 / dt) <= step < lift_step:
            close_opposed += int(touching)
        if phase == 'Hold':
            tilt = np.rad2deg(np.arccos(np.clip(rotation[2, 2], -1., 1.)))
            flange_p, flange_r = trial.flange_pose()
            relative = flange_r.T @ (position - flange_p)
            if held_reference is None:
                held_reference = relative
            hold.append((height, tilt, touching, np.linalg.norm(relative - held_reference)))
        if frame is not None:
            frames.append((frame, t, height, phase, forces.get('thumb', 0.), forces.get(opposed, 0.), arm_goal.copy(),
                           np.array(hand_goal, dtype=float)))

    minimum, tilt, fraction, slip = np.asarray(hold).T
    checks = {
        'grasp_pose_reached': grasp_position_error <= .002 and grasp_rotation_error <= 1.,
        'cup_undisturbed_before_close': preclose_displacement <= .002,
        'no_unexpected_robot_contacts': not unexpected,
        'arm_joint_limits_respected': max_limit_violation <= .001,
        'opposed_contacts_before_lift': close_opposed * dt >= .05,
        'hold_above_5_cm': bool(minimum.min() >= MIN_HEIGHT),
        'opposed_contacts_while_holding': bool(fraction.mean() >= .95),
        'cup_upright': bool(tilt.max() <= 20.),
        'hold_slip_below_10_mm': bool(slip.max() <= .01),
    }
    stats = dict(passed=all(checks.values()), checks=checks, minimum_hold_height_m=float(minimum.min()),
                 peak_height_m=peak, hold_seconds=HOLD_SECONDS, max_hold_tilt_deg=float(tilt.max()),
                 max_hold_slip_m=float(slip.max()), opposed_hold_fraction=float(fraction.mean()),
                 opposed_contact_time_before_lift_s=close_opposed * dt,
                 actual_grasp_position_error_m=grasp_position_error, actual_grasp_rotation_error_deg=grasp_rotation_error,
                 max_preclose_cup_displacement_m=preclose_displacement, unexpected_contacts=sorted(unexpected),
                 max_arm_joint_speed_rad_s=max_joint_speed.tolist(), max_arm_joint_error_rad=max_joint_error.tolist(),
                 max_arm_limit_violation_rad=max_limit_violation, phase_start_s=times)
    return stats, frames


def run(cli):
    source = cli.lift_dir.resolve()
    root = cli.xarm_root.resolve()
    grasp, stage2 = load_grasp(source, cli.frame)
    cli.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cli.output_dir.parent) as work:
        staging = Path(work) / 'result'
        staging.mkdir()
        urdf = arm_urdf_path(root, cli.arm_urdf, cli.xacro, Path(work))
        xml, base = build_scene(staging, source, grasp, urdf, root, cli.base_xy)
        scene = staging / 'scene.xml'
        ET.ElementTree(xml).write(scene, encoding='utf-8')
        model = mujoco.MjModel.from_xml_path(str(scene))
        trial = ArmTrial(model, grasp, xml)
        home = np.array(cli.home)
        if np.any(home < trial.bounds[0]) or np.any(home > trial.bounds[1]):
            raise ValueError('The home pose exceeds xArm7 joint limits')
        preshape = flowhoi_preshape(grasp) if cli.preshape == 'flowhoi' else None
        object_pose = grasp['T_scene_object']

        def attempt(yaw):
            object_flange = about_object_axis(yaw) @ grasp['T_object_flange']
            approach_dir = None
            if cli.approach == 'flowhoi':
                approach_dir = flowhoi_approach(grasp, object_pose @ about_object_axis(yaw))
            return plan(trial, object_pose @ object_flange, home, approach_dir=approach_dir, preshape=preshape)

        paths, targets, clearances, choice = plan_over_symmetry(attempt, cli.object_symmetry)
        approach_dir = flowhoi_approach(grasp, np.eye(4)) if cli.approach == 'flowhoi' else None
        used_preshape = preshape if choice['hand_mode'] == 'flowhoi' else None
        stats, frames = simulate(trial, paths, targets, home, str(grasp['contact_fingers'][1]), used_preshape)
        stats.update(model='xArm7 + RH56F1', replay_title='xArm7 cup lift', grasp_frame=cli.frame,
                     source_lift=str(source), contact_fingers=stage2['contact_fingers'],
                     cup_mass_kg=stage2['cup_mass_kg'], sliding_friction=stage2['sliding_friction'],
                     physics_timestep_s=model.opt.timestep,
                     planned_minimum_clearance_m=dict(zip(('transit', 'approach', 'lift'),
                                                          (d for d, _ in clearances))),
                     base_pose_scene=base.tolist(),
                     camera=dict(distance=1.46, azimuth=122, elevation=-27),
                     flowhoi_approach_available=approach_dir is not None,
                     flowhoi_preshape_available=preshape is not None, **choice)
        qpos, times, height, phase, thumb, other, commanded, hand_commanded = zip(*frames)
        np.savez_compressed(staging / 'lift_replay.npz', qpos=qpos, fps=FPS, time_s=times,
                            cup_height_m=height, phase=phase, thumb_force_n=thumb, opposing_force_n=other)
        np.savez_compressed(staging / 'xarm7_path.npz', time_s=times, phase=phase, joint_names=ARM_NAMES,
                            q_target_rad=commanded, q_actual_rad=np.array(qpos)[:, trial.arm_q],
                            hand_joint_names=grasp['joint_names'], q_hand_target_rad=hand_commanded,
                            T_scene_base=base, T_scene_pregrasp=targets[0], T_scene_grasp=targets[1],
                            T_scene_lift=targets[2], source_frame=cli.frame)
        start = trial.open_qpos.copy()
        start[trial.arm_q] = home
        ctrl = np.zeros(model.nu)
        ctrl[trial.arm_ctrl], ctrl[trial.hand_ctrl] = home, trial.open_angles
        keyframe = ET.SubElement(xml, 'keyframe')
        ET.SubElement(keyframe, 'key', name='home', qpos=text(start), ctrl=text(ctrl))
        ET.indent(xml)
        ET.ElementTree(xml).write(scene, encoding='utf-8')
        (staging / 'lift_metrics.json').write_text(json.dumps(stats, indent=2) + '\n')
        replay = Path(__file__).resolve().parents[1] / 'visualize/replay_rh56f1_lift.py'
        shutil.copyfile(replay, staging / 'replay.py')
        staging.rename(cli.output_dir)
    print(f'Results at {cli.output_dir} | hold {stats["minimum_hold_height_m"] * 100:.2f} cm | passed {stats["passed"]}')
    print(stats['checks'], flush=True)
    return stats['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lift-dir', type=Path, required=True, help='Passing stage 2 scene and retarget archive')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=49)
    add_arm_model_arguments(parser)
    parser.add_argument('--base-xy', type=float, nargs=2, help='xArm7 base position in scene metres')
    parser.add_argument('--home', type=float, nargs=7, default=HOME, help='Home arm angles in radians')
    add_flowhoi_motion_arguments(parser)
    cli = parser.parse_args()
    if cli.output_dir.exists():
        parser.error('Use a new output directory')
    if not np.isfinite(cli.home).all() or (cli.base_xy is not None and not np.isfinite(cli.base_xy).all()):
        parser.error('Home and base coordinates must be finite')
    try:
        passed = run(cli)
    except subprocess.CalledProcessError as error:
        parser.error(error.stderr.strip())
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
