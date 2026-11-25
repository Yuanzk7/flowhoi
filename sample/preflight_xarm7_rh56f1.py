"""Inspect a FlowHOI cup grasp against measured xArm7 and RH56F1 settings offline."""

import argparse
import json
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from sample.rh56f1_retarget import preshape_joints
from sample.validate_rh56f1 import LIFT_HEIGHT, load_retarget
from sample.validate_xarm7_rh56f1 import (
    ARM_NAMES, ArmTrial, PlanningFailure, add_arm_model_arguments, add_flowhoi_motion_arguments, arm_urdf_path,
    about_object_axis, build_scene, flange_position, flowhoi_approach, flowhoi_preshape, hand_mount_in_flange,
    load_grasp, plan, plan_over_symmetry, preshape_states,
)


RAW_NAMES = ('little', 'ring', 'middle', 'index', 'thumb_bend', 'thumb_rot')
# Default envelope of xarm-teleop/src/control/inspire_hand.py.
DEFAULT_RAW_LIMITS = {'min': [900, 900, 900, 900, 900, 1500],
                      'max': [1740, 1740, 1740, 1740, 1350, 1650]}
TRANSFORMS = ('T_base_object', 'T_base_table', 'T_flange_hand', 'T_flange_tcp')
REQUIRED_FIELDS = TRANSFORMS + ('table_size_m', 'start_joints_rad', 'min_clearance_m')


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite values with shape {shape}')
    return array


def rigid_pose(value, name):
    pose = finite_array(value, (4, 4), name)
    rotation = pose[:3, :3]
    if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5, rtol=0)):
        raise ValueError(f'{name} must be a rigid transform, with translation in metres')
    return pose


def read_calibration(path):
    config = json.loads(path.read_text()) if path is not None else {}
    if not isinstance(config, dict):
        raise ValueError('Calibration must be a JSON object')
    missing = [name for name in REQUIRED_FIELDS if config.get(name) is None]
    for name in TRANSFORMS:
        if config.get(name) is not None:
            config[name] = rigid_pose(config[name], name)
    if config.get('table_size_m') is not None:
        config['table_size_m'] = finite_array(config['table_size_m'], (3,), 'table_size_m')
        if np.any(config['table_size_m'] <= 0):
            raise ValueError('table_size_m must have three positive full dimensions')
    if config.get('start_joints_rad') is not None:
        config['start_joints_rad'] = finite_array(config['start_joints_rad'], (7,), 'start_joints_rad')
    if config.get('min_clearance_m') is not None:
        margin = finite_array(config['min_clearance_m'], (), 'min_clearance_m').item()
        if margin <= 0:
            raise ValueError('min_clearance_m must be positive')
        config['min_clearance_m'] = margin
    return config, missing


def inspect_hand(grasp, retarget, raw_limits):
    if tuple(grasp['joint_names']) != tuple(retarget.ACTUATED):
        raise ValueError('The RH56F1 archive joint names do not match the actuator mapping')
    limits = DEFAULT_RAW_LIMITS if raw_limits is None else raw_limits
    if not isinstance(limits, dict):
        raise ValueError('rh56_raw_limits must contain min and max vectors')
    lower = finite_array(limits.get('min'), (6,), 'rh56_raw_limits.min')
    upper = finite_array(limits.get('max'), (6,), 'rh56_raw_limits.max')
    bounds = np.r_[lower, upper]
    if (bounds != np.rint(bounds)).any() or (lower < 0).any() or (upper > 32767).any() or (lower >= upper).any():
        raise ValueError('Each RH56F1 raw range must have integer min < max in [0, 32767]')
    lower, upper = lower.astype(int), upper.astype(int)
    phases = {}
    violations = []
    commands = [(phase, grasp[field], field) for phase, field in
                (('open', 'q_open_rad'), ('fitted_grasp', 'q_hand_rad'), ('close_command', 'q_close_command_rad'))]
    if 'preshape' in grasp:
        # The widest FlowHOI preshape the hand is commanded to during the approach.
        widest = preshape_joints(grasp['q_open_rad'], float(np.max(grasp['preshape'])))
        commands.insert(0, ('preshape_widest', widest, 'preshape'))
    for phase, value, field in commands:
        radians = finite_array(value, (6,), field)
        if np.any(radians < -1e-8) or np.any(radians > retarget.UPPER + 1e-8):
            raise ValueError(f'{field} is outside the RH56F1 kinematic model bounds')
        raw = retarget.joints_to_raw(radians)
        phases[phase] = raw.tolist()
        for i, axis in enumerate(RAW_NAMES):
            if raw[i] < lower[i] or raw[i] > upper[i]:
                violations.append(dict(phase=phase, axis=axis, requested=int(raw[i]),
                                       min=int(lower[i]), max=int(upper[i])))
    return dict(status='failed' if violations else 'passed', axes=RAW_NAMES,
                limits_source='xarm-teleop defaults' if raw_limits is None else 'calibration input',
                raw_limits={'min': lower.tolist(), 'max': upper.tolist()},
                requested_raw=phases, violations=violations)


def describe_clearance(result, margin):
    distance, pair = result
    return dict(passed=bool(distance >= margin), distance_m=float(distance) if np.isfinite(distance) else None,
                pair=pair, at_least=pair is None)


def inspect_arm(cli, grasp, source, config):
    margin = config['min_clearance_m']
    start = config['start_joints_rad']
    table = config['T_base_table']
    up = table[:3, 2]
    object_pose = config['T_base_object']
    file = ET.parse(source / 'scene.xml')
    stage2_hand = file.find("worldbody/body[@name='lift_carriage']/body[@name='right_hand_base']")
    if stage2_hand is None:
        raise ValueError('The stage 2 scene has no RH56F1 mount')
    source_mount = hand_mount_in_flange(stage2_hand, rigid_pose(grasp['T_scene_flange'], 'T_scene_flange'))
    object_flange = rigid_pose(grasp['T_object_flange'], 'T_object_flange')
    # Preserve the cup-relative hand pose when the measured adapter differs from the simulated adapter.
    target = object_pose @ object_flange @ source_mount @ np.linalg.inv(config['T_flange_hand'])
    result = dict(status='failed', grasp_ik_found=False, min_clearance_m=margin,
                  T_base_flange_grasp=target.tolist(), T_base_tcp_grasp=(target @ config['T_flange_tcp']).tolist())

    root = cli.xarm_root.resolve()
    with TemporaryDirectory() as directory:
        work = Path(directory)
        urdf = arm_urdf_path(root, cli.arm_urdf, cli.xacro, work)
        xml, _ = build_scene(work, source, grasp, urdf, root, measured=config)
        scene = work / 'scene.xml'
        ET.ElementTree(xml).write(scene, encoding='utf-8')
        model = mujoco.MjModel.from_xml_path(str(scene))
        trial = ArmTrial(model, grasp, xml)
        trial.set_arm(start)
        mesh_rotation = trial.data.geom_xmat[trial.cup_geom].reshape(3, 3)
        cup_points = trial.cup_points @ mesh_rotation.T + trial.data.geom_xpos[trial.cup_geom]
        result['cup_bottom_above_table_m'] = float(np.min((cup_points - table[:3, 3]) @ up))
        if np.any(start < trial.bounds[0]) or np.any(start > trial.bounds[1]):
            result['reason'] = 'The measured starting pose exceeds xArm7 model joint limits'
            return result

        closed = grasp['q_close_command_rad']
        opened = grasp['q_open_rad']
        q_cup = trial.cup_q
        cup_quat = Rotation.from_matrix(object_pose[:3, :3]).as_quat(scalar_first=True)

        def during_close(fraction):
            trial.set_hand(opened + fraction * (closed - opened))

        def during_lift(fraction):
            trial.set_hand(closed)
            trial.data.qpos[q_cup:q_cup + 3] = object_pose[:3, 3] + fraction * LIFT_HEIGHT * up
            trial.data.qpos[q_cup + 3:q_cup + 7] = cup_quat

        def clearance(phase, path, samples, state=None, *, table_only=False):
            apply_state = {'close': during_close, 'lift': during_lift}.get(phase, state)
            return trial.clearance(path, phase in ('transit', 'approach'), samples,
                                   pairs=trial.table_pairs if table_only else None, apply_state=apply_state,
                                   cutoff=max(.10, margin * 1.05))

        preshape = flowhoi_preshape(grasp) if cli.preshape == 'flowhoi' else None

        def attempt(yaw):
            turned = object_pose @ about_object_axis(yaw)
            goal = turned @ object_flange @ source_mount @ np.linalg.inv(config['T_flange_hand'])
            approach_dir = flowhoi_approach(grasp, turned) if cli.approach == 'flowhoi' else None
            return plan(trial, goal, start, margin=margin, up=up, check_path=clearance,
                        approach_dir=approach_dir, preshape=preshape)

        try:
            paths, targets, _, choice = plan_over_symmetry(attempt, cli.object_symmetry)
            result.update(T_base_flange_grasp=targets[1].tolist(),
                          T_base_tcp_grasp=(targets[1] @ config['T_flange_tcp']).tolist())
        except PlanningFailure as error:
            result.update(grasp_ik_found=error.grasp_ik_found, reason=str(error),
                          best_sampled_clearance_by_phase={
                              phase: describe_clearance(sample, margin)
                              for phase, sample in error.best_clearances.items()
                          })
            return result

        transit, approach, lift = paths
        grasp_angles = approach(1.)

        def stationary(u):
            return np.broadcast_to(grasp_angles, (*np.shape(u), grasp_angles.size))

        used = preshape if choice['hand_mode'] == 'flowhoi' else None
        states = preshape_states(trial, used, np.linalg.norm(targets[0][:3, 3] - flange_position(trial, start)))
        phases = (('transit', transit, 97, states[0]), ('approach', approach, 65, states[1]),
                  ('close', stationary, 33, None), ('lift', lift, 65, None))
        result.update(status='passed', grasp_ik_found=True, joint_names=ARM_NAMES, **choice,
                      waypoints_rad=dict(start=start.tolist(), pregrasp=approach(0.).tolist(),
                                         grasp=grasp_angles.tolist(), lift=lift(1.).tolist()),
                      flange_targets={key: pose.tolist() for key, pose in zip(('pregrasp', 'grasp', 'lift'), targets)},
                      offsets=dict(T_flange_hand=config['T_flange_hand'].tolist(),
                                   T_flange_tcp=config['T_flange_tcp'].tolist()),
                      clearances={
                          phase: dict(samples=count,
                                      robot=describe_clearance(clearance(phase, path, count, state), margin),
                                      table=describe_clearance(clearance(phase, path, count, state, table_only=True),
                                                               margin))
                          for phase, path, count, state in phases
                      })
    return result


def run(cli):
    source = cli.lift_dir.resolve()
    grasp, _ = load_grasp(source, cli.frame)
    config, missing = read_calibration(cli.calibration)
    retarget = load_retarget(cli.retarget_module)
    hand = inspect_hand(grasp, retarget, config.get('rh56_raw_limits'))
    if missing:
        arm = dict(status='skipped', reason='Measured calibration is incomplete')
    else:
        try:
            arm = inspect_arm(cli, grasp, source, config)
        except subprocess.CalledProcessError as error:
            arm = dict(status='failed', reason=(error.stderr or str(error)).strip())
        except (OSError, ValueError, KeyError) as error:
            arm = dict(status='failed', reason=str(error))
    report = dict(kind='offline_xarm7_rh56f1_preflight', hardware_connected=False, hardware_validated=False,
                  source_lift=str(source), grasp_frame=cli.frame,
                  calibration=str(cli.calibration.resolve()) if cli.calibration is not None else None,
                  missing_calibration_fields=missing, hand=hand, arm=arm)
    if hand['status'] == arm['status'] == 'passed':
        report['status'] = 'passed'
    else:
        report['status'] = 'failed' if 'failed' in (hand['status'], arm['status']) else 'incomplete'
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile('w', dir=cli.output.parent, delete=False) as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
        stream.write('\n')
    Path(stream.name).replace(cli.output)
    print(f'{report["status"]}: hand {hand["status"]}, arm {arm["status"]} | {cli.output}')
    return report['status'] == 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lift-dir', type=Path, required=True, help='Passing stage 2 RH56F1 bundle')
    parser.add_argument('--calibration', type=Path, help='Measured geometry and optional RH56F1 limits JSON')
    parser.add_argument('--output', type=Path, required=True, help='Offline report JSON')
    parser.add_argument('--frame', type=int, default=49)
    parser.add_argument('--retarget-module', type=Path, help='Override the bundled RH56F1 retarget module')
    add_arm_model_arguments(parser)
    add_flowhoi_motion_arguments(parser)
    cli = parser.parse_args()
    if cli.output.suffix != '.json':
        parser.error('--output must end in .json')
    if cli.calibration is not None and cli.calibration.resolve() == cli.output.resolve():
        parser.error('Report and calibration paths must differ')
    if cli.output.resolve() == (cli.lift_dir / 'lift_metrics.json').resolve():
        parser.error('Report and source lift metrics paths must differ')
    try:
        passed = run(cli)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
