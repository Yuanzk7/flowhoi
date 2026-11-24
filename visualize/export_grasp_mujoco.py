"""Export a Stage 1 candidate and its object as a portable MuJoCo replay."""

import argparse
import json
from pathlib import Path
from shutil import copyfile
from struct import pack
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from sample.grasp_geometry import ManoHand, decode_hand, load_mesh
from sample.select_grasp import IDENTITY_POSE, pose_arrays, read_archive


HAND_COLORS = {'left': '0.35 0.67 0.96 1', 'right': '1 0.51 0.27 1'}


def write_mesh(path, vertices, faces):
    """Write MuJoCo's binary mesh format without resampling the surface."""
    with path.open('wb') as file:
        file.write(pack('<4i', len(vertices), 0, 0, len(faces)))
        np.asarray(vertices, dtype='<f4').tofile(file)
        np.asarray(faces, dtype='<i4').tofile(file)


def vector(values):
    return ' '.join(format(float(value), '.9g') for value in values)


def write_scene(directory, mesh, hands, faces, object_pos, object_rot, table_z):
    model = ET.Element('mujoco', model='FlowHOI Stage 1 playback')
    ET.SubElement(model, 'compiler', inertiafromgeom='false')
    option = ET.SubElement(model, 'visual')
    ET.SubElement(option, 'global', offwidth='1280', offheight='720', fovy='39')
    ET.SubElement(option, 'headlight', ambient='0.46 0.46 0.46', diffuse='0.65 0.65 0.65', specular='0.10 0.10 0.10')
    default = ET.SubElement(model, 'default')
    ET.SubElement(default, 'geom', contype='0', conaffinity='0')
    assets = ET.SubElement(model, 'asset')
    ET.SubElement(assets, 'texture', name='sky', type='skybox', builtin='gradient',
                  rgb1='0.19 0.23 0.29', rgb2='0.09 0.12 0.16', width='64', height='512')
    world = ET.SubElement(model, 'worldbody')
    ET.SubElement(world, 'light', pos='-0.7 0.1 1.6', dir='0.3 0 -1', diffuse='0.55 0.53 0.48', castshadow='false')

    (directory / 'meshes').mkdir()
    write_mesh(directory / 'meshes/object.msh', mesh.vertices, mesh.faces)
    ET.SubElement(assets, 'mesh', name='object', file='meshes/object.msh', smoothnormal='true')
    body = ET.SubElement(world, 'body', name='object_pose', mocap='true')
    ET.SubElement(body, 'geom', name='sam3d_object', type='mesh', mesh='object', rgba='0.40 0.85 0.72 1')

    bounds_min = np.full(3, np.inf)
    bounds_max = np.full(3, -np.inf)
    for translation, rotation in zip(object_pos, object_rot):
        corners = mesh.bounding_box.vertices @ rotation.T + translation
        bounds_min = np.minimum(bounds_min, corners.min(axis=0))
        bounds_max = np.maximum(bounds_max, corners.max(axis=0))
    for side, hand in hands.items():
        bounds_min = np.minimum(bounds_min, hand.vertices.min(axis=(0, 1)))
        bounds_max = np.maximum(bounds_max, hand.vertices.max(axis=(0, 1)))
        for frame, vertices in enumerate(hand.vertices):
            name = f'{side}_{frame:03d}'
            write_mesh(directory / f'meshes/{name}.msh', vertices, faces[side])
            ET.SubElement(assets, 'mesh', name=name, file=f'meshes/{name}.msh', smoothnormal='true', inertia='shell')
            ET.SubElement(world, 'geom', name=name, type='mesh', mesh=name,
                          group='2' if frame == 0 else '3', rgba=HAND_COLORS[side])

    # The tabletop is a wire grid so hands below its height remain visible.
    if table_z is not None:
        low = np.floor((bounds_min[:2] - 0.10) * 10) / 10
        high = np.ceil((bounds_max[:2] + 0.10) * 10) / 10
        for axis in range(2):
            for tick in np.linspace(low[axis], high[axis], round((high[axis] - low[axis]) * 10) + 1):
                start, end = np.r_[low, table_z], np.r_[high, table_z]
                start[axis] = end[axis] = tick
                ET.SubElement(world, 'geom', type='capsule', size='0.0011', fromto=vector(np.r_[start, end]),
                              rgba='0.42 0.49 0.58 0.45')
    center = (bounds_min + bounds_max) / 2
    extent = float(np.linalg.norm(bounds_max - bounds_min))
    ET.SubElement(model, 'statistic', center=vector(center), extent=f'{extent:.6g}')
    ET.indent(model)
    ET.ElementTree(model).write(directory / 'scene.xml', encoding='utf-8')
    detail = np.concatenate((corners, hands['right'].vertices[-1]))
    return center, extent, (detail.min(axis=0) + detail.max(axis=0)) / 2, float(np.linalg.norm(np.ptp(detail, axis=0)))


def export(cli):
    report = json.loads(cli.report.read_text()) if cli.report else {}
    record = None
    if cli.report:
        record = report.get('selected')
        if record is None:
            record = min(report['candidates'], key=lambda row: row['right_required_contact_radius_m'])
        source, index, mesh_path = Path(record['source']), record['index'], Path(report['mesh'])
    else:
        source, index, mesh_path = cli.candidates, cli.index, cli.mesh
    mesh = load_mesh(mesh_path)
    motions, mirrored, seeds = read_archive(source, mesh)
    if not 0 <= index < len(motions):
        raise ValueError(f'Candidate index must be in [0, {len(motions) - 1}]')
    if record is not None and (record['seed'] != seeds[index] or record['mirrored'] != mirrored):
        raise ValueError('The report and source candidate do not agree')
    with np.load(source, allow_pickle=False) as archive:
        text = str(archive['text'][index])
        method = str(archive['trajectory_method'].item()) if 'trajectory_method' in archive else 'Kinematic playback'

    default_pose = report.get('object_pose', IDENTITY_POSE)
    if cli.report and 'object_pose' not in report and report.get('selected') is not None:
        selected_path = cli.report.with_name('right_hand_grasp.npz')
        if selected_path.is_file():
            with np.load(selected_path, allow_pickle=False) as selected:
                default_pose = selected['object_pose']
    object_pose = np.asarray(default_pose) if cli.object_pose is None else np.load(cli.object_pose, allow_pickle=False)
    if object_pose.shape == (9,):
        object_pose = np.repeat(object_pose[None], len(motions[index]), axis=0)
    if object_pose.shape != (len(motions[index]), 9) or not np.isfinite(object_pose).all():
        raise ValueError('Expected one finite object pose or one pose per frame, each with 9 channels')
    object_pos, object_rot = pose_arrays(object_pose)
    object_quat = np.roll(Rotation.from_matrix(object_rot).as_quat(), 1, axis=-1)
    table_z = report.get('table_z_m') if cli.table_z is None else cli.table_z

    models = {side: ManoHand(cli.mano_dir, side) for side in HAND_COLORS}
    hands = {side: decode_hand(motions[index], models, side, mirrored) for side in models}
    faces = {side: model.model.faces for side, model in models.items()}
    if cli.object_pose is not None or cli.table_z is not None:
        record = None  # Evaluation measurements refer to the original placement.
    info = {'source': str(source.resolve()), 'index': index, 'seed': int(seeds[index]), 'text': text,
            'mirrored': mirrored, 'record': record, 'table_z_m': table_z, 'trajectory_method': method}
    output = cli.output_dir.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / 'replay'
        bundle.mkdir()
        center, extent, detail_center, detail_extent = write_scene(bundle, mesh, hands, faces, object_pos, object_rot, table_z)
        np.savez_compressed(bundle / 'replay.npz', object_pos=object_pos, object_quat=object_quat,
                            fps=cli.fps, camera_center=center, camera_extent=extent,
                            detail_center=detail_center, detail_extent=detail_extent)
        (bundle / 'info.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
        copyfile(Path(__file__).with_name('replay_grasp_mujoco.py'), bundle / 'replay.py')
        bundle.rename(output)
    print(f'Wrote MuJoCo replay for seed {seeds[index]}, row {index}, to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--report', type=Path, help='Use the selected or closest candidate from a selection report')
    source.add_argument('--candidates', type=Path, help='Stage 1 candidates archive')
    parser.add_argument('--index', type=int, help='Candidate row when using --candidates; defaults to 0')
    parser.add_argument('--mesh', type=Path, help='Prepared object mesh when using --candidates')
    parser.add_argument('--object-pose', type=Path, help='Override object placement with translation + 6D rotation')
    parser.add_argument('--table-z', type=float, help='Height in metres for the reference grid')
    parser.add_argument('--mano-dir', type=Path, default=Path('assets/smplx/mano'))
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--output-dir', type=Path, required=True)
    cli = parser.parse_args()
    if cli.candidates and cli.mesh is None:
        parser.error('--candidates requires --mesh')
    if cli.report and (cli.mesh is not None or cli.index is not None):
        parser.error('--report supplies the mesh and candidate index')
    cli.index = 0 if cli.index is None else cli.index
    if cli.fps < 1 or (cli.table_z is not None and not np.isfinite(cli.table_z)):
        parser.error('Expected a positive frame rate and finite table height')
    try:
        export(cli)
    except (KeyError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
