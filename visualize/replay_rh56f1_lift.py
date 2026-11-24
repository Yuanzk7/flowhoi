"""Replay recorded RH56F1 lift dynamics or export an annotated MP4."""

import argparse
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter, sleep

import numpy as np


def camera_setup(camera, model, metrics, detail):
    settings = {} if detail else metrics.get('camera', {})
    camera.lookat[:] = model.body('cup').pos + [0, 0, .05] if detail else model.stat.center
    camera.distance = settings.get('distance', .68)
    camera.azimuth = settings.get('azimuth', 125)
    camera.elevation = settings.get('elevation', -21)


def render(mj, model, data, replay, metrics, video, detail):
    from PIL import Image, ImageDraw, ImageFont

    def font(size, bold=False):
        try:
            return ImageFont.truetype('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf', size)
        except OSError:
            return ImageFont.load_default(size)

    cam, option = mj.MjvCamera(), mj.MjvOption()
    camera_setup(cam, model, metrics, detail)
    option.geomgroup[3] = 0
    option.sitegroup[3] = 0
    label_font, value_font = font(16), font(36, True)
    title_font, phase_font = font(27, True), font(17, True)
    video.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=video.parent) as tmp, mj.Renderer(model, 720, 960) as renderer:
        path = Path(tmp) / 'lift.mp4'
        command = ['ffmpeg', '-nostdin', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', '1280x720', '-r', str(int(replay['fps'])), '-i', '-', '-c:v', 'libx264',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for q, time, height, phase, thumb, other in zip(
                    replay['qpos'], replay['time_s'], replay['cup_height_m'], replay['phase'],
                    replay['thumb_force_n'], replay['opposing_force_n']):
                data.qpos[:] = q
                mj.mj_forward(model, data)
                renderer.update_scene(data, camera=cam, scene_option=option)
                frame = Image.new('RGB', (1280, 720), '#16212d')
                frame.paste(Image.fromarray(renderer.render()), (0, 0))
                draw = ImageDraw.Draw(frame)
                draw.text((986, 32), 'MUJOCO / PHYSICS', fill='#91b4da', font=label_font)
                draw.text((986, 64), metrics.get('replay_title', 'RH56F1 cup lift'), fill='white', font=title_font)
                fill, color = ('#204038', '#a7edc9') if metrics['passed'] else ('#4d332d', '#ffc3ac')
                draw.rounded_rectangle((984, 114, 1256, 154), radius=7, fill=fill)
                draw.text((997, 124), 'LIFT TEST PASSED' if metrics['passed'] else 'LIFT TEST FAILED',
                          fill=color, font=phase_font)
                draw.text((986, 186), str(phase).upper(), fill='#9ce2ce', font=phase_font)
                draw.text((986, 221), 'Cup height above table', fill='#bbcce0', font=label_font)
                draw.text((986, 252), f'{max(height, 0.) * 100:.1f} cm', fill='white', font=value_font)
                draw.line((986, 330, 1254, 330), fill='#385069')
                draw.text((986, 352), 'Recorded contact force', fill='#bbcce0', font=label_font)
                draw.text((986, 387), f'Thumb   {thumb:.2f} N', fill='white', font=label_font)
                finger = metrics['contact_fingers'][1].capitalize()
                draw.text((986, 418), f'{finger}   {other:.2f} N', fill='white', font=label_font)
                draw.text((986, 481), f'Minimum hold {metrics["minimum_hold_height_m"] * 100:.1f} cm',
                          fill='#bbcce0', font=label_font)
                draw.text((986, 512), f'Hold window {metrics["hold_seconds"]:.0f} s', fill='#bbcce0', font=label_font)
                draw.text((986, 628), f'{time:.2f} s', fill='#e0e8f4', font=phase_font)
                draw.rounded_rectangle((986, 669, 1254, 675), radius=3, fill='#385069')
                progress = 986 + round(268 * time / replay['time_s'][-1])
                draw.rounded_rectangle((986, 669, progress, 675), radius=3, fill='#80c9eb')
                encoder.stdin.write(frame.tobytes())
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('FFmpeg could not encode the lift replay')
        os.replace(path, video)
    print(f'Saved lift video to {video}')


def show(model, data, qpos, fps, metrics, detail):
    import mujoco as mj
    from mujoco import viewer

    with viewer.launch_passive(model, data) as window:
        with window.lock():
            camera_setup(window.cam, model, metrics, detail)
            window.opt.geomgroup[3] = 0
            window.opt.sitegroup[3] = 0
        start = perf_counter()
        while window.is_running():
            frame = int((perf_counter() - start) * fps) % len(qpos)
            with window.lock():
                data.qpos[:] = qpos[frame]
                mj.mj_forward(model, data)
            window.sync()
            sleep(.5 / fps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--detail', action='store_true', help='Center the camera on the cup')
    cli = parser.parse_args()
    if cli.video:
        os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(cli.scene_dir / 'scene.xml'))
    data = mujoco.MjData(model)
    metrics = json.loads((cli.scene_dir / 'lift_metrics.json').read_text())
    with np.load(cli.scene_dir / 'lift_replay.npz', allow_pickle=False) as source:
        replay = {name: source[name] for name in source.files}
    if not len(replay['qpos']) or replay['qpos'].shape[1] != model.nq:
        raise ValueError('Replay joint positions do not match the scene')
    if cli.video:
        render(mujoco, model, data, replay, metrics, cli.video, cli.detail)
    else:
        show(model, data, replay['qpos'], replay['fps'].item(), metrics, cli.detail)


if __name__ == '__main__':
    main()
