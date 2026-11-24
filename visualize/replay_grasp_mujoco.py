"""Play an exported Stage 1 scene in MuJoCo, or render it to MP4."""

import argparse
import json
import math
import os
from pathlib import Path
from queue import SimpleQueue
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter, sleep

import numpy as np


class Replay:
    def __init__(self, directory):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(directory / 'scene.xml'))
        self.data = mujoco.MjData(self.model)
        self.info = json.loads((directory / 'info.json').read_text())
        with np.load(directory / 'replay.npz', allow_pickle=False) as replay:
            self.pos = replay['object_pos']
            self.quat = replay['object_quat']
            self.fps = int(replay['fps'])
            self.center = replay['camera_center']
            self.extent = float(replay['camera_extent'])
            self.detail_center = replay.get('detail_center', self.center)
            self.detail_extent = float(replay.get('detail_extent', self.extent))
        self.frames = len(self.pos)
        if not self.frames or self.pos.shape != (self.frames, 3) or self.quat.shape != (self.frames, 4) or self.fps < 1:
            raise ValueError('Replay positions, orientations or frame rate are invalid')
        self.hand_geom = np.array([[self.model.geom(f'{side}_{i:03d}').id for side in ('left', 'right')]
                                   for i in range(self.frames)])
        self.object_id = self.model.body_mocapid[self.model.body('object_pose').id]
        self.frame = 0

    def set_frame(self, frame):
        self.model.geom_group[self.hand_geom[self.frame]] = 3
        self.model.geom_group[self.hand_geom[frame]] = 2
        self.data.mocap_pos[self.object_id] = self.pos[frame]
        self.data.mocap_quat[self.object_id] = self.quat[frame]
        self.mj.mj_forward(self.model, self.data)
        self.frame = frame

    def set_camera(self, camera, zoom=0.0):
        camera.lookat[:] = (1 - zoom) * self.center + zoom * self.detail_center
        camera.distance = (1 - zoom) * self.extent * 1.01 + zoom * self.detail_extent * 1.2
        camera.azimuth = 130 - 42 * zoom
        camera.elevation = -30 - 6 * zoom


def show(replay, speed):
    from mujoco import viewer

    keys = SimpleQueue()
    replay.set_frame(0)
    print('Use Space to pause or play and the arrow keys to step through frames.')
    print(f'Seed {replay.info["seed"]} | {replay.info["text"]}')
    with viewer.launch_passive(replay.model, replay.data, key_callback=keys.put) as window:
        with window.lock():
            replay.set_camera(window.cam)
            window.opt.geomgroup[3] = 0
        position = 0.0
        playing = True
        last = perf_counter()
        while window.is_running():
            now = perf_counter()
            if playing:
                position = (position + (now - last) * replay.fps * speed) % replay.frames
            last = now
            while not keys.empty():
                key = keys.get()
                if key == 32:
                    playing = not playing
                elif key in (262, 263):
                    playing = False
                    position = (int(position) + (1 if key == 262 else -1)) % replay.frames
            frame = int(position)
            if frame != replay.frame:
                with window.lock():
                    replay.set_frame(frame)
            window.sync()
            sleep(1 / 60)


def video_panel(replay):
    from PIL import Image, ImageDraw, ImageFont

    def font(size, bold=False):
        try:
            return ImageFont.truetype('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf', size)
        except OSError:
            return ImageFont.load_default(size)

    panel = Image.new('RGB', (320, 720), '#151d28')
    draw = ImageDraw.Draw(panel)
    draw.line((0, 0, 0, 720), fill='#344254', width=2)
    draw.text((25, 26), 'FLOWHOI / MUJOCO', fill='#83a6c9', font=font(14, True))
    draw.text((25, 52), 'Stage 1 replay', fill='#f3f7fc', font=font(29, True))

    record = replay.info.get('record')
    if record is None:
        badge, fill, color = 'UNSCORED CANDIDATE', '#27364b', '#afcce9'
    elif record['passed']:
        badge, fill, color = 'PASSED FILTERS', '#193a32', '#92e4bf'
    else:
        badge = 'NO RIGHT HAND CONTACT' if not record['checks']['right_contact'] else 'FAILED FILTERS'
        fill, color = '#492b26', '#ffb18b'
    draw.rounded_rectangle((24, 106, 296, 145), radius=8, fill=fill)
    draw.text((36, 117), badge, fill=color, font=font(13, True))
    draw.text((25, 171), f'Seed {replay.info["seed"]}  /  {replay.frames} frames', fill='#dbe6f3', font=font(17))
    draw.text((25, 198), replay.info.get('trajectory_method', 'Kinematic playback'), fill='#a5b5c8', font=font(16))

    y = 259
    for label, color in [('SAM3D object', '#70d9b8'), ('Right MANO hand', '#ff915c'), ('Left MANO hand', '#77b8fa')]:
        draw.rounded_rectangle((25, y + 4, 41, y + 20), radius=4, fill=color)
        draw.text((53, y), label, fill='#e0e8f4', font=font(17))
        y += 35

    if record is not None:
        draw.line((25, 378, 295, 378), fill='#344254')
        window = len(record['right_finger_distances_m'])
        if record['passed']:
            count = min(record['right_contact_counts'])
            draw.text((25, 400), 'Right hand contact', fill='#b1c3d8', font=font(17))
            draw.text((25, 425), f'{count} / 5 fingers', fill='#f3f7fc', font=font(35, True))
            draw.text((25, 479), f'Across final {window} frames', fill='#a5b5c8', font=font(15))
            depth = record['right_max_penetration_m'] * 1000
            draw.text((25, 505), f'Max penetration {depth:.2f} mm', fill='#a5b5c8', font=font(15))
        else:
            nearest = float(np.min(record['right_finger_distances_m']))
            draw.text((25, 400), 'Closest right finger', fill='#b1c3d8', font=font(17))
            draw.text((25, 425), f'{nearest * 100:.1f} cm', fill='#f3f7fc', font=font(36, True))
            draw.text((25, 481), f'From object in final {window} frames', fill='#a5b5c8', font=font(15))
    if replay.info['table_z_m'] is not None:
        draw.text((25, 542), 'Grid marks tabletop height', fill='#a5b5c8', font=font(15))
        draw.text((25, 565), 'Grid spacing 10 cm', fill='#a5b5c8', font=font(15))
    return panel, font(16, True)


def render_video(replay, output, speed, detail):
    from PIL import Image, ImageDraw

    output.parent.mkdir(parents=True, exist_ok=True)
    panel, frame_font = video_panel(replay)
    camera = replay.mj.MjvCamera()
    replay.set_camera(camera)
    options = replay.mj.MjvOption()
    options.geomgroup[3] = 0
    timeline = np.minimum(np.arange(math.ceil(replay.frames / speed)) * speed, replay.frames - 1).astype(int)
    zoom = np.zeros(len(timeline))
    if detail:
        timeline = np.r_[timeline, np.full(replay.fps, replay.frames - 1)]
        ease = np.linspace(0, 1, replay.fps)
        zoom = np.r_[zoom, ease * ease * (3 - 2 * ease)]
    timeline = np.r_[timeline, np.full(replay.fps, replay.frames - 1)]
    zoom = np.r_[zoom, np.full(replay.fps, float(detail))]

    with TemporaryDirectory(dir=output.parent) as directory, replay.mj.Renderer(replay.model, 720, 960) as renderer:
        video = Path(directory) / 'preview.mp4'
        command = ['ffmpeg', '-nostdin', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', '1280x720', '-r', str(replay.fps), '-i', '-', '-c:v', 'libx264',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as ffmpeg:
            last = None
            for frame, magnification in zip(timeline, zoom):
                if (frame, magnification) != last:
                    replay.set_frame(frame)
                    replay.set_camera(camera, magnification)
                    renderer.update_scene(replay.data, camera=camera, scene_option=options)
                    image = Image.new('RGB', (1280, 720))
                    image.paste(Image.fromarray(renderer.render()), (0, 0))
                    image.paste(panel, (960, 0))
                    draw = ImageDraw.Draw(image)
                    draw.text((985, 632), f'Frame {frame:02d} / {replay.frames - 1:02d}', fill='#e0e8f4', font=frame_font)
                    draw.rounded_rectangle((985, 668, 1255, 675), radius=3, fill='#344254')
                    end = 985 + int(270 * (frame + 1) / replay.frames)
                    draw.rounded_rectangle((985, 668, end, 675), radius=3, fill='#80c9eb')
                    rgb = image.tobytes()
                    last = (frame, magnification)
                ffmpeg.stdin.write(rgb)
            ffmpeg.stdin.close()
            if ffmpeg.wait():
                raise RuntimeError('ffmpeg could not encode the MuJoCo preview')
        os.replace(video, output)
    print(f'Saved MuJoCo preview to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, default=Path(__file__).resolve().parent,
                        help='Directory created by export_grasp_mujoco; defaults to this script’s directory')
    parser.add_argument('--video', type=Path, help='Render an MP4 instead of opening the interactive viewer')
    parser.add_argument('--detail', action='store_true', help='Zoom in on the final grasp in the MP4')
    parser.add_argument('--speed', type=float, default=0.5, help='Playback speed relative to the source rate')
    cli = parser.parse_args()
    if not math.isfinite(cli.speed) or cli.speed <= 0:
        parser.error('--speed must be finite and positive')
    if cli.video is not None:
        if cli.video.suffix.lower() != '.mp4':
            parser.error('--video must end with .mp4')
        os.environ.setdefault('MUJOCO_GL', 'egl')
    try:
        replay = Replay(cli.scene_dir)
        if cli.video is not None:
            render_video(replay, cli.video, cli.speed, cli.detail)
        else:
            show(replay, cli.speed)
    except (ImportError, ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
