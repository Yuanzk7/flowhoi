<p align="center">
  <h1 align="center"><strong>FlowHOI: Flow-based Semantics-Grounded Generation of Hand-Object Interactions for Dexterous Robot Manipulation</strong></h1>
  <p align="center">
    <a href="https://huajian-zeng.github.io/">Huajian Zeng</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=vdo-1aYAAAAJ&hl">Lingyun Chen</a><sup>1,2</sup>, <a href="https://scholar.google.com/citations?hl=zh-CN&user=f7ox7CIAAAAJ">Jiaqi Yang</a><sup>1,3</sup>, <a href="https://scholar.google.com/citations?user=Lh2CthAAAAAJ&hl">Yuantai Zhang</a><sup>1</sup>, <a href="https://fanshi14.github.io/me/">Fan Shi</a><sup>4</sup>, <a href="https://ethliup.github.io/">Peidong Liu</a><sup>5</sup>, <a href="https://xingxingzuo.github.io/">Xingxing Zuo</a><sup>1,*</sup>
    <br>
    <sup>1</sup>Mohamed bin Zayed University of Artificial Intelligence (MBZUAI), <sup>2</sup>Technical University of Munich (TUM), <sup>3</sup>ShanghaiTech University, <sup>4</sup>National University of Singapore (NUS), <sup>5</sup>Westlake University
    <br>
    <sup>*</sup>Corresponding author
    <br>
  </p>
  <p align="center"><strong>NeurIPS 2026</strong></p>
</p>

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2602.13444-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2602.13444)
[![Homepage](https://img.shields.io/badge/Homepage-%F0%9F%8C%90-blue)](https://huajian-zeng.github.io/projects/flowhoi/)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-flowhoi-yellow)](https://huggingface.co/huajian-zeng/flowhoi)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-flowhoi--data-yellow)](https://huggingface.co/datasets/huajian-zeng/flowhoi-data)

</div>

## Updates

[2026-09-25] FlowHOI was accepted to NeurIPS 2026. See you in Sydney!

[2026-09-07] Two-stage inference and visualization code released, along with pretrained checkpoints and GRAB/HOT3D demo data.

[2026-02-13] Paper uploaded to [arXiv](https://arxiv.org/abs/2602.13444).

## 🔥 Highlight

Before acting, a robot should first reason about where it is and how it should
interact with the objects around it. FlowHOI uses hand-object interaction as an
embodiment-agnostic intermediate representation that connects human
demonstrations to robot actions.

**FlowHOI** is a two-stage flow-matching framework that generates semantically
grounded, temporally coherent hand-object interaction (HOI) sequences -- hand
poses, object poses, and hand-object contact states -- conditioned on an
egocentric observation, a language instruction, and a 3D scene reconstruction.

A geometry-centric **Grasping** stage, pretrained on HOI data reconstructed
from large-scale egocentric videos (EgoDex), produces a contact-stable grasp;
a semantics-centric **Manipulation** stage generates the subsequent
interaction. This repository contains the inference and visualization code.

## Installation

1. Create and activate a conda environment:
    ```bash
    conda create -n flowhoi python=3.10 -y
    conda activate flowhoi
    ```
2. Install PyTorch (tested with 2.5.1 + CUDA 12.4). Install torchvision in
   the same command: `vit_pytorch` pulls it in, and letting pip fetch it from
   PyPI in the next step would replace your torch build with whatever version
   that torchvision pins. See the
   [official PyTorch website](https://pytorch.org/get-started/locally/) for
   the command matching your CUDA version:
    ```bash
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
    ```
3. Install the remaining dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Install chumpy (needed to load the MANO model files; the PyPI release is
   incompatible with recent numpy, and its setup requires disabling build
   isolation):
    ```bash
    pip install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
    ```

The T5-Large text encoder (about 3 GB) is downloaded automatically on first
use. It is always loaded, so prompts outside the precomputed text features
work as well.

## Download Checkpoints and Data

1. Pretrained checkpoints (grasping prior, GRAB and HOT3D manipulation
   models) from [huajian-zeng/flowhoi](https://huggingface.co/huajian-zeng/flowhoi):
    ```bash
    bash scripts/download_pretrained.sh
    ```
2. The example data package (the demo sequences with their normalization
   statistics, projection matrices, BPS encodings, text features, and HOT3D
   scene inputs) from
   [huajian-zeng/flowhoi-data](https://huggingface.co/datasets/huajian-zeng/flowhoi-data):
    ```bash
    bash scripts/download_data.sh
    ```
   The package covers the demo sequences only. The full preprocessed splits
   are not redistributed: they are rebuilt from the raw datasets with the
   preprocessing pipeline, which is not part of this release yet (see the TODO
   list below).
3. MANO / SMPL-X body models (registration required) from
   [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) into `assets/smplx/`
   (used for visualization; the path is configured in `aitvconfig.yaml`).
4. GRAB object meshes and subject hand templates (registration required) from
   [grab.is.tue.mpg.de](https://grab.is.tue.mpg.de/). Its licence does not
   permit redistribution, so they are not part of the data package. From the
   extracted `tools/` directory, copy the object meshes (`contact_meshes`,
   under `object_meshes`) into `assets/contact_meshes/`, and the per-subject
   templates (`s*_lhand.ply`, `s*_rhand.ply`, under `subject_meshes`) into
   `assets/female/` and `assets/male/`, keeping the two gender folders. Both
   are needed to render GRAB sequences; the HOT3D path uses neither.
5. HOT3D object models, for rendering HOT3D sequences. Download
   the object library with the [HOT3D toolkit](https://github.com/facebookresearch/hot3d)
   (`dataset_downloader_base_main.py -c Hot3DAssets_download_urls.json`) and put
   the `.glb` files into `assets/hot3d_assets/`. The renderer resolves them
   through `HOT3D_OBJECT_MESH_UID` in `visualize/visualize_sequences.py`.
6. `ffmpeg` and `ffprobe` on `PATH`, for writing videos. If either is
   missing, install both in the active environment:
    ```bash
    conda install -c conda-forge ffmpeg -y
    ```

## Quick Start

```bash
bash demo.sh
```

This generates the two GRAB demo sequences -- top-scoring test sequences with
distinct objects and actions (passing a mug, playing with a toy train) -- and
renders them to `outputs/demo/ours_videos/*.mp4`. Rendering needs the GRAB
meshes from step 4; generation itself does not. Likewise for the
scene-conditioned HOT3D model, whose object models come from step 5:

```bash
DATASET=hot3d bash demo.sh
```

The HOT3D demo writes its videos to `outputs/demo_hot3d/ours_videos/`.
Both demos select `--split_set demo`, the split included in the data package.

On machines without a display, the renderer falls back to an EGL context when
the installed aitviewer supports it; otherwise run `xvfb-run -a bash demo.sh`.

## Inference

Two-stage generation with explicit arguments:

```bash
python -m sample.sample_2stage \
    --model_path save/hot3d_full/model000200000.pt \
    --grasp_model_path save/egodex_grasp/model000500000.pt \
    --guidance --num_samples 1 --seed 42 --split_set demo \
    --output_dir outputs/demo_hot3d
```

The model configuration is read from the `args.json` next to each checkpoint;
arguments passed explicitly on the command line take precedence, including
`--name=value` and boolean overrides such as `--no_random_order`.
The released checkpoints use a 50-frame grasp phase (keyframe index 49);
`--max_frames_interaction` accepts integer lengths from 50 to 200.
`--guidance_param` defaults to 1, using one conditional prediction per step
as in the released sampler. Values above 1, such as `--guidance_param 2.5`,
explicitly enable classifier-free guidance and change the generated motion;
they are outside the validated demo configuration. `--guidance` separately
enables pose-gradient guidance. The grasp stage follows the released
checkpoint's sampling convention without an additional `x_start` pose condition.

The HOT3D model conditions on the reconstructed 3D scene (a global occupancy token plus
local Concerto/SceneSplat tokens); the sampler loads these scene inputs from
the data package and refuses to run without them rather than silently
dropping the conditioning.

`--num_samples N` selects up to N dataset conditions, and
`--num_repetitions R` generates R motions for each selected condition.
With `--eval_entire_set`, all conditions are generated and N becomes the batch
size; the last incomplete batch is retained. `--random_order` shuffles one shared
ID order for both stages. Saved results include the actual condition count,
`num_generated`, `requested_num_samples`, and each row's `repetition_index`.

### 추가 패키지 설치

```bash
python -m pip install open3d mujoco Pillow
```

### Stage 1 기준 결과 저장 및 재현

```bash
python -m sample.sample_2stage \
    --model_path save/grab_full/model000200000.pt \
    --grasp_model_path save/egodex_grasp/model000500000.pt \
    --split_set demo --num_samples 1 --seed 42 \
    --stage1_only --output_dir outputs/stage1_grab_mug

python -m sample.sample_grasp \
    --reference outputs/stage1_grab_mug/stage1_test_demo_b0000_r0000.npz \
    --device cpu --output outputs/stage1_grab_mug/replay.npz
```

### SAM3D 메시를 BPS로 변환

```bash
python -m utils.mesh_to_bps --mesh /path/to/sam3d_raw.obj \
    --longest-extent-m 0.12 --output-dir outputs/custom_object_bps
```

### 파지 후보 생성

```bash
python -m sample.sample_grasp \
    --text 'the individual grasps the bottle.' \
    --mesh outputs/custom_object_bps/object_mesh.ply \
    --bps outputs/custom_object_bps/object_bps.npy \
    --num-samples 10 --seed 42 --output outputs/custom_grasp/candidates.npz
```

### 컵 파지 보정 및 오른손 후보 선별

```bash
python -m sample.fit_cup_grasp \
    --candidates outputs/custom_grasp/candidates.npz --index 0 \
    --mesh outputs/custom_object_bps/object_mesh.ply \
    --output outputs/custom_grasp/geometry_fitted.npz

python -m sample.select_grasp \
    --candidates outputs/custom_grasp/geometry_fitted.npz \
    --mesh outputs/custom_object_bps/object_mesh.ply \
    --object-on-table --output-dir outputs/custom_grasp/selection
```

`fit_cup_grasp`는 생성된 파지의 엄지 방위각에서 탐색을 시작하고(`--yaw-source model`),
생성된 접근 궤적을 보정된 최종 파지에 맞게 강체 이동해 사용합니다(`--approach model`).
이동한 궤적이 검사를 통과하지 못하면 합성 접근으로 대체하며, 어느 쪽을 썼는지와
모델 대비 손바닥 편차가 `metrics_json`에 기록됩니다. 이전 동작은
`--yaw-source legacy --approach synth`입니다.

`select_grasp`는 `right_hand_grasp.npz`와 함께 로봇 무관 중간 표현
`grasp_spec.npz`를 저장합니다(`sample/grasp_interface.py`):

| 항목 | 의미 |
|---|---|
| `T_object_palm` | 물체 좌표계 손바닥 포즈. 원점은 MCP 4개 평균, +z 손목→중지 MCP, +y 새끼→검지 MCP |
| `contact_fingers`, `finger_distance_m` | 최종 프레임 접촉 손가락과 손가락별 표면 거리 |
| `approach_dir_object`, `approach_dir_palm` | 마지막 15프레임의 손바닥 이동 방향 (물체/손바닥 좌표계) |
| `remaining_m`, `preshape` | 프레임별 남은 손바닥 경로 길이와 정규화 개구도(1 최대, 0 최종 파지) |

### MuJoCo에서 Stage 1 궤적 재생

```bash
python -m visualize.export_grasp_mujoco \
    --report outputs/custom_grasp/selection/report.json \
    --output-dir outputs/custom_grasp/mujoco

# 뷰어 열기
python outputs/custom_grasp/mujoco/replay.py

# MP4 저장
python outputs/custom_grasp/mujoco/replay.py \
    --video outputs/custom_grasp/mujoco/preview.mp4
```

### RH56F1 리타게팅 및 리프트 검증

```bash
python -m sample.validate_rh56f1 \
    --grasp outputs/custom_grasp/selection/right_hand_grasp.npz \
    --output-dir outputs/custom_grasp/rh56f1_lift

# 뷰어 열기
python outputs/custom_grasp/rh56f1_lift/replay.py

# MP4 저장
python outputs/custom_grasp/rh56f1_lift/replay.py \
    --video outputs/custom_grasp/rh56f1_lift/lift.mp4
```

기본값 `--retarget-mode palm`은 RH56F1 URDF의 손가락 기저 관절로 같은 규칙의 손바닥
좌표계를 정의해 사람 손바닥에 맞춰 시작하고, 손바닥 포즈 비용을 걸고, 사람이 접촉한
손가락을 대향 손가락으로 먼저 시도합니다. 다음 허용 범위를 벗어난 해는 기각합니다.

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--max-palm-rotation-deg` | 30 | 손목 굽힘을 제외한 회전 |
| `--max-palm-flexion-deg` | 60 | 손바닥 측면축 회전(손목 굽힘). 사람 팔 자세를 따르기 어려워 넓게 둠 |
| `--max-palm-offset-m` | 0.04 | 손바닥 원점 거리 |
| `--palm-prior-scale` | 1 | 손바닥 비용 강도 (0이면 허용 범위만 적용) |

팔 손목(xArm link6/link7)을 플랜지 뒤 10 cm, 반경 6 cm 원기둥으로 근사해 테이블에서
5 mm 이상 떨어지도록 하는 제약도 함께 걸립니다. 결과의 `palm_*_from_human_*` 항목이
사람 손바닥 대비 편차입니다. 이전 동작은 `--retarget-mode legacy`입니다.

### xArm7 접근·파지·리프트 검증

```bash
python -m sample.validate_xarm7_rh56f1 \
    --lift-dir outputs/custom_grasp/rh56f1_lift \
    --output-dir outputs/custom_grasp/xarm7_lift

# 뷰어 열기
python outputs/custom_grasp/xarm7_lift/replay.py

# MP4 저장
python outputs/custom_grasp/xarm7_lift/replay.py \
    --video outputs/custom_grasp/xarm7_lift/lift.mp4
```

`--approach flowhoi`(기본값)는 FlowHOI 손바닥 이동 방향의 반대쪽 8 cm에 프리그래스프를
두고, 실패하면 수직축 기준 ±50°까지 돌린 방향, 그다음 기존 플랜지 축 방향을 시도합니다.
`--preshape flowhoi`(기본값)는 플랜지와 파지 위치 사이의 남은 거리에 맞춰 FlowHOI
개구도 곡선대로 손가락을 열고(엄지 회전은 유지), 충돌 검사에도 그 손 상태를 씁니다.
통과하지 못하면 열린 손으로 대체합니다. 선택 결과는 `lift_metrics.json`의
`approach_source`, `approach_deviation_deg`, `hand_mode`에, 손 명령 궤적은
`xarm7_path.npz`의 `q_hand_target_rad`에 기록됩니다. 이전 동작은
`--approach flange --preshape open`입니다.

손잡이 없는 컵처럼 +z축 대칭인 물체는 `--object-symmetry z`를 주면, 팔이 닿지 않을 때
파지와 접근 방향 전체를 물체 축 기준 30° 간격으로 돌려 다시 계획합니다. 사용한 각도는
`grasp_axis_rotation_deg`에 기록됩니다. 사전검사(`preflight_xarm7_rh56f1`)도 같은
`--approach`, `--preshape`, `--object-symmetry` 옵션을 받습니다.

### 실기 사전검사

```bash
# 손 명령 범위만 검사
python -m sample.preflight_xarm7_rh56f1 \
    --lift-dir outputs/custom_grasp/rh56f1_lift \
    --output outputs/custom_grasp/hand_preflight.json

# 실측 보정값 입력
cp -n sample/xarm7_preflight.example.json outputs/custom_grasp/measured_xarm7.json
vi outputs/custom_grasp/measured_xarm7.json

# 실측값으로 팔 경로까지 검사
python -m sample.preflight_xarm7_rh56f1 \
    --lift-dir outputs/custom_grasp/rh56f1_lift \
    --calibration outputs/custom_grasp/measured_xarm7.json \
    --output outputs/custom_grasp/preflight.json
```

## Visualization

```bash
python -m visualize.visualize_sequences \
    --folder_path outputs/demo \
    --dataset grab --output_fps 30 --range_max 2
```

For the HOT3D output generated above, use its dataset and frame rate:

```bash
python -m visualize.visualize_sequences \
    --folder_path outputs/demo_hot3d \
    --dataset hot3d --output_fps 15 --range_max 1
```

These commands render MANO hand meshes and the object mesh to mp4.
`--show_scene` can additionally render a scene point cloud for HOT3D.
By default it reads aligned `xyz` / `features` arrays from
`dataset/HOT3D_HANDS/local_scenes_5000/{sequence_name}.npz`; set
`--local_scenes_path` to use another directory. To render per-recording
Concerto point clouds instead, select that source explicitly:

```bash
python -m visualize.visualize_sequences \
    --folder_path outputs/demo_hot3d \
    --dataset hot3d --output_fps 15 --range_max 1 \
    --show_scene --scene_source concerto_grid
```

This mode reads `coords_grid.npy`, `coords_filtered.npy`, and
`features_grid.npy` from
`dataset/HOT3D_HANDS/scene_data/{recording_id}/concerto_features/point_cloud/`,
with the corresponding sequence anchor under `dataset/HOT3D_HANDS/anchors/`.
Neither the local scene files nor these per-recording point clouds are in
the default data package; its precomputed scene features support model
conditioning without these optional visualization inputs.

## Tests

The regression tests cover argument handling and sampler behavior on CPU
without downloading model weights or data:

```bash
python -m unittest discover -s tests -v
```

## TODO List

The current release ships **inference + visualization only**. The following
components are planned for future release:

- [ ] Evaluation code
- [ ] Training code
- [ ] Data preprocessing code

## Citation

If you find this repository useful for your research, please consider citing:

```bibtex
@inproceedings{zeng2026flowhoi,
  title   = {{FlowHOI}: Flow-based Semantics-Grounded Generation of Hand-Object Interactions for Dexterous Robot Manipulation},
  author  = {Zeng, Huajian and Chen, Lingyun and Yang, Jiaqi and Zhang, Yuantai and Shi, Fan and Liu, Peidong and Zuo, Xingxing},
  booktitle = {Advances in Neural Information Processing Systems},
  year    = {2026}
}
```

## License and Acknowledgements

This code is released under the
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license (see
`LICENSE`).

FlowHOI builds on [DiffH2O](https://github.com/diffh2o/diffh2o). Other adapted
code keeps its attribution in the file headers; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for its licences and sources.

The data package carries only data derived for FlowHOI. The GRAB, HOT3D and
MANO / SMPL-X assets it builds on are not redistributed here -- download them
from their own sites as described above, under their own licences.
