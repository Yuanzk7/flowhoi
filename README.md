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

### Stage 1 기준 결과 저장하기

GRAB 머그컵 데모로 텍스트와 BPS를 입력으로 쓰는 Stage 1의 기준 결과를 저장합니다.

```bash
python -m sample.sample_2stage \
    --model_path save/grab_full/model000200000.pt \
    --grasp_model_path save/egodex_grasp/model000500000.pt \
    --split_set demo --num_samples 1 --seed 42 \
    --stage1_only --output_dir outputs/stage1_grab_mug
```

같은 조건으로 Stage 2까지 실행하려면 `--save_stage1`을 사용합니다. 위 예제는
포즈 가이던스와 인페인팅을 끈 설정입니다. 다른 Stage 1 구현과 결과를 비교할 때는
체크포인트, 샘플링 설정, BPS, T5 특성, 초기 노이즈를 동일하게 맞춥니다.

배치와 반복마다 `stage1_*.npz` 파일이 하나씩 저장되며, `np.load`로 열 수 있습니다.
`motion`과 `normalized_motion`의 배열 크기는 모두 `[B, 50, 108]`입니다.
`motion`은 `std`와 `mean`으로 역정규화한 값이며 역랜덤투영은 생략합니다.
`object_bps`, `text`, `t5_emb`, `t5_mask`, 샘플러 입력 형태
`[B, 108, 1, 50]`의 초기 `noise`, `args_json`도 함께 저장됩니다.
데모는 좌우 반전된 GRAB 시퀀스이므로 기준 비교에는 NPZ에 저장된 반전 `mug` BPS를 사용합니다.

`reference_motion`은 파지 구간에 대응하는 데이터셋의 동작입니다. 물체의 이동은
채널 `108:111`, 6D 회전은 채널 `111:117`에 저장됩니다. GRAB에서 손 위치의
원점은 파지 프레임의 물체 위치이며, 물체 회전은 그대로 사용합니다. 파지 모습을
확인하려면 물체를 기준 동작의 49번 프레임 자세에 배치합니다. 생성 결과의 108개
채널에는 양손의 손목 위치, MANO PCA 값, 손목 회전, 관절별 물체 거리 예측값이
들어 있습니다. 채널 인덱스는 `REPRESENTATION_IDCS`에 정의되어 있습니다.

### SAM3D 메시를 Stage 1 입력으로 변환하기

저장소 루트에서 메시 변환을 실행합니다. 임의 단위로 생성한 SAM3D 메시에는
실제 물체의 가장 긴 치수를 미터 단위로 입력합니다. 예를 들어 12 cm라면
`--longest-extent-m 0.12`를 사용합니다.

```bash
python -m utils.mesh_to_bps --mesh /path/to/sam3d_raw.obj \
    --longest-extent-m 0.12 --output-dir outputs/custom_object_bps
```

메시의 위쪽을 +Z로 맞춥니다. 필요하면 `--rotate-xyz-deg RX RY RZ`로 고정된
X, Y, Z축을 기준으로 순서대로 회전합니다. 크기는 회전 후 바운딩 박스에서
가장 긴 축을 기준으로 맞춥니다. 이미 미터 단위인 메시에는 `--meters-per-unit 1`을
사용합니다. 기본 원점은 바운딩 박스 중심이며, GRAB 메시처럼 원점이 정해져
있으면 `--center keep`으로 보존합니다. 좌우 반전된 GRAB 기준을 재현하려면
`--mirror-x`를 사용합니다.

`object_bps.npy`는 `[1, 1024, 3]` 크기의 float32 배열입니다. 각 벡터는 공개된
기준점에서 메시의 가장 가까운 꼭짓점으로 향합니다. 시각화와 접촉 검사에는
`object_mesh.ply`를 사용합니다. `mesh_transform.json`에는 입력 꼭짓점을 물체
좌표계로 옮기는 4×4 변환이 저장됩니다. 메시나 변환 설정이 달라지면 새 출력
디렉터리를 지정합니다.

### Stage 1 후보 생성 및 오른손 파지 선별

준비한 메시와 BPS로 파지 후보를 생성합니다. 시드마다 50프레임 궤적 하나를
만듭니다. `--mesh`만 지정하면 그 메시에서 BPS를 계산합니다. 결과 NPZ에는
양손의 궤적, BPS, 시드가 저장됩니다.

```bash
python -m sample.sample_grasp \
    --text 'the individual grasps the bottle.' \
    --mesh outputs/custom_object_bps/object_mesh.ply \
    --bps outputs/custom_object_bps/object_bps.npy \
    --num-samples 10 --seed 42 --output outputs/custom_grasp/candidates.npz
```

독립 샘플러와 공개 샘플러의 결과를 비교하려면 포즈 가이던스와 인페인팅을 끄고
저장한 Stage 1 NPZ를 `--reference`로 지정합니다. 저장된 정규화 통계,
텍스트 특성, 초기 노이즈를 재사용합니다.

```bash
python -m sample.sample_grasp \
    --reference outputs/stage1_grab_mug/stage1_test_demo_b0000_r0000.npz \
    --device cpu --output outputs/stage1_grab_mug/replay.npz
```

파지 선별에는 MANO 모델과 Open3D가 필요합니다. Open3D는
`python -m pip install open3d`로 설치합니다. 물체가 바닥면으로 테이블 위에
놓인 경우 `--object-on-table`을 사용합니다. 동작 좌표계에서 측정한 테이블
높이는 `--table-z`로 지정할 수 있습니다. 물체의 기본 자세는 항등 변환입니다.
다른 자세로 배치하려면 이동과 6D 회전을 합친 9채널 `.npy`를 `--object-pose`로
지정합니다.

```bash
python -m sample.select_grasp \
    --candidates outputs/custom_grasp/candidates.npz \
    --mesh outputs/custom_object_bps/object_mesh.ply \
    --object-on-table --output-dir outputs/custom_grasp/selection
```

재구성한 손 표면과 닫힌 물체 메시 사이의 부호 있는 거리를 계산합니다. 마지막
3프레임의 손가락 접촉, 전체 50프레임의 오른손 최대 침투 깊이와 왼손 최소
여유를 검사합니다. 테이블 높이를 지정하면 손과 테이블 사이의 여유도 검사합니다.
GRAB 데모 두 개를 기준으로 비교하며, 여러 손가락이 물체에 더 가깝게 닿는
데모에서 기본 접촉 반경을 정합니다. 거리 임계값 세 가지를 모두 입력해 이
보정을 생략할 수도 있습니다. 선별은 실제로 계산한 기하학적 거리를 사용합니다.
모델이 예측한 관절 거리 21개와 재계산한 거리의 평균 절대 차이는 보고서에
기록합니다. [Open3D 부호 거리 API](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html#open3d.t.geometry.RaycastingScene.compute_signed_distance)

첫 후보들이 통과하지 않으면 오른손을 명시한 프롬프트를 사용하거나, 준비한
원래 메시와 `--mirror-x`로 `sample_grasp`를 다시 실행합니다. 반전 모드에서는
반전된 BPS로 샘플링한 뒤, 선별할 때 양손을 반전하고 서로 바꿔 원래 물체
좌표계에 맞춥니다. `--candidates`에 NPZ를 여러 개 전달하면 두 실행의 후보를
함께 평가합니다.

`report.json`에는 후보별 점수, 임계값, 후보 간 다양성이 저장됩니다. 후보가
통과하면 `right_hand_grasp.npz`에 50프레임의 손목 위치(미터), 손목 회전
행렬, MANO 관절의 로컬 축-각 회전(라디안), 3D 관절과 손끝, 메시 꼭짓점이
저장됩니다. 손끝 순서는 엄지, 검지, 중지, 약지, 새끼입니다. 통과한 후보가
없으면 보고서를 저장하고 종료 코드 1을 반환합니다.

컵을 세운 상태에서 생성된 손이 컵에 닿지 않으면, 컵 외벽을 기준으로 접근
자세를 보정한 뒤 다시 선별합니다.

```bash
python -m sample.fit_cup_grasp \
    --candidates outputs/custom_grasp/candidates.npz --index 0 \
    --mesh outputs/custom_object_bps/object_mesh.ply \
    --output outputs/custom_grasp/geometry_fitted.npz
```

이 후처리는 FlowHOI 후보의 시작 손목 위치에서 컵 외벽에 맞춘 오른손 자세로
접근 경로를 만듭니다. 전체 50프레임의 양손 메시를 검사하고, 손이 컵 외곽
표면을 뚫고 들어가는 깊이를 제한합니다. 마지막 3프레임에서는 컵 둘레를
기준으로 엄지와 다른 손가락의 접촉 방향이 최소 120° 떨어져 있어야 합니다.
결과 NPZ를 선별기의 `--candidates`로 전달합니다. 기본 테이블 높이는 컵
바닥입니다.

### MuJoCo에서 Stage 1 궤적 보기

FlowHOI 환경에서 장면을 내보냅니다. 보고서에 선택된 후보가 있으면 그 후보를
사용합니다. 선택된 후보가 없으면 필요한 오른손 접촉 반경이 가장 작은 후보를
내보냅니다.

```bash
python -m visualize.export_grasp_mujoco \
    --report outputs/custom_grasp/selection/report.json \
    --output-dir outputs/custom_grasp/mujoco
```

`mujoco`가 설치된 Python 환경에서 재생합니다. 스페이스바로 일시정지하고
좌우 방향키로 프레임을 이동합니다.

```bash
python outputs/custom_grasp/mujoco/replay.py
```

Pillow와 FFmpeg를 설치하고 `--video outputs/custom_grasp/mujoco/preview.mp4`를
붙이면 MP4를 저장합니다. 최종 파지를 확대하려면 `--detail`을 사용합니다.
장면은 기록된 MANO와 물체 자세를 미터 단위로 재생하며, 격자는 선별에 사용한
테이블 높이를 표시합니다. 개별 후보를 보려면 `--candidates`, `--mesh`,
`--index`를 지정합니다.

### RH56F1 리타게팅 및 컵 리프트 검증

MuJoCo, NumPy, SciPy, trimesh가 설치된 Python 환경을 사용합니다. 저장소
루트에서 선별한 컵 외벽 파지와 새 출력 디렉터리를 지정해 실행합니다.

```bash
python -m sample.validate_rh56f1 \
    --grasp outputs/custom_grasp/selection/right_hand_grasp.npz \
    --output-dir outputs/custom_grasp/rh56f1_lift
```

기본 파지 프레임은 49번입니다. 저장소에 포함된 `sample/rh56f1_retarget.py`와
`third_party/robot_models/rh56f1`의 모델을 사용합니다. 다른 구현이나 모델은
`--retarget-module`과 `--robot-mjcf`로 지정합니다. 선별된 NPZ는 +Z로 세워진
미터 단위 컵 메시를 참조해야 합니다. 메시를 옮겼다면 `--mesh`로 경로를
지정합니다.

MANO 손끝에서 시작해 RH56F1의 손가락 형상과 연동 관절이 도달할 수 있는
마주 보는 접점을 찾습니다. `retarget_frame49.npz`에는 컵 좌표계 기준
플랜지 자세, 관절각 6개(라디안), 열림·닫힘 명령, 접점 좌표가 저장됩니다.

`scene.xml`에는 로봇 손, 테이블, 자유 관절이 있는 컵이 들어 있습니다.
위치 서보 6개로 손가락을 닫고, 손목 방향을 유지한 채 수직 서보로 손목을
7.5 cm 올립니다. MuJoCo 접촉 계산에는 컵의 볼록 외곽 형상을, 렌더링에는
원본 메시를 사용합니다. 컵 질량 기본값은 60 g, 미끄럼 마찰계수 기본값은
0.65이며 `--cup-mass`와 `--friction`으로 조정합니다.

동작은 `lift_replay.npz`, 검사 결과는 `lift_metrics.json`에 저장합니다.
컵 바닥이 테이블에서 5 cm 이상 떨어진 상태로 2초 동안 유지되어야 통과합니다.
양쪽 손가락의 접촉력, 컵의 기울기와 미끄러짐, 손과 테이블 사이 여유도
검사합니다. 정적 파지와 열린 손 자세를 찾은 뒤 리프트에 실패하면 종료 코드 1을
반환합니다.

```bash
python outputs/custom_grasp/rh56f1_lift/replay.py
python outputs/custom_grasp/rh56f1_lift/replay.py \
    --video outputs/custom_grasp/rh56f1_lift/lift.mp4
```

MP4 저장에는 Pillow와 FFmpeg가 필요합니다.

### xArm7 접근·파지·리프트 검증

앞 단계에서 리프트 검사를 통과한 RH56F1 출력 디렉터리를 사용합니다.
플래너는 `third_party/robot_models/xarm7`에 포함된 확장 URDF, 메시, 충돌
검사 제외 쌍을 읽습니다. 다른 모델을 쓰려면 `--xarm-root`로 호환 모델
디렉터리나 xArm ROS 2 체크아웃을 지정합니다. Xacro 확장이 필요한 체크아웃은
Xacro를 설치하거나 `--xacro`로 실행 파일을 지정합니다. 이미 확장된 별도
URDF는 `--arm-urdf`로 전달합니다.

```bash
python -m sample.validate_xarm7_rh56f1 \
    --lift-dir outputs/custom_grasp/rh56f1_lift \
    --output-dir outputs/custom_grasp/xarm7_lift
```

홈 자세에서 파지 전 대기 자세까지 관절 공간 경로를 만들고, 이후 파지 위치와
7.5 cm 위의 목표까지 직교 좌표계 경로를 계산합니다. 플래너는 각 구간의
충돌을 검사하고, 접근 도중 컵을 피할 수 있도록 접근 방향을 탐색합니다.
MuJoCo 관절 서보가 xArm7을 움직이고 RH56F1을 닫아 자유 관절이 있는 컵을
들어 올립니다. 목표 파지 자세 도달, 예상 밖의 접촉, 손을 닫기 전 컵의
이동량, 컵 바닥이 테이블에서 5 cm 이상 떨어진 상태로 2초간 유지되는지를
검사합니다. 시뮬레이션의 베이스 위치는 `--base-xy`(미터), 홈 자세는
`--home`(라디안)으로 바꿀 수 있습니다.

`xarm7_path.npz`에는 목표 관절 궤적과 시뮬레이션에서 측정한 관절 궤적,
베이스·파지 전·파지·리프트 변환이 저장됩니다. `lift_metrics.json`에는
충돌 여유와 리프트 검사 결과가 들어 있습니다. `replay.py`를 실행하면
MuJoCo 뷰어가 열리고, 아래처럼 실행하면 MP4를 저장합니다. 컵을 화면
중심에 두려면 `--detail`을 추가합니다.

```bash
python outputs/custom_grasp/xarm7_lift/replay.py \
    --video outputs/custom_grasp/xarm7_lift/lift.mp4
```

### xArm7·RH56F1 실기 사전검사

오프라인 사전검사는 RH56F1 리프트 결과를 입력받아 손 명령이 허용 범위에
드는지 검사하고 JSON 보고서를 저장합니다. 실측 보정값이 있으면 xArm7의
역기구학(IK)과 이동·접근·닫기·리프트 구간에서 샘플링한 충돌 여유도 검사합니다.

```bash
python -m sample.preflight_xarm7_rh56f1 \
    --lift-dir outputs/custom_grasp/rh56f1_lift \
    --calibration path/to/measured_xarm7.json \
    --output outputs/custom_grasp/preflight.json
```

[`sample/xarm7_preflight.example.json`](sample/xarm7_preflight.example.json)을
복사해 실측값을 채웁니다. 모든 변환은 4×4 행렬이고, `T_A_B`는 B 좌표계의
점을 A 좌표계로 옮깁니다. 이동 성분의 단위는 미터입니다.

- `T_base_object`는 컵 메시 좌표계에서 실제 로봇 베이스 좌표계로의 변환입니다.
- `T_base_table`은 실제 테이블 상판 중심의 자세입니다. +Z는 위쪽이며
  `table_size_m`에는 상판의 전체 길이, 너비, 두께를 입력합니다.
- `T_flange_hand`는 RH56F1 URDF의 루트 좌표계에서 xArm 플랜지 좌표계로의
  변환입니다. 플래너는 실측 장착 자세와 시뮬레이션 장착 자세의 차이를 보정합니다.
- `T_flange_tcp`는 컨트롤러의 TCP 오프셋입니다. TCP가 플랜지와 일치하면
  항등 행렬을 사용합니다.
- `start_joints_rad`에는 실측한 팔의 시작 관절각을 J1부터 J7 순서로 입력합니다.
- `min_clearance_m`은 경로 샘플에서 확보해야 할 최소 여유입니다. 정합,
  도구 보정, 경로 추종의 오차를 고려해 정합니다.
- `rh56_raw_limits`는 선택 항목입니다. 실측한 `min`, `max` 벡터의 순서는
  새끼, 약지, 중지, 검지, 엄지 굽힘, 엄지 회전입니다. 값이 `null`이면
  `xarm-teleop` 드라이버의 기본 허용 범위를 사용합니다.

`--calibration`을 생략하면 손 명령을 검사하고 필요한 보정 항목 목록을 출력합니다.
보정값이 모두 있으면 저장소에 포함된 xArm7 모델로 팔 경로를 검사합니다.
다른 모델을 지정하는 옵션은 위 xArm7 검증과 같습니다. 보고서에는 손 명령값,
플랜지와 TCP의 파지 목표, 가장 가까운 충돌 형상 쌍, 계획이 완성된 경우 구간별
테이블 여유가 포함됩니다. 리프트 여유는 컵이 수직 이동을 따라간다고 가정해
계산합니다. 검사 실패 또는 입력 부족 시 종료 코드 1을 반환합니다. 이 명령은
장비 연결 없이 실행되며 보고서에 `hardware_validated=false`를 기록합니다.

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
