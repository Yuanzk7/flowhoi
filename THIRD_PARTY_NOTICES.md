# Third-party code and assets

FlowHOI's own code uses the license in `LICENSE`. The following adapted code
retains its upstream notices and license terms:

| Source | Used in FlowHOI | Included license |
| --- | --- | --- |
| [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion) | `diffusion/nn.py` | [MIT](third_party/licenses/guided-diffusion-MIT.txt) |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | `utils/hfargparse.py` | [Apache 2.0](third_party/licenses/transformers-Apache-2.0.txt) |
| [Text-to-Motion](https://github.com/EricGuo5513/text-to-motion) | `data_loaders/humanml/` | [MIT](third_party/licenses/text-to-motion-MIT.txt) |
| [PyTorch3D](https://github.com/facebookresearch/pytorch3d) | `utils/rotation_conversions.py`, via ACTOR | [BSD](utils/PYTORCH3D_LICENSE) |
| [INSPIRE-ROBOTS RH56F1 description](https://github.com/KUKU-Robot-Lab/urdf) | `third_party/robot_models/rh56f1/` | [BSD 3-Clause](third_party/robot_models/rh56f1/LICENSE) |
| [UFACTORY xArm ROS 2](https://github.com/xArm-Developer/xarm_ros2/tree/62936f7ea1846a85f7350de2c4c18f39e6d19715) | `third_party/robot_models/xarm7/` | [BSD 3-Clause](third_party/robot_models/xarm7/LICENSE) |

The repository also builds on [DiffH2O](https://github.com/diffh2o/diffh2o).
Existing copyright notices in adapted files are retained. FlowHOI modifies
these components for hand-object data, two-stage flow matching, configuration
loading, and inference.

The RH56F1 URDF and meshes came from the vendor description used by the
local `text2hoi_spider/converter/robots/rh56f1` model. Its MuJoCo XML uses
the local pinky joint names and repository-relative mesh paths.
`sample/rh56f1_retarget.py` came from `kang/lab/robot_execution` at
`eb7620f`, with the URDF path adjusted for this repository. The lab checkout
contained no separate license declaration for that module.

The bundled arm URDF was expanded from `xarm_device.urdf.xacro` with seven
joints, limited ranges, STL meshes, and the gripper disabled. It retains
the link, joint, and material definitions. The SRDF retains collision
exclusions for the arm links, and the referenced STL files are included.

Installed dependencies retain their own licenses. T5 weights, checkpoints,
and example data are downloaded separately. MANO / SMPL-X, GRAB meshes and
subject templates, and HOT3D object models must be obtained from their
respective providers under their own terms, as described in `README.md`.
