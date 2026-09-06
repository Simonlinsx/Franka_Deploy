# Dexgrasp environments

This project has two deliberately separate Python environments. The lightweight
environment loads RealSense or saved point clouds and visualizes grasp poses. The
official environment runs the AnyDexGrasp neural network. Installing the shell
does not make the AnyDex backend available.

| Profile | Purpose | Target stack |
| --- | --- | --- |
| `visualization-shell` | Capture/load object and scene point clouds, read saved poses, and visualize them with Open3D | Python 3.8+, NumPy, Open3D; OpenCV and RealSense are optional for saved input |
| `official-anydex-inference` | Produce grasp proposals and Inspire-hand scores from the official checkpoint | Python 3.8, PyTorch 1.13.1/cu117, CUDA toolkit 11.7, MinkowskiEngine 0.5.x, NumPy 1.23.x, `pointnet2`, and `knn` |

## Check without changing the environment

The checker imports installed modules and inspects checkpoint paths. It never
installs packages, builds extensions, downloads weights, opens the camera, or
connects to robot hardware.

```bash
cd /home/qiaoguanren/code/franka

python dexgrasp/scripts/check_anydex_env.py --profile shell
python dexgrasp/scripts/check_anydex_env.py --profile official
python dexgrasp/scripts/check_anydex_env.py --profile all
```

An exit code of zero means every required check in the selected profile passed.
Warnings identify optional components. The official profile treats an unavailable
GPU as a failure because the bundled `pointnet2` extension has no CPU inference
fallback. If a managed sandbox hides `/dev/nvidia*`, rerun the check in the normal
host terminal before diagnosing the NVIDIA driver.

The existing `dynamic` environment is useful for the shell:

```bash
/home/qiaoguanren/anaconda3/envs/dynamic/bin/python \
  dexgrasp/scripts/check_anydex_env.py --profile shell
```

At the time of the initial audit it contained Python 3.10, PyTorch 2.5/cu121,
Open3D, OpenCV, and pyrealsense2. It did not contain MinkowskiEngine,
graspnetAPI, or the AnyDex CUDA extensions, so it is not the official inference
environment. The repository `.venv` can also run the visualization shell but has
no Torch installation.

## Official reproduction environment

The upstream checkout at `third_party/AnyDexGrasp` explicitly specifies Python
3.8, PyTorch 1.13 with CUDA 11.7, and MinkowskiEngine 0.5. Use a dedicated
environment so older sparse-convolution dependencies do not disturb the existing
camera and point-cloud stack.

The following is an installation recipe, not an action performed by the checker:

```bash
conda create -n anydex python=3.8
conda activate anydex

conda install pytorch==1.13.1 torchvision==0.14.1 \
  pytorch-cuda=11.7 -c pytorch -c nvidia

# A runtime-only CUDA package is insufficient: nvcc 11.7 is needed to compile
# MinkowskiEngine, pointnet2, and knn. Install a complete 11.7 toolkit using the
# package/channel available on this machine, then verify it before building.
conda install cuda-toolkit=11.7 -c nvidia
conda install openblas-devel ninja -c anaconda

python -m pip install -r dexgrasp/requirements-official.txt

export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST=8.6  # RTX 3060
export MAX_JOBS=2               # recommended by the AnyDex README for ME builds

python -c 'import torch; print(torch.__version__, torch.version.cuda)'
nvcc --version
```

Both commands must report CUDA 11.7 before compiling extensions. Build
MinkowskiEngine 0.5.x from its pinned source revision with OpenBLAS and forced
CUDA support. Then install the bundled extensions from the AnyDex checkout:

```bash
cd /home/qiaoguanren/code/franka/dexgrasp/third_party/AnyDexGrasp
python -m pip install -v --no-build-isolation ./knn
python -m pip install -v --no-build-isolation ./pointnet2
```

Build `knn` only where `torch.cuda.is_available()` is true. Its upstream
`setup.py` selects a CUDA build using that runtime check and otherwise silently
creates a CPU-only extension. For this visualization-only milestone, do not
install `ur_toolbox`, UR drivers, Allegro controllers, or other robot-control
dependencies.

On this workstation, `/usr/bin/cmake` is usable. A user-local `cmake` entry was
observed to fail because its Python package was missing; prefer `/usr/bin/cmake`
if a source build unexpectedly invokes the broken wrapper.

## Why Torch 2.5/cu121 is not the official backend

The AnyDex Python model code still uses APIs available in Torch 2.5, and a
read-only syntax check of the bundled `pointnet2` and `knn` C++ translation units
against the installed Torch 2.5 headers succeeded. That does not establish a
working backend: upstream MinkowskiEngine 0.5 is not a supported drop-in build
for Torch 2.5 and CUDA 12.1. Using a community Torch-2/CUDA-12 fork would create a
separate port that must pin its revision, compile every CUDA source, load the
official checkpoint, and pass numerical inference tests. It should not be called
an official reproduction.

## Required model assets

The Git checkout contains no model weights. This bridge keeps downloaded assets
outside the vendored checkout. The minimal downloader writes one representation
checkpoint and eight Inspire decision heads to:

```text
dexgrasp/weights/
└── logs/
    ├── model/
    │   ├── checkpoint.tar.18
    │   └── inspire_model/obj140/480/<class>/*.pth
```

`checkpoint.tar.18` supplies the representation network's `model_state_dict`.
The Inspire `.pth` files score dexterous grasp type and depth; they are not needed
only when inspecting two-finger proposals. Dataset files under
`graspnet_v1_newformat` are needed for training/evaluation, not for inference on
an already prepared object point cloud.

```bash
python -m pip install gdown
python dexgrasp/scripts/download_official_weights.py
```
