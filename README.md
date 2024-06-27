# Ray Reordering for Hardware-Accelerated Neural Volume Rendering
This repository hosts the training code for the research detailed in the following paper:

```

```

## Requirements
```
sudo apt-get install build-essential git python3-dev python3-pip libopenexr-dev libxi-dev \
                     libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev pybind11-dev libeigen3-dev
```
## Compilation
```
git clone https://github.com/dingjr7/rr-nerf.git
cd rr-nerf
git submodule sync --recursive
git submodule update --init --recursive
```

```
conda env create --file environment.yml
conda activate rr
```

## Demo
Modify dataset_dir to nerf_synthetic dataset.
```
python train_envr.py
```