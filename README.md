# Ray Reordering for Hardware-Accelerated Neural Volume Rendering
This repository hosts the training code for the research detailed in the following paper:

```
@ARTICLE{10574847,
  author={Ding, Junran and He, Yunxiang and Yuan, Binzhe and Yuan, Zhechen and Zhou, Pingqiang and Yu, Jingyi and Lou, Xin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Ray Reordering for Hardware-Accelerated Neural Volume Rendering}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Rendering (computer graphics);Hardware;Image color analysis;Casting;Neural networks;Parallel processing;Interpolation;Neural Volume Rendering (NVR);Ray Reordering;Cache Locality;Hardware Accelerator},
  doi={10.1109/TCSVT.2024.3419761}}
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
