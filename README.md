# Astro-UNETR
- This is a 3D transformer implementation for tracking the superbubbles in magnetohydrodynamics simulation environement. 

## sam2 
- clone from the original [sam2 repo](https://github.com/facebookresearch/sam2).

### Installation
- `conda create -n sam2 python=3.10`
    - In order to install the correct version of dependency packages they required python version 3.10
- `conda install pytorch` 
    - I always had to install it in advance to ensure smooth installation
-  `pip install -e .`

### Checkpoint download
```
cd checkpoints/
./download_checkpoints.sh
```



## TODOs
- Methods
- [ ] Implement UNet models

- Documentation
- [ ] Add teaser figure
