# Astro-UNETR
- This is a 3D transformer implementation for tracking the superbubbles in magnetohydrodynamics simulation environement. 

## Backbone
### `swin-unetr`
- A transformer based unet segmentation model, originally used for 

### `unet-3d`


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
- Documentation
- [ ] Add teaser figure

---------------------------------------------------------------------------

# `Elephant` command
```
python analysis/SB230/wholeCube_SN_target_k3d.py -h "/srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc" -m /home/joy0921/Desktop/Dataset/MHD-3DIS/SB_tracks/SB450_1 -st 450 -et 510 -k /home/joy0921/Desktop/Dataset/MHD-3DIS/k3d_html -i 10


python vel3d_SN_visualization.py -hr /srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc  -st 380 -et 380 -i 10 -k /home/joy0921/Desktop/Dataset/MHD-3DIS/htmls
    
    
python post-process/wholeCube_SN_target_k3d.py -hr /srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc -m ~/Desktop/Dataset/MHD-3DIS/SB_tracks/SN_20915 -k /UBC-O/joy0921/Desktop/Dataset/MHD-3DIS/htmls/SN_20915 -st 209 -et 231 -i 1    


python rescale_mask_10002256.py --input_root /UBC-O/joy0921/Desktop/Dataset/VOS_output/astro_0219 --output_root /UBC-O/joy0921/Desktop/Dataset/MHD-3DIS/SB_tracks/SN_20915

python add_black_masks.py --mask_root /UBC-O/joy0921/Desktop/Dataset/MHD-3DIS/SB_tracks/SN_20915

python img-flip-xy.py --input_root /UBC-O/joy0921/Desktop/Dataset/MHD-3DIS/SB_tracks/SN_20617
```
# `compute2.idsl` command
```
python remove-unwanted-masks.py --data_dir /home/joy0921/Desktop/Dataset/MHD-3DIS/SB_tracks/230/410 --start_z 0 --end_z 223

python instance-seg-point-prompt.py --sam2_root /home/joy0921/Desktop/sam2 --data_dir /home/joy0921/Desktop/Dataset/MHD-3DIS --timestamp 520

python slice_da_chimney.py -hr /home/joy0921/Desktop/Dataset/MHD-3DIS/hdf5 -cz 170 -t 420 -o /UBC-O/joy0921/Desktop/Dataset/MHD-3DIS/chimneys
```