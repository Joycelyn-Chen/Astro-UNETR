import os
import yt
import argparse

parser = argparse.ArgumentParser(description="Extracting center sliceplot from HDF5")
parser.add_argument("--hdf5_root", default="./Dataset", type=str, help="input directory")
parser.add_argument("--output_root", default="./Dataset", type=str, help="output directory")
parser.add_argument('-st', '--start_timestamp', help='Input the starting timestamp', type=int)
parser.add_argument('-et', '--end_timestamp', help='Input the ending timestamp', type=int)
parser.add_argument('-i', '--interval', help='Timestamp interval', default = 10, type=int)
parser.add_argument('-cz', '--center_z', help='Center z coordinate to be sliced through', default = 171, type=int)

# python hdf52sliceplot.py --output_root /home/joy0921/Desktop/Dataset/MHD-3DIS/sliceplots --hdf5_root /srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc -st 380 -et 670 -i 10 -cz 171
def timestamp2time_Myr(timestamp):
    return (timestamp - 200) * 0.1 + 191

def pixel2pc(px):
    return int((px * 1000) / 256)

def main(args):
    for slice_timestamp in range(args.start_timestamp, args.end_timestamp + 1, args.interval):        # 100 timestamp = 10 Myr
        slice_time_Myr = timestamp2time_Myr(slice_timestamp)
        slice_filename = f"sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{slice_timestamp}"
        slice_ds = yt.load(os.path.join(args.hdf5_root, slice_filename))
        slp = yt.SlicePlot(slice_ds, 'z', 'dens', center = [0, 0, args.center_z] * yt.units.pc)
        slp.annotate_timestamp(text_args={'size': 40})
        slp.annotate_scale(text_args={'size': 40})
        slp.save(os.path.join(args.output_root, f'{slice_time_Myr}.png'))
    
    print(f"Done! Plots saved at {args.output_root}")


if __name__ == "__main__":  
    args = parser.parse_args()
    main(args)

