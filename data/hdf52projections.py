import yt
import os
import argparse

parser = argparse.ArgumentParser(description="Generating projection plots for designated timestamp range.")
parser.add_argument("-hr", "--hdf5_root", default="./Dataset", type=str, help="input hdf5 directory")
parser.add_argument("-o", "--output_root", default="./Dataset", type=str, help="output directory")
parser.add_argument('-i', '--incr', help='The timestamp increment unit', default=1, type=int)
parser.add_argument('-st', '--start_timestamp', help='Input the starting timestamp', type=int)
parser.add_argument('-et', '--end_timestamp', help='Input the ending timestamp', type=int)

def timestamp2Myr(timestamp):
    return (timestamp - 200) * 0.1 + 191

if __name__ == "__main__":
    args = parser.parse_args()
    for timestamp in range(args.start_timestamp, args.end_timestamp, args.incr):
        filename = f"sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{timestamp}"
        time_Myr = timestamp2Myr(timestamp=timestamp)
        ds = yt.load(os.path.join(args.hdf5_root, filename))
        prj = yt.ProjectionPlot(ds, 'z', 'dens', center = [0, 0, 0] * yt.units.pc)
        prj.annotate_timestamp()
        prj.annotate_scale()

        prj.save(os.path.join(args.output_root, f'{time_Myr}.png'))

# python hdf52projections.py -hr /srv/data/stratbox_simulations/stratbox_particle_runs/bx5/smd132/sn34/pe300/4pc_resume/4pc -st 209 -et 231 -i 1 -o /home/joy0921/Desktop/Dataset/MHD-3DIS/projections