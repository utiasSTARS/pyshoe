import numpy as np
from ins_tools.util import *
import ins_tools.visualize as visualize
from ins_tools.INS import INS
import csv
import glob
import scipy.io as sio

source_dir = "data/stairs/test/"
stats = []
saved_trajectories = {}
det_list = ['ared', 'shoe', 'adaptive', 'lstm']
thresh_list =  [0.55, 8.5e7, [1e7, 35e7, 1e7], 0]
W_list = [5, 5, 5, 0]
flight_list = ['2', '4', '6', '8']
legend = ['ARED', 'SHOE', 'Adaptive', 'LSTM', 'Floor Level']
topdown_legend = ['ARED', 'SHOE', 'Adaptive', 'LSTM']

load_traj=True #set to false to recompute the trajectories, or true to reload the previously saves trajectories (much faster to reload)
if load_traj==True:
    stored_trajectories = sio.loadmat("results/stored_stair_trajectories.mat")
    
for f in glob.glob('{}/*/processed_data.mat'.format(source_dir),recursive=True):
    traj_list = []
    trial_name = f.replace(source_dir,'').replace('/processed_data.mat','')
    print(trial_name)
    trial_stats = [trial_name]
    data = sio.loadmat(f)
    imu = data['imu']
    ts = data['ts'][0]
    gt = data['gt'][0]
    trigger_ind = data['gt_idx'][0]
        ###add custom detector and its zv output to lists:

    ins = INS(imu, sigma_a = 0.00098, sigma_w = 8.7266463e-5, T=1.0/200) #microstrain

    ###Estimate trajectory for each zv detector
    for i in range(0, len(det_list)):
        if load_traj!=True:
            zv = ins.Localizer.compute_zv_lrt(W=W_list[i], G=thresh_list[i], detector=det_list[i])
            x = ins.baseline(zv=zv)
        else:
            x = stored_trajectories["{}_det_{}_G_{}".format(trial_name, det_list[i], thresh_list[i])]
        traj_list.append(x[0:(trigger_ind[-1]+1)])
    
    traj_list_topdown = traj_list.copy()
    traj_list_topdown.append(0)
    visualize.plot_topdown(traj_list_topdown, trigger_ind = None, gt_method='none', title='Stair-Climbing Trial (Top-Down View)', save_dir='results/figs/stairs/{}_topdown.eps'.format(trial_name), legend=topdown_legend)                
    visualize.plot_stairs(ts[0:(trigger_ind[-1]+1)], traj_list, gt, trigger_ind=list(trigger_ind), title='Stair Climbing Trial (Vertical Plane)', legend=legend, save_dir='results/figs/stairs/{}.eps'.format(trial_name))
