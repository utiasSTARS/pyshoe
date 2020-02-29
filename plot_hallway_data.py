import numpy as np
from ins_tools.util import *
import ins_tools.visualize as visualize
from ins_tools.INS import INS
import csv
import glob
import scipy.io as sio

source_dir = "data/hallway/"
stats = []
saved_trajectories = {}

        ###add custom detector and its zv output to lists:
modes = ['comb', 'run', 'walk']
subjects = ['0', '1', '2', '3', '4']
det_list = ['ared', 'shoe', 'adaptive', 'lstm']
thresh_list =  [0.55, 8.5e7, [1e7, 35e7,1e7], 0] #zero-velocity thresholds for various detectors (lstm has no threshold)
W_list = [5, 5, 5, 0]   #window size used for classical detectors (LSTM requires no window size)
legend = ['ARED', 'SHOE', 'Adaptive', 'LSTM', 'Ground Truth'] #used for plotting results.

load_traj=True  #set to false to recompute the trajectories, or true to reload the previously saves trajectories (much faster to reload)
if load_traj==True:
    stored_trajectories = sio.loadmat("results/stored_hallway_trajectories.mat")
for f in sorted(glob.glob('{}*/*/*/*.mat'.format(source_dir))):
    traj_list = []
    trigger_ind_list = []
    trial_name = f.replace(source_dir,'').replace('/processed_data.mat','')
    print(trial_name)
    trial_type, person, folder = trial_name.split('/')
    trial_stats = [trial_type, person, folder]
    
    data = sio.loadmat(f)
    imu = data['imu']
    ts = data['ts'][0]
    gt = data['gt']
    trigger_ind = data['gt_idx'][0]
    ins = INS(imu, sigma_a = 0.00098, sigma_w = 8.7266463e-5, T=1.0/200) #microstrain
    ###Estimate trajectory for each zv detector
    for i in range(0, len(det_list)):
        if load_traj !=True:
            zv = ins.Localizer.compute_zv_lrt(W=W_list[i], G=thresh_list[i], detector=det_list[i])
            x = ins.baseline(zv=zv)
        else:
            x = stored_trajectories["{}_{}_{}_det_{}_G_{}".format(trial_type,person, folder, det_list[i], thresh_list[i])]
        x, gt = align_plots(x,gt, dist=0.8, use_totstat=True, align_idx=trigger_ind[1]) #rotate data    
        if trial_type == 'run':
            ind = 6#3 for halfway
        else:
            ind= 14 #7 for halfway
        traj_list.append(x[0:(trigger_ind[ind]+1)])

    traj_list.append(gt)  
    if trial_type == 'comb':
        trial = 'Mixed-Motion Trial'
    if trial_type == 'run':
        trial = 'Running Trial'
    if trial_type == 'walk':
        trial = 'Walking Trial'
#    visualize.plot_topdown(traj_list, trigger_ind = list(trigger_ind[0:ind+1]), gt_method='sparse', title='{} (Top-Down View)'.format(trial), save_dir='results/figs/hallway/{}_{}_{}.eps'.format(trial_type, person, folder), legend=legend)
#    visualize.plot_vertical(ts[0:(trigger_ind[ind]+1)], traj_list, trigger_ind = list(trigger_ind[0:ind+1]), title='{} (Vertical View)'.format(trial), save_dir='results/figs/hallway/{}_{}_{}_vert.eps'.format(trial_type, person, folder), legend=legend)
    visualize.plot_topdown(traj_list, trigger_ind = list(trigger_ind[0:ind+1]), gt_method='sparse', title=None, save_dir='results/figs/hallway/{}_{}_{}.eps'.format(trial_type, person, folder), legend=legend)
    visualize.plot_vertical(ts[0:(trigger_ind[ind]+1)], traj_list, trigger_ind = list(trigger_ind[0:ind+1]), title=None, save_dir='results/figs/hallway/{}_{}_{}_vert.eps'.format(trial_type, person, folder), legend=legend)