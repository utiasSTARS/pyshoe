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
thresh_list =  [[0.3, 0.55, 0.8], [1e7, 8.5e7, 35e7], [[1e7, 35e7,1e7]],[0]]
W_list = [5, 5, 5, 0]
error_logger = HallwayErrorLogger(modes, subjects, det_list, thresh_list)

load_traj=True  #set to false to recompute the trajectories, or true to reload the previously saves trajectories (much faster to reload)
if load_traj==True:
    stored_trajectories = sio.loadmat("results/stored_hallway_trajectories.mat")
for f in sorted(glob.glob('{}*/*/*/*.mat'.format(source_dir))):
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
    
    for i in range(0, len(det_list)):   #Iterate through detector list
        for j in range(0, len(thresh_list[i])): #iterate through threshold list
            if load_traj != True:
                zv = ins.Localizer.compute_zv_lrt(W=W_list[i], G=thresh_list[i][j], detector=det_list[i])
                x = ins.baseline(zv=zv)
                saved_trajectories["{}_{}_{}_det_{}_G_{}".format(trial_type,person, folder, det_list[i], thresh_list[i][j])] = x
            else:
                x = stored_trajectories["{}_{}_{}_det_{}_G_{}".format(trial_type,person, folder, det_list[i], thresh_list[i][j])]
            x, gt = align_plots(x,gt, dist=0.8, use_totstat=True, align_idx=trigger_ind[1]) #rotate data
            ###Calculate ARMSE between estimate and Vicon
            armse_3d = compute_error(x[trigger_ind], gt, '3d')
            error_logger.update(armse_3d, trial_type, person, det_list[i], j)
            print("ARMSE for {}: {}".format(det_list[i], armse_3d))
            trial_stats.append(armse_3d)
    stats.append(trial_stats)

###Process the results and save to csv files (saves a raw csv, and a processed csv that reproduces the paper results) 
error_logger.process_results()       
stats_header = ['Motion', 'Subject', 'Trial']
for i in range(0, len(det_list)):
    for j in range(0, len(thresh_list[i])):
        stats_header.append("{}_G={}_3d_error".format(det_list[i], thresh_list[i][j]))

csv_filename = 'results/hallway_results_raw.csv'
with open(csv_filename, "w") as f:
    writer = csv.writer(f)
    writer.writerow(stats_header)
    writer.writerows(stats)

if load_traj != True:
    sio.savemat("results/stored_hallway_trajectories.mat", saved_trajectories)