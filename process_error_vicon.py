import numpy as np
from ins_tools.util import *
from ins_tools import visualize
from ins_tools.INS import INS
import csv
import glob
import scipy.io as sio

folder = "data/vicon/processed"
stats = []
saved_trajectories = {}
best_detector_count = {'shoe': 0, 'ared': 0, 'amvd': 0, 'mbgtd':0, 'vicon':0}

load_traj=True  #set to false to recompute the trajectories, or true to reload the previously saves trajectories (much faster to reload)
if load_traj==True:
    stored_trajectories = sio.loadmat("results/stored_vicon_trajectories.mat")
    
for f in glob.glob('{}/*.mat'.format(folder)):
    print("Processing {}".format(f))
    trial_name = f.replace(folder,'')
    trial_stats = [trial_name]
    
    data = sio.loadmat(f)
    imu = data['imu']
    ts = data['ts']
    gt = data['gt']
    zv_shoe_opt = data['zv_shoe_opt'][0]
    zv_ared_opt = data['zv_ared_opt'][0]
    zv_amvd_opt = data['zv_amvd_opt'][0]
    zv_mbgtd_opt = data['zv_mbgtd_opt'][0]
    zv_vicon_opt = data['zv_vicon_opt'][0]
    best_detector = data['best_detector'][0]
    best_detector_count[best_detector]+=1
    
        ###add custom detector and its zv output to lists:
    det_list = ['shoe'] #['shoe', 'ared', 'amvd', 'mbgtd', 'vicon']
    zv_list =  [zv_shoe_opt, zv_ared_opt, zv_amvd_opt, zv_mbgtd_opt, zv_vicon_opt]

    ins = INS(imu, sigma_a = 0.00098, sigma_w = 8.7266463e-5, T=1.0/200) #microstrain

    ###Estimate trajectory
    for i in range(0, len(det_list)):
        if load_traj!=True:
            x = ins.baseline(zv=zv_list[i])
            print(x.shape)
            x, gt = align_plots(x,gt) #rotate data
            saved_trajectories['{}_{}'.format(trial_name, det_list[i])] = x
        else:
            x = stored_trajectories['{}_{}'.format(trial_name, det_list[i])]
        ###Calculate ARMSE between estimate and Vicon
        armse_2d = compute_error(x, gt, '2d')
        armse_3d = compute_error(x, gt, '3d')

#        print("ARMSE for {}: {}".format(det_list[i], armse_2d))
        trial_stats.append(armse_2d)
        trial_stats.append(armse_3d)
    
    stats.append(trial_stats)

print("best detector count:")
print(best_detector_count)   

###Process the results and save to csv files (saves csv with errors for each detector for each trial) 
stats_header = ['folder']
for i in range(0, len(det_list)):
    stats_header.append(det_list[i]+'_2d')
    stats_header.append(det_list[i]+'_3d')

csv_filename = 'results/vicon_results_raw.csv'
with open(csv_filename, "w") as f:
    writer = csv.writer(f)
    writer.writerow(stats_header)
    writer.writerows(stats)
    
if load_traj!=True:
    sio.savemat("results/stored_vicon_trajectories.mat", saved_trajectories)
