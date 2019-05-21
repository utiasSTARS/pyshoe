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
###add custom detector and its zv output to lists:
det_list = ['ared', 'shoe', 'adaptive', 'lstm']
thresh_list =  [[0.55], [8.5e7], [[1e7, 35e7, 1e7]], [0]]
W_list = [5, 5, 5, 0]
flight_list = ['2', '4', '6', '8']

error_logger = StairErrorLogger(flight_list, det_list)

load_traj=True #set to false to recompute the trajectories, or true to reload the previously saves trajectories (much faster to reload)
if load_traj==True:
    stored_trajectories = sio.loadmat("results/stored_stair_trajectories.mat")
    
for f in glob.glob('{}/*/processed_data.mat'.format(source_dir),recursive=True):
    trial_name = f.replace(source_dir,'').replace('/processed_data.mat','')
    print(trial_name)
    trial_stats = [trial_name]
    data = sio.loadmat(f)
    imu = data['imu']
    ts = data['ts'][0]
    gt = data['gt'][0]
    trigger_ind = data['gt_idx'][0]
    ins = INS(imu, sigma_a = 0.00098, sigma_w = 8.7266463e-5, T=1.0/200) #microstrain

    ###Estimate trajectory for each zv detector
    for i in range(0, len(det_list)):
        for j in range(0, len(thresh_list[i])):
            if load_traj != True:
                zv = ins.Localizer.compute_zv_lrt(W=W_list[i], G=thresh_list[i][j], detector=det_list[i])
                x = ins.baseline(zv=zv)
                saved_trajectories["{}_det_{}_G_{}".format(trial_name, det_list[i], thresh_list[i][j])] = x
            else:
                x = stored_trajectories["{}_det_{}_G_{}".format(trial_name, det_list[i], thresh_list[i][j])]

            ###Calculate ARMSE between estimate and Vicon
            idx = int((trigger_ind.shape[0]-1)/2)
            furthest_point_idx = trigger_ind[idx]
            furthest_point_error = np.abs(x[furthest_point_idx,2] - gt[idx]) 
            loop_closure_z = np.abs(x[-1,2])
            loop_closure_3d = np.abs(np.sqrt(np.sum(np.power(x[-1,0:3] - np.zeros(3),2))))
            trial_stats.append(furthest_point_error)
            trial_stats.append(loop_closure_z)
            trial_stats.append(loop_closure_3d)
            error_logger.update(np.array([loop_closure_3d, loop_closure_z, furthest_point_error]), trial_name[-1], det_list[i])
                
    stats.append(trial_stats)

###Process the results and save to csv files (saves a raw csv, and a processed csv that reproduces the paper results)    
error_logger.process_results()   
stats_header = ['Trial']
for i in range(0, len(det_list)):
    for j in range(0, len(thresh_list[i])):
        stats_header.append("{}_G={}_furthest_z".format(det_list[i], thresh_list[i][j]))
        stats_header.append("{}_G={}_closure_z".format(det_list[i], thresh_list[i][j]))
        stats_header.append("{}_G={}_closure_3d".format(det_list[i], thresh_list[i][j]))

csv_filename = 'results/stair_results_raw.csv'
with open(csv_filename, "w") as f:
    writer = csv.writer(f)
    writer.writerow(stats_header)
    writer.writerows(stats)

if load_traj != True:
    sio.savemat("results/stored_stair_trajectories.mat", saved_trajectories)