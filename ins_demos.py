import numpy as np
from ins_tools.util import *
import ins_tools.visualize as visualize
import matplotlib.pyplot as plt
from ins_tools.INS import INS
import scipy.io as sio

vicon_demo = False	#processes a trajectory from our VICON dataset
stair_demo = False	#processes a trajectory from our stair dataset
hallway_demo = False	#processes a trajectory from our hallway dataset
adaptive_demo = False #runs our adaptive zero-velocity detector with motion classification
lstm_demo = True	#runs our zero-velocity classifier

if vicon_demo:
    print("Vicon Demo")
    source_dir = "data/vicon/processed/"
    folder = "2017-11-27-11-13-10"
    data = sio.loadmat('{}{}.mat'.format(source_dir, folder))
    
    imu = data['imu']
    ts = data['ts'][0]
    gt = data['gt']
    
    ins = INS(imu, sigma_a = 0.00098, sigma_w = 8.7266463e-5, T=1.0/200) 
    
        ###Optimize zero-velocity threshold for given trial
    #G_opt_shoe, _, zv_opt_shoe = optimize_gamma(ins, gt, thresh=[0.5e7, 10e7], W=5, detector='shoe')
    #G_opt_ared, _, zv_opt_shoe = optimize_gamma(ins, gt, thresh=[0.1, 2], W=5, detector='ared')
        ###load the pre-computed optimal thresholds
    G_opt_shoe = float(data['G_shoe_opt'])
    G_opt_ared = float(data['G_ared_opt'])
    
        ###Estimate trajectory
    x_shoe = ins.baseline(W=5, G=G_opt_shoe, detector='shoe')
    x_ared = ins.baseline(W=5, G=G_opt_ared, detector='ared')
    
    x_shoe, _ = align_plots(x_shoe,gt) #rotate data
    x_ared, _ = align_plots(x_ared,gt)
    
    visualize.plot_topdown([x_shoe,x_ared, gt])
    plt.figure()
    plt.plot(data['zv_shoe_opt'][0])
    
            ###Calculate ARMSE between estimate and Vicon
    shoe_error = compute_error(x_shoe, gt,'2d')
    ared_error = compute_error(x_ared, gt, '2d')
    
    print("ARMSE for ARED: {}".format(ared_error))
    print("ARMSE for SHOE: {}".format(shoe_error))

if hallway_demo:
    print("Hallway Demo")
    folder = "data/hallway/run/0/2018-04-09-10-21-01"
    data = sio.loadmat('{}/processed_data.mat'.format(folder))
    imu = data['imu']
    gt = data['gt']
    trigger_ind = data['gt_idx'][0]
    ins = INS(imu, sigma_a = 0.00098, sigma_w = 8.7266463e-5, T=1.0/200) #microstrain
    
        #shoe
    ins.baseline(G=5e7)
    ins.x, gt = align_plots(ins.x, gt, dist=0.8, use_totstat=True) #rotate with totalstation points
    visualize.plot_topdown([ins.x, gt], gt_method='sparse', legend=['SHOE', 'Ground Truth'])

        ###Calculate ARMSE between estimate and Vicon
    shoe_error = compute_error(ins.x[trigger_ind], gt,'2d')
    print("ARMSE for SHOE: ", shoe_error)

if stair_demo:
    print("Stair Demo")
    folder = "data/stairs/test/2018-08-11-13-31-07-up-8"
    data = sio.loadmat('{}/processed_data.mat'.format(folder))
    imu = data['imu']
    ts = data['ts'][0]
    gt = data['gt'][0]
    gt_coords = data['gt'][0]
    trigger_ind = data['gt_idx'][0]
    ins = INS(imu, sigma_a = 0.00098, sigma_w = 8.7266463e-5, T=1.0/200) #microstrain
    
        #shoe
    ins.baseline(W=5, G=1e7, detector='shoe')
            ###Calculate ARMSE between estimate and Vicon
    shoe_error = compute_error(ins.x[trigger_ind], gt, dim='z')
    idx = int((trigger_ind.shape[0]-1)/2)
    furthest_point_idx = trigger_ind[idx]
    furthest_point_error = ins.x[furthest_point_idx,2] - gt[idx] 
    visualize.plot_stairs(ts[0:(trigger_ind[-1]+1)], [ins.x[0:(trigger_ind[-1]+1)]], gt, trigger_ind=list(trigger_ind), title='Stair Climbing Trial (Vertical Plane)')
    print("ARMSE for SHOE: {}".format(shoe_error))
    print("Furthest point error: {}".format(furthest_point_error))

if adaptive_demo:
    print("Adaptive Detector Demo")
    source_dir = "data/vicon/processed/"
    folder = "2017-11-22-11-22-03"
    data = sio.loadmat('{}{}.mat'.format(source_dir, folder))
    imu = data['imu']
    ts = data['ts'][0]
    gt = data['gt']
    
    ins = INS(imu, sigma_a = 0.00098, sigma_w = 8.7266463e-5, T=1.0/200) #microstrain
    x_adapt = ins.baseline(W=5, G=[1e7, 35e7, 1e7], detector='adaptive')
    x_adapt, gt = align_plots(x_adapt, gt) #Align with Ground Truth
    
    visualize.plot_topdown([x_adapt, gt], legend=['Adaptive', 'Ground Truth'])
            ###Calculate ARMSE between estimate and Vicon
    adapt_error = compute_error(x_adapt, gt, '2d')
    print("ARMSE for Adaptive thresholding: {}".format(adapt_error))

if lstm_demo:
    print("LSTM Detector Demo")
    source_dir = "data/vicon/processed/"
    folder = "2017-11-22-11-22-03"
    data = sio.loadmat('{}{}.mat'.format(source_dir, folder))   
    imu = data['imu']
    ts = data['ts'][0]
    gt = data['gt']
    
    ins = INS(imu, sigma_a = 0.00098, sigma_w = 8.7266463e-5, T=1.0/200) #microstrain
    
    zv_lstm = ins.Localizer.LSTM()
        ###Estimate trajectory
    x = ins.baseline(zv=zv_lstm)
    x, gt = align_plots(x, gt) #rotate data
    
    visualize.plot_topdown([x,gt], legend=['LSTM', 'Ground Truth'])
    
            ###Calculate ARMSE between estimate and Vicon
    lstm_error = compute_error(x, gt,'2d')
    print("ARMSE for LSTM: {}".format(lstm_error))
    
