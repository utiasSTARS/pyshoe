import numpy as np
import glob
from ins_tools.INS import INS
from ins_tools.util import *
from ins_tools import geometry_helpers
import scipy.io as sio

def main():
    print("Optimizing thresholds for all VICON data")
    print("Warning: This may take several hours (please download the processed data instead of running this)")
    source_dir='data/vicon/raw/'
    for folder in sorted(glob.glob('{}/*'.format(source_dir))):
        print('Importing file: ' + folder)
        data = sio.loadmat('{}/processed_data.mat'.format(folder))
        imu = data['imu']
        gt = data['gt']
        gt_rpy = data['gt_rpy']
        ts = data['ts']
        
        ins = INS(imu, sigma_a = 0.00098, sigma_w = 8.7266463e-5, T=1.0/200)
        ins.Localizer.ts = ts
        ins.Localizer.gt = gt
        G_vicon_opt, vicon_err, zv_vicon_opt = optimize_gamma(ins, gt, thresh=list(np.arange(1e-2,1e-1,0.25e-2))+list(np.arange(1e-1,1,0.25e-1)), detector='vicon') #do not call temporal_align prior to this step
        G_shoe_opt, shoe_err, zv_shoe_opt = optimize_gamma(ins, gt, W=5, thresh=list(np.arange(1e5,1e6,0.25e5))+list(np.arange(1e6,1e7,0.25e6))+list(np.arange(1e7,1e8,0.25e7))+list(np.arange(1e8,1e9,0.25e8)), detector='shoe')
        G_amvd_opt, amvd_err, zv_amvd_opt = optimize_gamma(ins, gt, W=5, thresh=list(np.arange(1e-4,1e-3,0.25e-4))+list(np.arange(1e-3,1e-2,0.25e-2))+list(np.arange(1e-2,1e-1,0.25e-2))+list(np.arange(1e-1,2,0.25e-1)), detector='amvd')
        G_ared_opt, ared_err, zv_ared_opt = optimize_gamma(ins, gt, W=5, thresh=list(np.arange(1e-2,1e-1,0.25e-2))+list(np.arange(1e-1,1,0.25e-1))+list(np.arange(1,3, 0.05)), detector='ared')
        G_mbgtd_opt, mbgtd_err, zv_mbgtd_opt = optimize_gamma(ins, gt, W=2, thresh=list(np.arange(1e-3,1e-2,0.25e-3))+list(np.arange(1e-2,1e-1,0.25e-2))+list(np.arange(1e-1,1,0.25e-1)), detector='mbgtd')                  
        data['G_vicon_opt'] = G_vicon_opt
        data['zv_vicon_opt'] = zv_vicon_opt
        data['G_shoe_opt'] = G_shoe_opt
        data['zv_shoe_opt'] = zv_shoe_opt
        data['G_ared_opt'] = G_ared_opt
        data['zv_ared_opt'] = zv_ared_opt
        data['G_amvd_opt'] = G_amvd_opt
        data['zv_amvd_opt'] = zv_amvd_opt
        data['G_mbgtd_opt'] = G_mbgtd_opt
        data['zv_mbgtd_opt'] = zv_mbgtd_opt     
        
        det = ['shoe', 'ared', 'amvd', 'mbgtd', 'vicon']
        errors = np.array([shoe_err, ared_err, ared_err, mbgtd_err, vicon_err])
        best_idx = np.argmin(errors)
        best_error = errors[best_idx]
        best_det = det[best_idx]
        data['trial_error'] = errors
        data['best_error'] = best_error
        data['best_detector'] = best_det
        
        output_name = folder.replace(source_dir,'')
        print("Processed folder {}".format(output_name))
        sio.savemat('data/vicon/processed/{}.mat'.format(output_name), data)

main()