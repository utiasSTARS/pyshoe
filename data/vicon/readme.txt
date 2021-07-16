Processed MAT files here include:

imu: N x 6 
	-IMU readings for N timesteps

ts: 1 x N 
	- timesteps for the N IMU readings

gt: M x 3
	- The 3D position of the foot at each IMU timestep recorded using VICON 3D tracking

zv_*_opt: 1 x N
	- The 'optimal' zero-velocity updates when using the specified detector whose thresholds were optimized via a grid search. e.g., zv_shoe_opt, zv_ared_opt

G_*_opt: 1 x 1
	- The optimized threshold used for the zv_*_opt outputs

trial_error: 5 x 1
	- The RMSE for each detector in the list: [shoe_err, ared_err, ared_err, mbgtd_err, vicon_err]

best detector: 1 x 1 string
	- The detector (one of five) that produced the lowest error for the sequence

