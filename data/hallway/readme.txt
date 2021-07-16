Processed MAT files in the three subdirectories include:

imu: N x 6 
	-IMU readings for N timesteps

ts: 1 x N 
	- timesteps for the N IMU readings

gt_idx: 1 x M 
	- M ground truth readings, where each value indicates the timestep (a value between 0 and N) 	that the ground truth value was collected at

gt: M x 3
	- In this case, ground truth is the 3D position of the foot at the M timesteps that can be compared with the estimated foot pose


