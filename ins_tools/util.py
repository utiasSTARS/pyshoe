import numpy as np
from numpy import linalg as LA
from ins_tools.geometry_helpers import *
import ins_tools.SVM as SVM
import copy
import csv
from liegroups import SO3
    
#rotates a (Nx3) trajectory traj1 to align with another (Nx3)trajectory traj2
def align_plots(traj1,traj2, dist=0.8, use_totstat=False, align_idx=None):
    if use_totstat == False:
            #remove any offset between trajectories (origins should begin at 0)
        traj1 = traj1 - traj1[0]
        traj2 = traj2 - traj2[0]    
            #align the orientation
        xdist = np.cumsum(np.diff(traj1[:,0]))
        ydist = np.cumsum(np.diff(traj1[:,1]))
        d = np.sqrt(np.power(xdist,2) + np.power(ydist,2))
        if np.max(d)>=dist:
            ind = np.where(d>=dist)[0][0]
        else:
            ind = 300 #choose arbitrary point to align because this isn't a good trajectory
            
        vec1 = traj1[ind,0:3]
        vec1[2] = 0 #z-position is 0 
        
        if np.abs(traj2.shape[0] - traj1.shape[0]) <5: #for temporally aligned traj2
            vec2 = traj2[ind,0:3]
        if np.abs(traj2.shape[0] - traj1.shape[0]) >=5: #for non-aligned traj2
            #ts_1m_vic = np.where(ts_vic>=ts_imu[ind])[0][0] 
            #vec2 = traj2[ts_1m_vic]
            xdist2 = np.cumsum(np.diff(traj2[:,0]))
            ydist2 = np.cumsum(np.diff(traj2[:,1]))

            dist2 = np.sqrt(np.power(xdist2,2) + np.power(ydist2,2))
            if dist2 >= dist:
                ind2 = np.where(dist2>=dist)[0][0] +1
                vec2 = traj2[ind2,0:3]
            else:
                print("Warning: distance is less than " + str(dist))
            

        vec2[2]=0
    if use_totstat == True:
                    #align the orientation
        xdist = np.cumsum(np.diff(traj1[:,0]))
        ydist = np.cumsum(np.diff(traj1[:,1]))
        d = np.sqrt(np.power(xdist,2) + np.power(ydist,2))
        if np.max(d)>=dist:
            ind = np.where(d>=dist)[0][0]
        else:
            ind = 300 #choose arbitrary point to align because this isn't a good trajectory
        if align_idx:
            ind = align_idx
        vec1 = np.copy(traj1[ind,0:3]) 
        vec1[2]=0  #only align on the xy plane since we already know the initial roll/pitch angles
        vec2 = np.copy(traj2[1,0:3])
        vec2[2] = 0

    crossvec = np.cross(vec1,vec2)
    sign_angle = np.sign(crossvec[2])
    angle = vec1.dot(vec2)/(LA.norm(vec1)*LA.norm(vec2))
    if angle >= 1.0:
        angle = 1.0
           
    angle = sign_angle*np.arccos(angle)
    
    Rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    
    x_hatz = traj1[:,2].reshape(-1,1)
    x_rot = np.einsum('ij, kj->ki', Rotation, traj1[:,0:2])  #multiply R by each value in traj1
    x_rot = np.hstack((x_rot,x_hatz))
    return x_rot, traj2

### Compute the average RMSE for a trial (works for continuous or sparse ground truth, for 2d, 3d, or vertical-only estimates)    
def compute_error(imu, vic, dim='2d'): #enter imu then vicon
    if dim =='2d':
        d = 2
    if dim == '3d':
        d = 3 
    if dim != 'z':    
        diff = imu[:,0:d]-vic[:,0:d]
    else:
        diff = (imu[:,2]-vic).reshape((-1,1))
    diff_sq = np.power(diff,2)
    mean_diff_sq = np.mean(diff_sq,axis=1)
    RMSE = np.sqrt(mean_diff_sq)
    ARMSE = np.average(RMSE).round(3)
    return ARMSE
    
   
###Moving average filter of size W
def moving_average(a, n) : #n must be odd)
    if n == 1:
        return a
    else:
        for i in range(a.shape[1]):
            if (n % 2) == 0:
                n -=1
            ret = np.cumsum(a[:,i], dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            a[:,i] = np.pad(ret[n - 1:-2] / n , int((n-1)/2+1), 'edge')
        return a

### Used in optimize_gamma for zero-velocity threshold optimization
def gridsearch(ins, W, G, detector='shoe'): #import window size, multiple zupt threshold,and IMU data
    imu = ins.imudata
    lrt = ins.Localizer.compute_zv_lrt(detector=detector, return_zv=False)
    zv_grid = np.zeros((len(G),imu.shape[0]))
    for i in range(len(G)):
        zv_grid[i,:]=lrt<G[i]
    return zv_grid.astype(int)

### Uses a grid search to optimize a zero-velocity threshold for a specific sequence of data.  Requires position ground truth.    
def optimize_gamma(ins, vic, thresh=[0.5e7, 10e7], W=5, detector='shoe'):
    opt_error = 1000.
    if len(thresh)==2:
        thresh = np.arange(thresh[0], thresh[1], (thresh[1]-thresh[0])/20)
    zv_grid = gridsearch(ins, W, thresh, detector=detector)
    for i in range(0, zv_grid.shape[0]):  
        ins.baseline(zv=zv_grid[i])
        ins.x_rot, vic_rot = align_plots(ins.x,vic) #rotate all traj to match the first
        error = compute_error(ins.x_rot, vic_rot,'2d') #use 2d because z-axis isn't perfectly aligned
        if error <= opt_error:
            opt_error = error
            print(error)
            opt_thresh=thresh[i]
            zv_opt = zv_grid[i]
#            print("new minimum error: {} at gamma={}".format(opt_error, opt_thresh))
    print("minimum ARMSE: {} at gamma={}".format(opt_error, opt_thresh))
    return opt_thresh, opt_error, zv_opt

def angle_wrap(angle,radians=False):
    '''
    Wraps the input angle to 360.0 degrees.

    if radians is True: input is assumed to be in radians, output is also in
    radians

    '''

    if radians:
        wrapped = angle % (np.pi)
        if wrapped < 0.0:
            wrapped = np.pi + wrapped
        if angle < 0:
            wrapped = -np.pi + wrapped

    else:

        wrapped = angle % 360.0
        if wrapped < 0.0:
            wrapped = 360.0 + wrapped

    return wrapped 

def rotate_attitude_to_gt(est_rpy, gt_rpy):
    est_rpy[:,2] = gt_rpy[0,2] - (np.unwrap(est_rpy[:,2]) - est_rpy[0,2])
    for i in range(0,est_rpy.shape[0]):
        est_rpy[i,2] = angle_wrap(est_rpy[i,2],radians=True)
    est_rpy[:,0] = gt_rpy[0,0] + (est_rpy[:,0]-est_rpy[0,0])
    est_rpy[:,1] = gt_rpy[0,1] + (est_rpy[:,1]-est_rpy[0,1])
    return est_rpy

def compute_attitude_error(est_rpy, gt_rpy):
    r_gt = gt_rpy[0,:]
    r_est = est_rpy[0,:]
    
    R_gt = SO3.from_rpy(r_gt[0], r_gt[1], r_gt[2])
    R_est = SO3.from_rpy(r_est[0], r_gt[1], r_gt[2])
    dR = R_gt.dot(R_est.inv())
    
    ang_error = np.zeros(est_rpy.shape)
    for i in range(0, gt_rpy.shape[0]):
        R = SO3.from_rpy(est_rpy[i,0], est_rpy[i,1], est_rpy[i,2])
    #    new_R = dR.dot(R)
    #    x_shoe[i,0:3] = dR.as_matrix().dot(x_shoe[i,0:3])
        
        R_gt = SO3.from_rpy(gt_rpy[i,0], gt_rpy[i,1], gt_rpy[i,2])
        
        error = np.eye(3)- (R.dot(R_gt.inv())).as_matrix()
        ang_error[i,2] = -error[0,1]
        ang_error[i,1] = error[0,2]
        ang_error[i,0] = -error[2,1]
#    rpy_norm_error = np.linalg.norm(ang_error,axis=1)
    return ang_error 

###class for recording results for hallway dataset, which formats the csv to exactly reproduce the paper results.
class HallwayErrorLogger():
    def __init__(self, modes, subjects, detector_list, thresh_list):
        self.modes=modes
        self.subjects= subjects
        self.detector_list = detector_list
        self.thresh_list = thresh_list
        self.num_thresh =  [len(t) for t in self.thresh_list]
        self.num_detector = len(detector_list)
        self.results = {}
        self.results['total']={}
    
        for m in modes:
            self.results[m] = {}
            for sub in self.subjects:
                self.results[m][sub] = {}
                for det,c in zip(self.detector_list, self.num_thresh):
                    self.results[m][sub][det] = np.zeros(c)
                    self.results['total'][det] = np.zeros(c)
        self.count = copy.deepcopy(self.results)
        
    def update(self, error, mode, subject, detector, thresh_idx):
        self.count[mode][subject][detector][thresh_idx] +=1
        self.results[mode][subject][detector][thresh_idx] += error
        self.results['total'][detector][thresh_idx] += error
        self.count['total'][detector][thresh_idx] += 1
    
    def process_results(self):
        results_list = []
        header_1, header_2 = ['Mode', 'Subject'], ['', '']
        for d, t in zip(self.detector_list, self.thresh_list):
            for thresh in t:
                header_1.append(d)
                header_2.append(str(thresh))
        
        for m in self.modes:
            for s in self.subjects:
                line = [m,s]
                avg_error = ['', 'Total']
                for det, thresh in zip(self.detector_list, self.thresh_list):
                    for t in range(0,len(thresh)):
                        error = self.results[m][s][det][t] / self.count[m][s][det][t]
                        avg_error.append((self.results['total'][det][t] / self.count['total'][det][t]).round(4))
                        line.append(str(error.round(4)))
                results_list.append(line)
                        
                
        csv_filename = 'results/hallway_results.csv'
        with open(csv_filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header_1)
            writer.writerow(header_2)
            writer.writerows(results_list)  
            writer.writerow(avg_error)

###class for recording results for stairway dataset, which formats the csv to exactly reproduce the paper results.    
class StairErrorLogger():
    def __init__(self, flights, detector_list):
        self.flights = flights
        self.detector_list = detector_list
        self.num_detector = len(detector_list)
        self.results = {}
        self.results['total']={}
        self.count = {}
        self.count['total'] = {}
    
        for d in self.detector_list:
            self.results[d] = {}
            self.count[d] = {}
            self.count['total'][d] = 0
            for f in self.flights:
                self.count[d][f] = 0
                self.results[d][f] = np.zeros(3)
                self.results['total'][d] = np.zeros(3)
            
        
    def update(self, error, flight, detector):
        self.count[detector][flight] +=1
        self.results[detector][flight] += error
        self.results['total'][detector] += error
        self.count['total'][detector] += 1
    
    def process_results(self):
        results_list = []
        header_1 = ['# Flights', 'Detector', '', 'Position Errors (m)', '']
        header_2 = ['', '', 'Loop-Closure (3D)', 'Loop-Closure (Vertical)', 'Furthest-Point (Vertical)']
        
        for f in self.flights:
            for d in self.detector_list:
                row = [f, d]
                for e in range(0,3):
                    error = (self.results[d][f][e] / self.count[d][f]).round(3)
                    row.append(error)
                results_list.append(row)    
        
        for d in self.detector_list:
            row = ['Mean', d]
            for e in range(0,3):
                avg_error = (self.results['total'][d][e] / self.count['total'][d] ).round(3)
                row.append(avg_error)
            results_list.append(row)
        
                        
                
        csv_filename = 'results/stair_results.csv'
        with open(csv_filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header_1)
            writer.writerow(header_2)
            writer.writerows(results_list)  