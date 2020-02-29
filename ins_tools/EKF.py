import numpy as np
from numpy import linalg as LA
import ins_tools.LSTM as lstm #remove if there is no pytorch installation
import ins_tools.SVM as SVM #remove if there is no sci-kit-learn installation
from ins_tools.util import *
from ins_tools.geometry_helpers import quat2mat, mat2quat, euler2quat, quat2euler
from sklearn.externals import joblib
import sys
sys.path.append('../')

class Localizer():
    def __init__(self, config, imudata):
        self.config = config
        self.imudata = imudata
        self.count=1
    def init(self):
        imudata = self.imudata 
        x = np.zeros((imudata.shape[0],9)) #initialize state to be at 0
        q = np.zeros((imudata.shape[0],4)) #Initialize quaternion 
        avg_x = np.mean(imudata[0:20,0])
        avg_y = np.mean(imudata[0:20,1])
        avg_z = np.mean(imudata[0:20,2]) #gets avg accelerometer values for finding roll/pitch
        
        heading = 0
        roll = np.arctan2(-avg_y,-avg_z)
        pitch = np.arctan2(avg_x,np.sqrt(avg_y*avg_y + avg_z*avg_z))
           
        attitude = np.array([roll, pitch, heading])
        x[0, 6:9] = attitude
        q[0, :] = euler2quat(roll, pitch, heading, 'sxyz')

        P_hat = np.zeros((imudata.shape[0],9,9)) #initial covariance matrix P
        P_hat[0,0:3,0:3] = np.power(1e-5,2)*np.identity(3) #position (x,y,z) variance
        P_hat[0,3:6,3:6] = np.power(1e-5,2)*np.identity(3) #velocity (x,y,z) variance
        P_hat[0,6:9,6:9] = np.power(0.1*np.pi/180,2)*np.identity(3) #np.power(0.1*np.pi/180,2)*np.identity(3)
        return x, q, P_hat
          
    def nav_eq(self, xin,imu,qin,dt):
        #update Quaternions
        x_out = np.copy(xin) #initialize the output
        omega = np.array([[0,-imu[3], -imu[4], -imu[5]],  [imu[3], 0, imu[5], -imu[4]],  [imu[4], -imu[5], 0, imu[3]],  [imu[5], imu[4], -imu[3], 0]])
    
        norm_w = LA.norm(imu[3:6])
        if(norm_w*dt != 0):
            q_out = (np.cos(dt*norm_w/2)*np.identity(4) + (1/(norm_w))*np.sin(dt*norm_w/2)*omega).dot(qin) 
        else:
            q_out = qin

        attitude = quat2euler(q_out,'sxyz')#update euler angles
        x_out[6:9] = attitude    
        
        Rot_out = quat2mat(q_out)   #get rotation matrix from quat
        acc_n = Rot_out.dot(imu[0:3])       #transform acc to navigation frame,  
        acc_n = acc_n + np.array([0,0,self.config["g"]])   #removing gravity (by adding)
        
        x_out[3:6] += dt*acc_n #velocity update
        x_out[0:3] += dt*x_out[3:6] +0.5*np.power(dt,2)*acc_n #position update
        
        return x_out, q_out, Rot_out    
    def state_update(self, imu,q, dt):
#        return F,G
        F = np.identity(9)
        F[0:3,3:6] = dt*np.identity(3)

        Rot = quat2mat(q)
        imu_r = Rot.dot(imu[0:3])
        f_skew = np.array([[0,-imu_r[2],imu_r[1]],[imu_r[2],0,-imu_r[0]],[-imu_r[1],imu_r[0],0]])
        F[3:6,6:9] = -dt*f_skew 
        
        G = np.zeros((9,6))
        G[3:6,0:3] = dt*Rot
        G[6:9,3:6] = -dt*Rot
       
        return F,G
    def corrector(self, x_check, P_check, Rot):
        eye3 = np.identity(3)
        eye9 = np.identity(9)
        omega = np.zeros((3,3))        
        
        K = (P_check.dot(self.config["H"].T)).dot(LA.inv((self.config["H"].dot(P_check)).dot(self.config["H"].T) + self.config["R"]))
        z = -x_check[3:6] ### true state is 0 velocity, current velocity is error
        q=mat2quat(Rot)   
        dx = K.dot(z) 
        x_check += dx  ###inject position and velocity error
         
        omega[0:3,0:3] = [[0,-dx[8], dx[7]],[dx[8],0,-dx[6]],[-dx[7],dx[6],0]] 
        Rot = (eye3+omega).dot(Rot)
        q = mat2quat(Rot)
        attitude = quat2euler(q,'sxyz')
        x_check[6:9] = attitude    #Inject rotational error           
        P_check = (eye9-K.dot(self.config["H"])).dot(P_check)
        P_check = (P_check + P_check.T)/2
        return x_check, P_check, q


    def SHOE(self, W=5):
        imudata = self.imudata
        T = np.zeros(np.int(np.floor(imudata.shape[0]/W)+1))
        zupt = np.zeros(imudata.shape[0])
        a = np.zeros((1,3))
        w = np.zeros((1,3))
        inv_a = (1/self.config["var_a"])
        inv_w = (1/self.config["var_w"])
        acc = imudata[:,0:3]
        gyro = imudata[:,3:6]
    
        i=0
        for k in range(0,imudata.shape[0]-W+1,W): #filter through all imu readings
            smean_a = np.mean(acc[k:k+W,:],axis=0)
            for s in range(k,k+W):
                a.put([0,1,2],acc[s,:])
                w.put([0,1,2],gyro[s,:])
                T[i] += inv_a*( (a - self.config["g"]*smean_a/LA.norm(smean_a)).dot(( a - self.config["g"]*smean_a/LA.norm(smean_a)).T)) #acc terms
                T[i] += inv_w*( (w).dot(w.T) )
            zupt[k:k+W].fill(T[i])
            i+=1
        zupt = zupt/W
        return zupt
        
    def ARED(self, W=5): #angular rate energy detector
        imudata = self.imudata
        T = np.zeros(np.int(np.floor(imudata.shape[0]/W)+1))
        zupt = np.zeros(imudata.shape[0])
        w = np.zeros((1,3))
        gyro = imudata[:,3:6]
    
        i=0
        for k in range(0,imudata.shape[0]-W+1,W): #filter through all imu readings
            for s in range(k,k+W):
                w.put([0,1,2],gyro[s,:])
                T[i] += w.dot(w.T)
            zupt[k:k+W].fill(T[i])
            i+=1
        zupt = zupt/W
        return zupt
        
    def AMVD(self, W=5): #angular rate energy detector
        imudata = self.imudata
        T = np.zeros(np.int(np.floor(imudata.shape[0]/W)+1))
        zupt = np.zeros(imudata.shape[0])
        w = np.zeros((1,3))
        acc = imudata[:,0:3]
        i=0
        for k in range(0,imudata.shape[0]-W+1,W): #filter through all imu readings
            mean_a = np.mean(acc[k:k+W,:],axis=0)
            for s in range(k,k+W):
                w.put([0,1,2],acc[s,:])
                T[i] += (w-mean_a).dot((w-mean_a).T)
            zupt[k:k+W].fill(T[i])
            i+=1
        zupt = zupt/W    
        return zupt
        
    def MBGTD(self,W=5):
        imudata = self.imudata
        zupt = np.ones(imudata.shape[0])
        acc = imudata[:,0:3]
        D = np.zeros((W,W))
        
        def dist(a, b, W, acc):
            Dist=0
            for i in range(0,a+1):
                for j in range(b,W):
                    Dist += np.sqrt(np.sum(np.power(acc[i,:]-acc[j,:],2)))
            return Dist/((b-a)*(W-b+1))
                  
        for k in range(0, imudata.shape[0]-W+1):
            count_a = 0
            for s in range(k,k+W-1): #for each split,    
                S = s-k
                count_b=0
                for t in range(S+1,W):
                    D[count_a, count_b] = dist(S,t, W, acc[k:k+W,:]) #euclidean distance
                    count_b+=1
                count_a+=1
            zupt[k]=np.max(D)
        return zupt
    
    def vicon_zv(self, W=5):
        gt = np.copy(self.gt)
        gt[:,0:3] = moving_average(gt[:,0:3],W)
        d_ts = np.diff(self.ts)
        vel_x = np.diff(gt,axis=0)[:,0]/d_ts
        vel_y = np.diff(gt,axis=0)[:,1]/d_ts
        vel_z = np.diff(gt,axis=0)[:,2]/d_ts
        
        vel = np.hstack((vel_x.reshape(-1,1), vel_y.reshape(-1,1), vel_z.reshape(-1,1)))
        norm_vel = LA.norm(vel,axis=1).reshape(-1,1)        
        norm_vel = np.vstack((norm_vel, norm_vel[-1]))
        return norm_vel[:,0]
    
    def LSTM(self):
        lstm_detector = lstm.LSTM()
        zv_lstm = lstm_detector(self.imudata)
        return zv_lstm
    
    def adaptive_zv(self, W=5, G=[1e7, 35e7]): #specify [G_walk, G_run]
        G_walk = G[0]
        G_run = G[1]
        if len(G) == 3:
            G_stair = G[2]

        imu = self.imudata
        sample_len=200
        clf = joblib.load('results/pretrained-models/3class_vn100_200hz.pkl')
        motion = SVM.predict(imu, clf, sample_len)
 
        c=0
        offset = (imu.shape[0]-sample_len)/1000.0
        G = np.zeros(imu.shape[0])
    
        for i in np.arange(0,imu.shape[0]-(sample_len+1),offset):
            i = np.int(i)
            G[i:i+sample_len] = motion[c]
            c+=1
            
        G[np.where(G == 0)] = G_walk
        G[np.where(G==1)] = G_run
        G[np.where(G==2)] = G_stair
        G = moving_average(G.reshape(-1,1), 200).reshape((-1))
        zv = self.compute_zv_lrt(W=W, G=G, detector='shoe')

        return zv
        
    ### if a custom zero-velocity detector was added as a function above, additionally modify this list:    
    def compute_zv_lrt(self, W=5, G=3e8, detector='shoe', return_zv=True): #import window size, zv-threshold
        if detector=='shoe':    
            zv = self.SHOE(W=W)
        if detector=='ared':
            zv = self.ARED(W=W)
        if detector=='amvd':
            zv = self.AMVD(W=W)
        if detector=='mbgtd':
            zv = self.MBGTD(W=W)
        if detector == 'lstm':
            return self.LSTM()
        if detector == 'adaptive':
            return self.adaptive_zv(W,G)
        if detector == 'vicon':
            zv = self.vicon_zv(W=W)
        if return_zv:
            zv=zv<G
        return zv

        
