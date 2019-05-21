import numpy as np
from ins_tools.util import *
from ins_tools.EKF import Localizer

class INS():
    def __init__(self, imudata, sigma_a=0.01, sigma_w=0.1*np.pi/180, T=1.0/125, dt=None):
        self.config = {
        "sigma_a": sigma_a,
        "sigma_w": sigma_w,
        "g": 9.8029,#9.658,
        "T": T,
        "dt": dt,
            }
        self.imudata = imudata
        self.sigma_a = self.config["sigma_a"]
        self.sigma_w = self.config["sigma_w"]
        self.var_a = np.power(self.sigma_a,2)
        self.config["var_a"] = self.var_a
        self.var_w = np.power(self.sigma_w,2)
        self.config["var_w"] = self.var_w
        self.g = self.config["g"]
        self.T = self.config["T"]
        ##process noise in body frame
        self.sigma_acc = 0.5*np.ones((1,3))
        self.var_acc = np.power(self.sigma_acc,2)
        self.sigma_gyro = 0.5*np.ones((1,3))*np.pi/180
        self.var_gyro = np.power(self.sigma_gyro,2)
    
        self.Q = np.zeros((6,6))  ##process noise covariance matrix Q
        self.Q[0:3,0:3] = self.var_acc*np.identity(3)
        self.Q[3:6,3:6] = self.var_gyro*np.identity(3)
        self.config["Q"] = self.Q
        
        self.sigma_vel = 0.01 #0.01 default
        self.R = np.zeros((3,3))
        self.R[0:3,0:3] = np.power(self.sigma_vel,2)*np.identity(3)   ##measurement noise, 0.01 default
        self.config["R"] = self.R
        
        self.H = np.zeros((3,9))
        self.H[0:3,3:6] = np.identity(3)
        self.config["H"]= self.H        
        
        self.Localizer = Localizer(self.config, imudata)
        
    def baseline(self,W=5, G=5e8, detector='shoe', zv=None):
        imudata = self.imudata
        
        x_check,q, P_check = self.Localizer.init() #initialize state
        x_hat = x_check 
        self.x = x_hat
        
        if zv is None:
            ### Compute the trial's zero-velocity detections using the specified detector
            self.zv = self.Localizer.compute_zv_lrt(W,G, detector=detector) #ZV detection 
        else:
            ### Use a pre-computed zero-velocity estimate
            self.zv = zv

        for k in range(1,x_check.shape[0]):      
            #predictor
            if self.config['dt'] is None:
                dt = self.config['T']
            else:
                dt = self.config['dt'][k-1]
            x_check[k,:], q[k,:],Rot = self.Localizer.nav_eq(x_check[k-1,:], imudata[k,:], q[k-1,:], dt) #update state through motion model
            
            F,G = self.Localizer.state_update(imudata[k,:],q[k-1,:], dt) 
        
            P_check[k,:,:] = (F.dot(P_check[k-1,:,:])).dot(F.T) + (G.dot(self.Q)).dot(G.T)
            P_check[k,:,:] = (P_check[k,:,:] + P_check[k,:,:].T)/2 #make symmetric
            #corrector
            if self.zv[k] == True: 
                x_hat[k,:], P_check[k,:,:], q[k,:] = self.Localizer.corrector(x_check[k,:], P_check[k,:,:], Rot )
            else:
                x_hat[k,:] = x_check[k,:]
            self.x[k,:] = x_hat[k,:]  
        self.x[:,2] = -self.x[:,2] 
        return self.x
    
