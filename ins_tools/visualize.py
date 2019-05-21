import matplotlib.pyplot as plt
import numpy as np

###Plot 2d (xy) trajectories (traj is a list of trajectories, with gt being the final one) 
def plot_topdown(traj, trigger_ind=None, gt_method = 'dense', title='', save_dir=None, legend=[], Loc=4, markerind =[]):
    #gt_method is dense for Vicon, and sparse for hallway + stairs
    plt.figure()
    plt.clf()
#    plt.rc('text', usetex=True)
#    plt.rc('font', family='serif')
#    plt.rcParams["font.family"] = "Times New Roman"
    colour = ['blue', 'green', 'darkorange', 'gold']
    for i in range(len(traj)-1):
        if trigger_ind is not None:
            plt.plot(-traj[i][:,0], traj[i][:,1], '-gD', markevery=trigger_ind, linewidth = 1.7, color=colour[i], markersize=5, markeredgewidth=0.005, markeredgecolor='black')
        else:
            plt.plot(-traj[i][:,0], traj[i][:,1], linewidth = 1.7, color=colour[i])
    if gt_method == 'sparse':
        plt.scatter(-traj[-1][:,0], traj[-1][:,1], s=13, color='red',zorder=len(traj)+1)
    elif gt_method == 'none':
        gt=None
    else:
        plt.plot(-traj[-1][:,0], traj[-1][:,1], color='red')

    plt.title(title, fontsize=20, color='black')
    plt.ylabel('y (m)', fontsize=22)
    plt.xlabel('x (m)', fontsize=22)
    plt.tick_params(labelsize=22)
    plt.subplots_adjust(top=0.8)
    plt.legend(legend, fontsize=15, numpoints=1)
    plt.grid()
    if save_dir:
        plt.savefig(save_dir, dpi=400, bbox_inches='tight')

###Plot the vertical estimate wrt time.  (traj is a list of trajectories, no ground truth is required)
def plot_vertical(ts, traj, trigger_ind=None, title='', save_dir=None, legend=[], Loc=4, markerind =[]):
    plt.figure()
    plt.clf()
#    plt.rc('text', usetex=True)
#    plt.rc('font', family='serif')
#    plt.rcParams["font.family"] = "Times New Roman"
    colour = ['blue', 'green', 'darkorange', 'gold']
    for i in range(len(traj)-1):
        if trigger_ind is not None:
            plt.plot(ts, traj[i][:,2], '-gD', markevery=trigger_ind, linewidth = 1, color=colour[i], markersize=5, markeredgewidth=0.005, markeredgecolor='black')
        else:
            plt.plot(ts, traj[i][:,2], linewidth = 1, color=colour[i])

    plt.title(title, fontsize=20, color='black')
    plt.ylabel('z (m)', fontsize=22)
    plt.xlabel('Time (s)', fontsize=22)
    plt.xlim([ts[trigger_ind[0]]-1, ts[trigger_ind[-1]]+1])
    plt.tick_params(labelsize=22)
    plt.subplots_adjust(top=0.8)
    plt.legend(legend, fontsize=15, numpoints=1)
    plt.grid()
    if save_dir:
        plt.savefig(save_dir, dpi=400, bbox_inches='tight')        

###plot stair trajectories, along with the floor levels       
def plot_stairs(ts, traj, gt, title='', legend=None, trigger_ind=None, save_dir=None):       
    plt.figure()
    plt.clf()
#    plt.rc('text', usetex=True)
#    plt.rc('font', family='serif')
#    plt.rcParams["font.family"] = "Times New Roman"
    colour = ['blue', 'green', 'darkorange', 'gold']
    for i in range(0, len(traj)):
        if trigger_ind is not None:
            plt.plot(ts, traj[i][:,2], '-gD', markevery=trigger_ind, linewidth = 1.7, color=colour[i], markersize=5, markeredgewidth=0.005, markeredgecolor='black')
        else:
            plt.plot(ts, traj[i][:,2])
    for i in range(0, int((gt.shape[0]-1)/2 +1)):    
        plt.plot(ts, gt[i]*np.ones(len(ts)), '--', linewidth=1.25, color='red')
    plt.title(title, fontsize=20, color='black')
    plt.xlabel('Time (s)', fontsize=22)
    plt.ylabel('Vertical Height (m)', fontsize=22)  
    plt.tick_params(labelsize=22)
    plt.grid()
    if legend is not None:
        plt.legend(legend, fontsize=15, numpoints=1)
    if save_dir:
        plt.savefig(save_dir, dpi=400, bbox_inches='tight')

        ###plot IMU linear acceleration
def plot_acc(imudata):
    plt.figure()
    plt.tick_params(labelsize=19)
    plt.plot(imudata[:,0]/9.8)
    plt.plot(imudata[:,1]/9.8)
    plt.plot(imudata[:,2]/9.8)
    plt.title(['Linear Acceleration'], fontsize=19)
    plt.legend(['x', 'y', 'z'], fontsize=14)
    plt.ylabel('linear acceleration (Gs)', fontsize=19)
    plt.grid()

        ###plot IMU angular velocity
def plot_vel(imudata):
    plt.figure()
    plt.tick_params(labelsize=19)
    plt.plot(imudata[:,3]*180/np.pi)
    plt.plot(imudata[:,4]*180/np.pi)
    plt.plot(imudata[:,5]*180/np.pi)
    plt.title(['Angular Velocity'],fontsize=19)
    plt.legend(['x', 'y', 'z'], fontsize=14)
    plt.ylabel('Angular Velocity (deg/s)', fontsize=19)
    plt.grid()
