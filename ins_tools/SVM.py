import numpy as np
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import scipy.io as sio
import sys
sys.path.append('../')
import ins_tools.geometry_helpers 

def importdata(data, crop=300,samples_per_file=1000, seq_len=125):
    inputs_train, inputs_test = np.empty((0,seq_len*6)), np.empty((0,seq_len*6))
    for f in data:
        data = sio.loadmat('{}.mat'.format(f))
        imu = data['imu']
        imu = imu[crop:-crop,:]
        imu = normdata(imu)
        imu = split_data(imu, samples_per_file, sample_size=seq_len )
        imu = random_rotate(imu)
        imu = imu.reshape(imu.shape[0],imu.shape[1]*6)

        imu_split1, imu_split2 = np.split(imu, 2, axis=0)
        inputs_train = np.vstack((inputs_train,imu_split1))
        inputs_test = np.vstack((inputs_test,imu_split2))

    return inputs_train, inputs_test

#splits data into gyro data and accel data, normalizes each separately, and recombines
#if new =1, data is "cropped" to remove the beginning and end data that may be stationary
def normdata(data):
    #normalize the data
    gyrodata = data[:,0:3]
    gyrodata_n = normalize(gyrodata)
    acceldata = data[:,3:6]
    acceldata_n=normalize(acceldata)

    #recombine normalized data
    data_n = np.hstack((gyrodata_n,acceldata_n))
    return data_n

# breaks Nx6 dataset into a datset of size (num_samples x sample_size x 6)
def split_data(data,num_samples,sample_size=200):

    samplesize=int(sample_size)
    output = np.zeros((num_samples,samplesize,6))

    for i in range(num_samples):
        offset = int(i*(len(data)-samplesize)/num_samples)
        output[i] = data[ (offset) : (offset + samplesize)]

    return output

def random_rotate(input):
    output = np.copy(input)
    for i in range(0, input.shape[0]):
        euler = np.random.uniform(0, np.pi, size=3)
        input_acc = input[i,:,0:3]
        input_rot = input[i,:,3:6]
        Rot = ins_tools.geometry_helpers.euler2mat(euler[0],euler[1],euler[2])
        output_acc = np.dot(Rot, input_acc.T).T
        output_rot = np.dot(Rot, input_rot.T).T
        output[i,:,:] = np.hstack((output_acc, output_rot))  
        
    return output

def traindata(G, *argv): #argv is data for each class
    dataset = argv[0]
    targets = np.zeros((argv[0].shape[0],1))
    for i in range(len(argv)-1):
        dataset = np.vstack((dataset,argv[i+1]))
        targets = np.vstack((targets, (i+1)*np.ones((argv[i+1].shape[0],1))))
    targets = targets.ravel()
    #parameters = {'kernel':(['rbf']), 'gamma':[0.00001, 0.0001, 0.001, 0.1], 'C':[ 1, 10, 100, 500]}
    #parameters = {'kernel':(['poly']), 'C':[100], 'gamma':[0.0001]}
    #svr = svm.SVC()
    #clf = GridSearchCV(svr, parameters)
    clf = svm.SVC(gamma=G, C=1, probability = False)
    clf.fit(dataset,targets)

    return clf

#input correctly processed datasets and their known target, and output the accuracy
def testdata(target, clf, data):
    targets = target*np.ones(data.shape[0])
    score = clf.score(data, targets)
    return score

    
def predict(data, clf, sample_len):
    data_norm = normdata(data)
    data_split = split_data(data_norm, 1000, sample_len)
    data_unravel = data_split.reshape(data_split.shape[0],data_split.shape[1]*6)
    return clf.predict(data_unravel)
