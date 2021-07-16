import numpy as np
import ins_tools.SVM as svm
from sklearn.externals import joblib #Used for saving the model
import sys
sys.path.append('../')

walk_dirs = ['data/vicon/processed/2017-11-22-11-22-03',
             'data/vicon/processed/2017-11-22-11-22-46',
             'data/vicon/processed/2017-11-22-11-25-20',
             'data/vicon/processed/2017-11-22-11-40-44',        
             ]

run_dirs = ['data/vicon/processed/2017-11-22-11-44-47',
            'data/vicon/processed/2017-11-27-11-11-24',
            'data/vicon/processed/2017-11-27-11-11-53',
            'data/vicon/processed/2017-11-27-11-12-18',
            'data/vicon/processed/2017-11-27-11-13-10',
            ]
stair_dirs = ['data/stairs/train/2018-07-24-16-16-52/processed_data',
              'data/stairs/train/2018-07-24-16-20-53/processed_data' ,
              'data/stairs/train/2018-07-24-16-50-37/processed_data' ,
              'data/stairs/train/2018-07-24-16-53-05/processed_data',
              'data/stairs/train/2018-07-24-16-55-02/processed_data',
              'data/stairs/train/2018-07-24-16-56-23/processed_data',
              ]

samples_per_file = 2000
sample_duration = 200 #samples consist of 2 seconds of inertial data
walk_train, walk_test  = svm.importdata(walk_dirs,crop=300, samples_per_file=samples_per_file, seq_len=sample_duration)
run_train, run_test = svm.importdata(run_dirs,crop=300, samples_per_file=samples_per_file, seq_len=sample_duration)
stair_train, stair_test = svm.importdata(stair_dirs, crop=300, samples_per_file=samples_per_file, seq_len=sample_duration)

## uncomment to train instead of loading the current one
#clf = svm.traindata(0.001, walk_train, run_train, stair_train)
#joblib.dump(clf, 'results/3class_vn100_2_new.pkl', compress=1)
#print("testing accuracy")
clf = joblib.load('results/3class_vn100_2.pkl')

motionlist = [walk_test, run_test, stair_test]
c=0
m = len(motionlist)

#Test with new data:
walk_acc=np.zeros(m)
run_acc = np.zeros(m)

acc = np.zeros((m,m))
for motion in motionlist:
    print('motion: ', c)
    for i in range(m):
        acc[c,i]=svm.testdata(i,clf,motion)
        print('subject: ',i)
    c+=1
print(acc)
