# pyshoe
Code for "Robust Data-Driven Zero-Velocity Detection for Foot-Mounted Inertial Navigation": an open source foot-mounted, zero-velocity-aided inertial navigation system (INS) that includes implementations of four classical zero-velocity detectors, in addition to our two proposed learning-based detectors. We provide three mixed-motion inertial datasets and evaluation code for the comparison of the various zero-velocity detection methods.

<img src="https://github.com/utiasSTARS/pyshoe/blob/master/main_figure.png" width="400px"/>

## Dependencies:
* numpy
* scipy 
* [scikit-learn](https://scikit-learn.org/stable/) to run the adaptive zero-velocity detector
* [pytorch](https://pytorch.org/) to run the LSTM-based zero-velocity classifier

Scikit-Learn and Pytorch do not need to be installed if you do not intend to use our zero-velocity detectors.  You must remove the import of LSTM and SVM from ins_tools/EKF.py to do so.

# Datasets

Download the inertial dataset [here](https://drive.google.com/open?id=1eMjS3DCNwnkbHXt9knmGAcLB8CI4G27h) or simply run the following bash script from the source directory to automatically download and extract the dataset:

```
bash download_data.sh
```
We currently provide three inertial datasets: 

* VICON Dataset: Stored in `data/vicon`.  Collected within a ~3x3m motion capture area.  There are 56 trials in total, and all of their raw data is process into a .mat file in `data/vicon/processed`. This dataset has complete position ground truth. 

* Hallway Dataset: Stored in `data/hallway`.  Consists of walking and running along three hallways (39 trials in total).  There are intermediate ground truth locations along the path: we provide the timesteps (indices of the IMU sequence) that correspond with the ground truth location. 

* Stair Dataset: Stored in `data/stairs`.  Consists of stairway trials (six training trials and eight testing trials).  There is intermediate vertical position ground truths at each flight: we provide the timesteps (indices of the IMU sequence) that correspond with these locations.

# Zero-Velocity-Aided INS
We open-source our Python-based zero-velocity-aided INS that uses an error-state extended Kalman filter to fuse the motion model with the zero-velocity measurements that are produced by a zero-velocity detector.  We refer to our past papers and their citations for more technical details of the INS. We acknowledge [openshoe](http://www.openshoe.org/) as we have based some components of our zero-velocity-aided INS on their open-source Matlab implementation.

We include a script `ins_demos.py` with five separate demos that provide sample code for running our INS with data from the three datasets that are provided. Two of the demos indicate how our proposed learning-based zero-velocity detectors can be used.

# Reproduction of Paper Results
We include two scripts that reproduce the two main results tables in our paper:

* `process_error_hallway.py` reproduces table 3 of the paper by iteratively evaluating the error of specified zero-velocity detectors (using any specified list of zero-velocity thresholds) for all trials in the hallway dataset. Two csv files are automatically generated in the results folder: `hallway_results_raw.csv`, which shows the error for all individual trials, and `hallway_results.csv`, which is a processed version that reproduces paper results. The processed results are saved to `results/hallway_results.csv`.  `plot_hallway_data.py` generates plots for all of the hallway trials within `results/figs/hallway/`,

* `process_error_stairs.py`: Reproduces table 4 of the paper by iteratively evaluating the error of the specified zero-velocity detectors (each with a list of threshold values) for all trials in the stair dataset Two csv files are automatically generated in the results folder: `stair_results_raw.csv`, which shows the error for all individual trials, and `stair_results.csv`, which is a processed version that reproduces the paper results. The processed results are saved to `results/stair_results.csv`. `plot_stair_data.py` generates plots for all of the hallway trials within `results/figs/hallway/`.

# Additional Scripts
* `run_threshold_optimization`: Generates the zero-velocity labels that we used to train our LSTM-based zero-velocity classifier. The zero-velocity thresholds for the five zero-velocity detectors (SHOE, ARED, AMVD, MBGTD, VICON) are optimized for each VICON dataset trial, and the most accurate detector's output is used as the ground truth label for each trial.

* We additionally include a script `process_error_vicon.py` that iteratively evaluates the error of the 5 classical zero-velocity detectors when their optimized zero-velocity outputs are used.  The results are saved in `results/vicon_results_raw.csv`. 

* `train_motion_classifier.py` will train a three-class motion classifier (walk vs. run vs. stairs) using a subset of the training data, and will reproduce the accuracy of the classifier for the validation set.

# Citation
If you use this code in your research, please cite:

TBA

