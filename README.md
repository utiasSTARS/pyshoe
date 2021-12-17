# PyShoe

Code for "Robust Data-Driven Zero-Velocity Detection for Foot-Mounted Inertial Navigation": an open source foot-mounted, zero-velocity-aided inertial navigation system (INS) that includes implementations of four classical zero-velocity detectors, in addition to our two proposed learning-based detectors. We provide three mixed-motion inertial datasets and evaluation code tools for comparison of the various zero-velocity detection methods.

<img src="https://github.com/utiasSTARS/pyshoe/blob/master/data/main_figure.png" width="400px"/>

## Dependencies:
* numpy
* scipy 
* [scikit-learn](https://scikit-learn.org/stable/) to run the adaptive zero-velocity detector (version 0.19.1) Note: Python 3.7 => version 0.19.2
* [pytorch](https://pytorch.org/) to run the LSTM-based zero-velocity classifier

Scikit-Learn and PyTorch do not need to be installed if you do not intend to use our zero-velocity detectors.  You must remove the import of LSTM and SVM from ins_tools/EKF.py to do so.

# Datasets

Download the full inertial dataset (~2.9 GB) by running the following bash script from the source directory (to automatically download and extract the dataset):

```
bash download_data.sh
```

Alternatively, the dataset can be downloaded through [IEEE dataport](https://ieee-dataport.org/open-access/university-toronto-foot-mounted-inertial-navigation-dataset), which is accessible without any membership.

We currently provide three separate inertial datasets in the full package:

* VICON Dataset: Stored in `data/vicon`. Collected within a ~3x3m motion capture area. There are 56 trials in total, and all of the raw data has been processed into a .mat file in `data/vicon/processed`. This dataset has complete position ground truth. 

* Hallway Dataset: Stored in `data/hallway`.  Consists of walking and running motions along three hallways (39 trials in total). There are intermediate ground truth positions along the path: we provide the time steps (indices of the IMU sequence) that correspond with the ground truth positions. 

* Stair Dataset: Stored in `data/stairs`.  Consists of stairway trials (six training trials and eight testing trials). There is intermediate vertical position ground truth for each flight: we provide the time steps (indices of the IMU sequence) that correspond with these locations.

# Zero-Velocity-Aided INS
We have open-sourced our Python-based zero-velocity-aided INS that uses an error-state extended Kalman filter to fuse zero-velocity pseudo-measurements that are produced by a zero-velocity detector.  We refer to our papers and the citations therein for more technical details about the INS. We would like to acknowledge [openshoe](http://www.openshoe.org/); we based some components of our zero-velocity-aided INS on their open-source MATLAB implementation.

We have included a script, `ins_demos.py`, with five separate demos - this sample code shows how to run our INS with data from the three datasets that are provided. Two of the demos show how our proposed learning-based zero-velocity detectors can be used.

# Reproduction of Paper Results
We have included two scripts that reproduce the main results in our paper:

* `process_error_hallway.py` 

This script reproduces Table 3 of the paper by iteratively evaluating the error of specified zero-velocity detectors (using any specified list of zero-velocity thresholds) for all trials in the hallway dataset. Two CSV files are automatically generated in the results folder: `hallway_results_raw.csv`, which shows the error for all individual trials, and `hallway_results.csv`, which is a processed version that reproduces the paper results. The processed results are saved to `results/hallway_results.csv`.

* `plot_hallway_data.py` 

This script generates plots for all of the hallway trials within `results/figs/hallway/`.

* `process_error_stairs.py`

This script reproduces Table 4 of the paper by iteratively evaluating the error of the specified zero-velocity detectors (each with a list of threshold values) for all trials in the stair dataset Two CSV files are automatically generated in the results folder: `stair_results_raw.csv`, which shows the error for all individual trials, and `stair_results.csv`, which is a processed version that reproduces the paper results. The processed results are saved to `results/stair_results.csv`. 

* `plot_stair_data.py` 

This script generates plots for all of the hallway trials within `results/figs/hallway/`.

# Additional Scripts
* `run_threshold_optimization`: Generates the zero-velocity labels that we used to train our LSTM-based zero-velocity classifier. The zero-velocity thresholds for the five zero-velocity detectors (SHOE, ARED, AMVD, MBGTD, VICON) are optimized for each VICON dataset trial, and the most accurate detector's output is used as the ground truth label for each trial.

* `process_error_vicon.py`: Iteratively evaluates the error of the five classical zero-velocity detectors when their optimized zero-velocity outputs are used. The results are saved in `results/vicon_results_raw.csv`. 

* `train_motion_classifier.py`: Trains a three-class motion classifier (walk vs. run vs. stairs) using a subset of the training data, and will reproduce the accuracy of the classifier for the validation set.

# Citation
If you use this code in your research, please cite:
```
@article{2019_Wagstaff_Robust,
  title = {Robust Data-Driven Zero-Velocity Detection for Foot-Mounted Inertial Navigation},
  author = {Brandon Wagstaff and Valentin Peretroukhin and Jonathan Kelly},
  journal = {IEEE Sensors Journal},
  pages = {957--967},
  number = {2},
  url = {https://arxiv.org/abs/1910.00529},
  volume = {20},
  year = {2019}
}
```
```
@inproceedings{2018_Wagstaff_LSTM-Based,
  address = {Nantes, France},
  author = {Brandon Wagstaff and Jonathan Kelly},
  booktitle = {Proceedings of the International Conference on Indoor Positioning and Indoor Navigation {(IPIN'18)}},
  date = {2018-09-24/2018-09-27},
  month = {Sep. 24--27},
  title = {LSTM-Based Zero-Velocity Detection for Robust Inertial Navigation},
  url = {http://arxiv.org/abs/1807.05275},
  year = {2018}
}
```
```
@inproceedings{2017_Wagstaff_Improving,
  address = {Sapporo, Japan},
  author = {Brandon Wagstaff and Valentin Peretroukhin and Jonathan Kelly},
  booktitle = {Proceedings of the International Conference on Indoor Positioning and Indoor Navigation {(IPIN'17)}},
  date = {2017-09-18/2017-09-21},
  doi = {10.1109/IPIN.2017.8115947},
  month = {Sep. 18--21},
  title = {Improving Foot-Mounted Inertial Navigation Through Real-Time Motion Classification},
  url = {http://arxiv.org/abs/1707.01152},
  year = {2017}
}
```
