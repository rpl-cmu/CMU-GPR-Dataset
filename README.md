# CMU-GPR-Dataset

## Introduction

<center><img src="misc/title_drawing.png" width="750" style="center"></center>
&nbsp;

The CMU-GPR dataset contains trajectory sequences, which include subsurface measurements from ground penetrating radar (GPR) along with conventional proprioceptive sensors (wheel odometer, IMU). One benefit of radar-based perception is its robust performance in challenging weather conditions. Beyond more traditional imaging radar systems being used actively on vehicles, surface penetrating radar allows for additional robustness to spatio-temporal change. While the line-of-sight environment may change over time, subsurface features remain mostly consistent.

The CMU-GPR dataset consists of 15 sequences containing synchronized odometry, subsurface, and ground truth measurements. In this experimentation, a single-channel, off-the-shelf Sensors and Software Noggin 500 GPR was used. This GPR provides 1D measurements at each location, which can be used to construct 2D images through motion. Each sequence contains revisitation events, where subsurface features are observed more than once.


## Dependencies

All utility functions are tested using Python 3.6.12 and the following libraries:

- numpy (v1.19.1)
- scipy (v1.4.1)
- hydra (v1.0.5)
- matplotlib (v3.1.3)
- tqdm (v4.58.0)
- pywt (v1.1.1)
- skimage (v0.16.2)
- logging (v0.5.1.2)


## GPR Image Construction

The process to construct 2D, interpretable GPR images from 1D traces is summarized in the figure below.

<img src="misc/gpr.png" width="500" style="center">

(a) Unprocessed localized
traces received by the device. (b) Horizontally stacked traces from (a), where
the amplitudes correspond to pixel intensity. (c) Measurements after filtering
and gain. (d) Final image after thresholding.

The routine to create these images is provided in `metric_gpr_image.py`, where the following operations are applied:

- Raw traces are collected with their corresponding wheel odometry measurements.
- Rubber band interpolation is applied to produce a uniformly spaced image.
- Background removal is applied to remove repetitive horizontal banding in the data. This can be applied over small windows or all available data.
- Dewow filter is applied to remove DC bias and low-frequency noise. This is caused by antenna saturation.
- Triangular bandpass filter to remove high frequency noise.
- Zero time correction at the highest amplitude of the first airwave.
- User-defined Spreading and Exponential Compensation (SEC) gain to improve the visibility of deeper objects.
- Wavelet denoising to further remove high frequency noice.
- Gaussian filter to blur the processed image as a final noise reduction pass.


## System Parameters

In order to create this dataset, we constructed a manually-pulled test article named SuperVision, containing a:

- Sensors and Software OEM Noggin 500 GPR
- XSENS MTi-30 9-axis IMU
- YUMO quadrature encoder with 1024 PRR
- Intel RealSense D435 (RGB images at 15Hz only)
- Leica TS15 Total Station (used for ground truth)

The system architecture is described in the image below:

<center><img src="misc/supervision.png" width="300" style="center"></center>

Drawings of system setup with an approximate extrinsic calibration will be available shortly.

## Dataset Files

The dataset is available at the links shown in the table below. Data is available in the form of individual sequence files as well as files containing the entire set of sequences in particular locations. Additional data is provided without ground truth, but with highly accurate wheel odometry, which can be used in model construction and validation.

**Sequences with ground truth positions:**
| Sequence Number | Location | Filename | Correlated Sequences | Size (MB) | Link |
| -- | -- | -- | -- | -- | -- |
| A.0  |  gates_g | 1613063428-0-gates_g-cmu-gpr.zip | --  | 596.4  | [[Link]](https://drive.google.com/file/d/17ITUXjN0GIF6t1DgDQuWlWgNAty9jZH0/view?usp=sharing) |
| A.1  |  gates_g | 1613063708-0-gates_g-cmu-gpr.zip | --  | 380.2  | [[Link]](https://drive.google.com/file/d/1PvTPZKp6kLRO1KJS2CUXj2QM1vCO6cuh/view?usp=sharing) |
| A.2  |  gates_g | 1613063877-0-gates_g-cmu-gpr.zip | --  | 727.3  | [[Link]](https://drive.google.com/file/d/1v9HXvZywzaaLyXHtDxrnEHT4W85o7FLp/view?usp=sharing) |
| A.3  |  gates_g | 1613064209-0-gates_g-cmu-gpr.zip | --  | 965.5  | [[Link]](https://drive.google.com/file/d/11qnJ0H6s4d3b8PnrlNPj1jzJX5Ebsfzo/view?usp=sharing) |
| A.4  |  gates_g | 1613064646-0-gates_g-cmu-gpr.zip | --  | 623.0  | [[Link]](https://drive.google.com/file/d/1tJA6YBZOGP4uHb5DAVM9z0RN1Apn4LaA/view?usp=sharing) |
| A.5  |  gates_g | 1613064932-0-gates_g-cmu-gpr.zip | --  | 493.8  | [[Link]](https://drive.google.com/file/d/1wXVETKcLPPM9jtxYa5drsCT4eUGxDG2R/view?usp=sharing) |
| A.6  |  gates_g | 1613065150-0-gates_g-cmu-gpr.zip | --  | 89.2  | [[Link]](https://drive.google.com/file/d/1-saceUYe8sZslHAGQuCiuzKzBhEwJRCR/view?usp=sharing) |
| A.7  |  nsh_b | 1613059265-0-nsh_b-cmu-gpr.zip | --  | 718.2  | [[Link]](https://drive.google.com/file/d/19Pt9HxOCgSuNsjWb9-im1wpaHN1Fn7Pi/view?usp=sharing) |
| A.8  |  nsh_b | 1613059477-0-nsh_b-cmu-gpr.zip | --  | 492.9  | [[Link]](https://drive.google.com/file/d/1jgTCpVGW1unvA04d5vQnuIPfZoUN4W-s/view?usp=sharing) |
| A.9  |  nsh_b | 1613059699-0-nsh_b-cmu-gpr.zip | --  | 680.9  | [[Link]](https://drive.google.com/file/d/1QwjiqPNfjDvwvAeLQ1NJGeeJ57ZgP-PX/view?usp=sharing) |
| A.10  | nsh_b | 1613059996-0-nsh_b-cmu-gpr.zip | --  | 1,193.0  | [[Link]](https://drive.google.com/file/d/1RzHPNT421uTjZ_pVIckfQV56z19enMDW/view?usp=sharing) |

**Full unprocessed datasets:**
| Sequence Number | Location | Filename | Correlated Sequences | Size (MB) | Link |
| -- | -- | -- | -- | -- | -- |
| B.0 |  gates_g | 1613063411-767709970-gates_g_all-cmu-gpr.zip | --  | 3,997.1  | [[Link]](https://drive.google.com/file/d/1o8GBDv3d1qlOGaVfzocIPFAaOEWwDmS7/view?usp=sharing) |
| B.1 |  nsh_b | 1613059186-574372053-nsh_b_all-cmu-gpr.zip | --  | 3,205.3  | [[Link]](https://drive.google.com/file/d/1qoHHCSUx-PUPfOEek2GftmWOAfqqY_Zt/view?usp=sharing) |
| B.2 |  nsh_h | 1613061585-228924036-nsh_h_all-cmu-gpr.zip | --  | 1,258.4  | [[Link]](https://drive.google.com/file/d/1N0bDwdnkmr7xCjS5LhK3JI10Pte4YNij/view?usp=sharing) |


**Pure odometry data:** 

*Useful for training, debugging, and small experiments*

| Sequence Number | Location | Filename | Correlated Sequences | Size (MB) | Link |
| -- | -- | -- | -- | -- | -- |
| C.0 |  nrec | 1611959465-373785018-nrec-cmu-gpr.zip | --  | 11  | [[Link]](https://drive.google.com/file/d/1m56aXM7P-UKVzbGb0hLFw5TxzlezQHVN/view?usp=sharing) |
| C.1 |  nrec | 1611959921-603359937-nrec-cmu-gpr.zip | --  | 32.3  | [[Link]](https://drive.google.com/file/d/10XCwAfRfrrKVbGaa1U3QvT1ygDH7xXbX/view?usp=sharing) |
| C.2 |  smith | 1612204529-582686901-smith-cmu-gpr.zip | --  | 32.3  | [[Link]](https://drive.google.com/file/d/1mcKLDv2Y4EaPNRNmudlqJhwjXEs_PAXX/view?usp=sharing) |



## Directory format
```bash
cmu-gpr-dataset
├── time_s-time_ns-loc-cmu-gpr
│   ├── camera
│   │   ├── <timestamp_s>.png
│   │   └── ...
│   ├── ts_meas.csv
│   ├── imu_meas.csv
│   ├── gpr_meas.csv
│   └── we_odom.csv
└── ...
```

## Data Format

The data format for each type of measurement is shown below.

<img src="misc/data_type.png" width="500" style="center">

The amplitude of the GPR measurements can be represented in millivolts by dividing by 32767 and multiplying by 50 (as specified by the manufacturer).


## Utility Functions

The general interface for accessing GPR submaps is summarized below.

<img src="misc/api.png" width="500" style="center">



## Citation

If you use this dataset in your research, please cite the following papers:

```bibtex
@misc{baikovitz2021dataset,
      title={CMU-GPR Dataset: Ground Penetrating Radar Dataset for Robot Localization and Mapping}, 
      author={Alexander Baikovitz and Paloma Sodhi and Michael Dille and Michael Kaess},
      year={2021}
}
```

```bibtex
@misc{baikovitz2021ground,
      title={Ground Encoding: Learned Factor Graph-based Models for Localizing Ground Penetrating Radar}, 
      author={Alexander Baikovitz and Paloma Sodhi and Michael Dille and Michael Kaess},
      year={2021},
      eprint={2103.15317},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## Additional Details
<!-- TODO add the paper to something and link. -->
Additional details about the CMU-GPR dataset can be found here: [[Paper]](misc/baikovitz2021dataset.pdf).

An example using the data collected can be used can be found here: [[Paper]](https://arxiv.org/abs/2103.15317).

## Licence
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License and is intended for non-commercial academic use. If you are interested in using the dataset for commercial purposes please contact us at abaikovitz@cmu.edu.

:heart::robot: