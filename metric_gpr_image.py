import sys
import os
from scipy import interpolate

import numpy as np
import matplotlib.pyplot as plt
import logging
from PIL import Image

from skimage.restoration import denoise_wavelet
from scipy import signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from measurement_structures import GprMeasurements
from signal_processing_utils import *

class GprImage:
  """Constructs a GPR image from a linear trajectory segment."""

  MAX_VALUE = None
  MAX_TRACE_VALUE = None
  ZERO_POSITION = None

  def __init__(self, config, gpr_m, im_std, zero_position):
    self.config = config
    
    self.t_begin = gpr_m.times[0]
    self.t_end = gpr_m.times[-1]

    self.zero_position = zero_position

    self.min_range = np.min(gpr_m.ranges)
    self.max_range = np.max(gpr_m.ranges)

    self.direction = gpr_m.directions

    self.average_trace = None
    self.im_std = None
    self.max_value = None

    self.time_to_range = interpolate.interp1d(
        gpr_m.times, gpr_m.ranges, fill_value="extrapolate")

    new_gpr = np.empty((gpr_m.measurements.shape[0],0))
    new_ranges = list()

    temp_measurement = gpr_m.measurements[:,0][:,None]

    for i in range(1, gpr_m.ranges.size):
      if not np.any(np.abs(np.array(new_ranges) - gpr_m.ranges[i]) < .03):
        new_ranges.append(gpr_m.ranges[i])
        temp_measurement = np.hstack((temp_measurement, 
                                      gpr_m.measurements[:,i][:,None]))

        new_gpr = np.hstack((new_gpr, 
                             np.average(temp_measurement, axis=-1)[:,None]))
        temp_measurement = np.empty((gpr_m.measurements.shape[0],0))

      else:
        temp_measurement = np.hstack((temp_measurement, 
                                      gpr_m.measurements[:,i][:,None]))

    num_rows = new_gpr.shape[0]

    self.y_values = np.arange(num_rows).astype(float)

    print(gpr_m.ranges.size, new_gpr.shape[1])

    self.range_to_gpr_f = interpolate.interp2d(
      new_ranges, self.y_values, new_gpr, kind="linear")

    self.average_trace = np.average(new_gpr, axis=-1)
  
    self.init = True
    # self.zero_position = 0
    self.max_term = 1

  def __contains__(self, other):
    if other >= self.t_begin and other <= self.t_end:
      return True

  def get_gpr_image_at_time(self, 
                            config, 
                            time, 
                            total_range, 
                            resolution, 
                            gt_point='center', 
                            v_flag=False):
    """Constructs image at the specified time.
    
    params:
      config: im_props yaml config formatted as an AttrDict.
      time: Time of image acquisition. If set to -1, then is all data.
      total_range: Size of image in meters.
      resolution: Sampling resolution of image in meters/pixel.
      gt_point: Location of timestamp in image.
      v_flag: Turn on visualization for debugging or demonstration.
    """
    # For time=-1, parse entire image at desired resolution.
    center = 0
    
    if time == -1:
      start = self.time_to_range(self.t_begin)
      end = self.time_to_range(self.t_end)
      num = int(np.abs(self.max_range - self.min_range) / resolution)
    
    # If -1 flag is not applied and time is not in range, return empty array.
    elif time < self.t_begin or time > self.t_end: 
      raise ValueError(f"Invalid time {time} provided.")
    
    else:
      num = int(total_range / resolution)
      
      if gt_point == 'left':
        start = self.time_to_range(time)
        end = start + self.direction*total_range
        
      elif gt_point == 'center':
        center = self.time_to_range(time)
        start = center - 1*self.direction*total_range/2
        end = center + self.direction*total_range/2
      
    if (start < self.min_range or start > self.max_range or 
        end < self.min_range or end > self.max_range): 
      return np.array([])

    sampling_points = np.linspace(start=start, stop=end, num=num)
    im_unfiltered = self.range_to_gpr_f(sampling_points, self.y_values)
    
    rows, cols = im_unfiltered.shape
    im_filtered = np.zeros((rows, cols))

    im_unfiltered = bgr(im_unfiltered, window=config.background_removal_window)
    im_unfiltered = dewow(config.dewow_params, im_unfiltered)

    if v_flag:
      plt.figure()
      plt.imshow(np.copy(im_unfiltered), cmap='gray')
      plt.title('dewowed')

    im_unfiltered = triangular(config.triangular_params, im_unfiltered)

    if v_flag:
      plt.figure()
      plt.imshow(np.copy(im_unfiltered), cmap='gray')
      plt.title('triangular')

    if config.background_removal:
      if config.background_removal_window == -1:
        im_unfiltered = im_unfiltered - np.average(im_unfiltered, axis=-1)[:,None]
      

    if v_flag:
      plt.figure()
      plt.imshow(np.copy(im_unfiltered), cmap='gray')
      plt.title('bgr')

    im_filtered[0:rows-self.zero_position,:] = im_unfiltered[self.zero_position:,:]

    if v_flag:
      plt.figure()
      plt.imshow(im_filtered, cmap='gray')
      plt.title('zero_time')

    im_filtered = sec_gain(im_filtered, 
                           a=config.sec_gain_params.a, 
                           b=config.sec_gain_params.b, 
                           threshold=config.sec_gain_params.thresh)

    if v_flag:
      plt.figure()
      plt.imshow(np.copy(im_filtered), cmap='gray')
      plt.title('gained')

    im_filtered = denoise_wavelet(im_filtered, multichannel=False)

    if v_flag:
      plt.figure()
      plt.imshow(np.copy(im_filtered), cmap='gray')
      plt.title('wavelet')

    if (config.gaussian_params.use and 
        config.gaussian_params.sigma > 0):

      im_filtered = gaussian_filter1d(im_filtered, 
                                      sigma=config.gaussian_params.sigma, 
                                      order=config.gaussian_params.order,
                                      axis=1)

    if v_flag:
      plt.figure()
      plt.imshow(np.copy(im_filtered), cmap='gray')
      plt.title('thresholded')

    return im_filtered

class MetricGprImage:
  """Creates metric (range-based) image from GPR measurements.

  Params:
    config: im_props yaml config formatted as an AttrDict. 
    gpr_m: GPR measurements and estimated ranges.
    imu_m: IMU measurements (accel, gyro, magnetometer).
    gt_point: Location of timestamp in image.
    v_flag: Turn on visualization for debugging or demonstration.
  """

  def __init__(self, config, gpr_m, imu_m, gt_point='center', v_flag=True):
    self.config = config
    # TODO(abaikovitz): Add gyro value to segment based on rotation as well.
    self.gpr_images = list()
    self.gpr_im_height = gpr_m.measurements.shape[0]
  
    self.t_begin = gpr_m.times[0]
    self.t_end = gpr_m.times[-1]

    self.gt_point = gt_point

    # Filter the initial range signal to remove high frequency noise.
    b, a = signal.butter(self.config.split.range.butter.order, 
                         self.config.split.range.butter.wn, 
                         btype='lowpass')
    ranges = signal.filtfilt(b, a, gpr_m.ranges)

    if imu_m.measurements.size > 0:
      inertial_meas = bw_filter(imu_m.measurements[3,:], 
                                imu_m.measurements[4,:], 
                                imu_m.measurements[5,:],
                                imu_m.measurements[0,:],
                                imu_m.measurements[1,:],
                                imu_m.measurements[2,:])

      self.accel_x, self.accel_y, self.accel_z, self.ang_x, self.ang_y, self.ang_z = inertial_meas

    velocities = np.diff(ranges)
    accelerations = np.diff(velocities)

    range_maxima = signal.argrelextrema(ranges, np.greater)
    range_minima = signal.argrelextrema(ranges, np.less)
    final_elem = np.array([ranges.size-1])
    all_range_peaks = np.sort(np.concatenate((range_maxima[0], 
                                              range_minima[0], 
                                              final_elem)))

    range_peaks_arr = [0]
    prev_peak = 0
    for peak in all_range_peaks:

      if prev_peak is not None and np.abs(ranges[peak]-ranges[prev_peak]) < 1:
        continue
      
      range_peaks_arr.append(peak)
      prev_peak = peak

    accel_peaks = signal.find_peaks(np.abs(accelerations), 
                                    height=self.config.split.range.accel)
    if self.config.split.gyro.use and imu_m.measurements.size > 0:
      ang_z_peaks = signal.find_peaks(np.abs(self.ang_z), 
                                      height=self.config.split.gyro.height, 
                                      distance=self.config.split.gyro.dist)

    traj_segments = range_peaks_arr

    if self.config.split.gyro.use and imu_m.measurements.size > 0:
      for ang_z_peak in ang_z_peaks[0]:
        peak_time = imu_m.times[ang_z_peak]
        range_peak = np.argmin(np.abs(gpr_m.times - peak_time))

        if np.any(np.abs(traj_segments - range_peak) > 20):
          traj_segments = np.append(traj_segments, range_peak)

    self.traj_segments = np.sort(traj_segments)
    
    print(self.traj_segments)
    if v_flag:
      fig, [ax1, ax2, ax3] = plt.subplots(3,1)
      # Plot ranges with filtered trajectory.
      ax1.plot(gpr_m.ranges)
      ax1.plot(ranges)
      ax1.plot(self.traj_segments, ranges[self.traj_segments], '*')
      ax1.set_ylabel("Range [m]")
      ax1.legend(["Unfiltered", "Filtered", "Accepted Peaks"])

      # Plot filtered velocity.
      ax2.plot(velocities)
      ax2.set_ylabel("Velocity [m/s]")

      # Plot filtered acceleration.
      ax3.plot(accelerations)
      ax3.plot(accel_peaks[0], accelerations[accel_peaks[0]], '*')
      ax3.set_ylabel("Acceleration [m/s^2]")
      ax3.set_xlabel("Range Measurement Index")
      ax3.legend(["Acceleration", "Accepted Peaks"])

      fig.suptitle("Range, Velocity, Acceleration from Wheel Encoder")

      if v_flag and self.config.split.gyro.use and imu_m.measurements.size > 0:
        plt.figure()
        plt.plot(imu_m.measurements[2,:])
        plt.plot(self.ang_z)
        plt.plot(ang_z_peaks[0], self.ang_z[ang_z_peaks[0]], '*')

        plt.xlabel("Gyroscope Measurement Index")
        plt.ylabel("Gyroscope Value [rad/s]")
        plt.title("Gyroscope Measurements")

        plt.legend(["Unfiltered", "Filtered"])

        plt.figure()
        plt.plot(imu_m.measurements[3,:])
        plt.plot(self.accel_x)

        plt.xlabel("Accelerometer Measurement Index")
        plt.ylabel("Acceleration [m/s]")
        plt.title("Accelerometer X Measurements")

        plt.legend(["Unfiltered", "Filtered"])
    
    gained=sec_gain(gpr_m.measurements, 
                             a=config.sec_gain_params.a, 
                             b=config.sec_gain_params.b, 
                             threshold=config.sec_gain_params.thresh)
    if config.gaussian_params.use:
      im_std = np.std(gaussian_filter1d(gained, 
                                        config.gaussian_params.sigma, 
                                        order=config.gaussian_params.order))
    else:
      im_std = np.std(gained)
    
    for i in range(self.traj_segments.size-1):
      i1 = self.traj_segments[i]
      i2 = self.traj_segments[i+1]

      gpr_seg = GprMeasurements(gpr_m.measurements.shape[0])
      gpr_seg.measurements = gpr_m.measurements[:,i1:i2]      
      gpr_seg.times = gpr_m.times[i1:i2]
      gpr_seg.ranges = gpr_m.ranges[i1:i2]
      gpr_seg.directions = np.sign(np.average(velocities[i1:i2]))

      if gpr_seg.times.size == 0:
        print("Invalid time segment.")
        continue
      self.gpr_images.append(GprImage(config,
                                      gpr_seg, 
                                      im_std, 
                                      zero_position=np.average(
                                        gpr_m.measurements, axis=-1).argmin()))
  
    large_im = self.get_gpr_image(config, -1, -1, self.config.resolution)
    self.maxI = config.std_max * np.std(large_im)
    self.minI = -1*self.maxI

  def get_gpr_image(self, config, time, total_range, resolution):
    # If time is -1, then provide the entire image.
    if time == -1:
      total_im = np.empty((self.gpr_im_height, 0))
      for gi in self.gpr_images:
        im = gi.get_gpr_image_at_time(config, 
                                      time, 
                                      total_range, 
                                      resolution, 
                                      gt_point=self.gt_point)

        if im.size != 0:
          total_im = np.hstack((total_im, im))

      return total_im

    if time < self.t_begin or time > self.t_end:
      print(f"Invalid time {time} provided.\t"
            f"Not within range: [{self.t_begin}, {self.t_end}].")
      np.array([])

    # Otherwise, search for time in the images.
    for i, gi in enumerate(self.gpr_images):
      if time in gi:
        im = gi.get_gpr_image_at_time(config, 
                                      time,
                                      total_range,
                                      resolution, 
                                      gt_point=self.gt_point)
        return im

    # If the time is not in the test space, then return nothing.
    return np.array([])

  def get_odom_and_dir(self, time):
    for gi in self.gpr_images:
      if time in gi:
        rng = gi.time_to_range(time)
        direction = gi.direction

        return rng, direction

    print("Invalid time provided to get_odom_and_dir.")
    return np.array([])

  def create_gpr_image(self, im_filtered, path, file_name, v_flag=False):

    if not os.path.isdir(path):
      os.mkdir(path)

    img_filename = file_name + ".png"
    img_path = os.path.join(path, img_filename) 

    if len(file_name) > 0:
      plt.imsave(img_path, 
                im_filtered, 
                cmap='gray',
                format='png',
                vmin=self.minI, 
                vmax=self.maxI)

    if v_flag:
      plt.figure()
      plt.imshow(im_filtered, 
                cmap='gray', 
                vmin=self.minI, 
                vmax=self.maxI
                )
