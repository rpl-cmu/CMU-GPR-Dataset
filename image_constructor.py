import os, sys
import numpy as np
import matplotlib.pyplot as plt
import logging 

from scipy import interpolate
from tqdm import tqdm

from measurement_structures import (TotalStationMeasurements, 
                                    GprMeasurements, 
                                    OdometryMeasurements, 
                                    ImuMeasurements)
from metric_gpr_image import MetricGprImage
import signal_processing_utils

class ImageConstructor:
  MOTION_THRESHOLD = 0.001 # meters.

  # Standard data csv files.
  WHEEL_ODOM_MEAS_FILE = 'we_odom_meas.csv'
  TS_MEAS_FILE = 'ts_meas.csv'
  IMU_MEAS_FILE = 'imu_meas.csv'
  GPR_MEAS_FILE = 'gpr_meas.csv'

  def __init__(self, config):
    self.config = config

    self.directory_paths = config.directory_paths

    self.im_props = config.im_props
    self.ground_truth_point = config.ground_truth_point
    self.output_path = config.training_data_output_path

    if not os.path.isdir(self.output_path):
      os.mkdir(self.output_path)

    self.data = list()
    self.load_data(config.directory_paths)

    self.image_size = config.submaps.training_data_image_size
    self.image_range = config.submaps.training_data_image_range
    self.image_resolution = config.submaps.training_data_image_resolution

    self.current_gpr_measurement = None
    self.current_ts_measurement = None

    self.max_time = config.training_data_timeout # seconds.

    self.index = 0

    self.save_flipped_image = config.save_flipped_image

    self.ts_spacing = config.submaps.ts_spacing
    self.we_spacing = config.submaps.we_spacing

    # Visualization code.
    self.visualize_peaks = config.visualize.imu_peaks
    self.visualize_image = config.visualize.gpr_image

  def load_data(self, directory_path):    
    for path in self.directory_paths:

      gpr_path = os.path.join(path, ImageConstructor.GPR_MEAS_FILE)
      we_path = os.path.join(path, ImageConstructor.WHEEL_ODOM_MEAS_FILE)
      ts_path = os.path.join(path, ImageConstructor.TS_MEAS_FILE)
      imu_path = os.path.join(path, ImageConstructor.IMU_MEAS_FILE)

      gpr_m = GprMeasurements.load_from_file(gpr_path)
      we_m = OdometryMeasurements.load_from_file(we_path)
      ts_m = TotalStationMeasurements.load_from_file(ts_path)
      imu_m = ImuMeasurements.load_from_file(imu_path)

      # Assume constant velocity between wheel encoder measurements
      # to determine the position of GPR traces.
      time_to_range = interpolate.interp1d(we_m.times, 
                                           we_m.measurements, 
                                           fill_value="extrapolate")
      gpr_m.ranges = time_to_range(gpr_m.times)

      self.data.append([we_m, ts_m, gpr_m, imu_m])

  def create_radargram(self):

    for measurement in self.data:
      _, _, gpr_m, imu_m = measurement

      im = MetricGprImage(self.im_props, gpr_m, imu_m, v_flag=True)
      image = im.get_gpr_image(self.im_props, -1, -1, self.image_resolution)

      timestamp_start_gpr = gpr_m.times[0]
      file_name = f"radargram_{timestamp_start_gpr}.png"
      im.create_gpr_image(image, "../Results", file_name, v_flag=True)


  def create_submaps(self):
    for i, measurement in enumerate(self.data):
      we_m, ts_m, gpr_m, imu_m = measurement
      self.index = i
      # Construct image from run using private member function.

      if ts_m.times.size == 0 and we_m.times.size > 0:
        logging.info(f"Using pure wheel encoder odometry without ground truth for bagfile {i}.")
        self._construct_odom_image(gpr_m, we_m, imu_m)

      elif ts_m.times.size > 0 and we_m.times.size > 0:
        logging.info(f"Using wheel encoder odometry with total station stamp for bagfile {i}.")
        self._construct_odom_image_with_ts(gpr_m, ts_m, imu_m)

      else:
        logging.info(f"No valid measurement configuration for bagfile {i}")

  def _construct_odom_image_with_ts(self, gpr_m, ts_m, imu_m):
    im = MetricGprImage(self.im_props, gpr_m, imu_m, v_flag=self.visualize_peaks, gt_point=self.ground_truth_point)
    
    prev_meas = None
    
    with tqdm(total=ts_m.measurements.shape[1]) as pbar:

      for i in range(ts_m.measurements.shape[1]):
        pbar.update()
        curr_meas = ts_m.measurements[:2,i]

        if prev_meas is None or np.linalg.norm(curr_meas - prev_meas) >= self.ts_spacing:
          ts_time = ts_m.times[i]
          ts_measurement = ts_m.measurements[:,i]
          time_stamp_elem = str(ts_time).split('.')
          yaw = int(np.degrees(ts_m.directions[i]) + 180)

          if ts_time < im.t_begin or ts_time > im.t_end:
            continue

          image = im.get_gpr_image(self.im_props, ts_time, self.image_range, self.image_resolution)

          if image.size == 0:
            continue

          odom, direction = im.get_odom_and_dir(ts_time)

          file_name = (f"{time_stamp_elem[0]}_{time_stamp_elem[1]}"
                      f"_X_{np.round(ts_measurement[0], 4)}" 
                      f"_Y_{np.round(ts_measurement[1], 4)}_T"
                      f"_yaw_{np.round(yaw, 1)}"
                      f"_odom_{np.round(odom, 4)}"
                      f"_dir_{direction}"
                      f"_{self.index}")
          
          im.create_gpr_image(image, self.output_path, file_name, v_flag=self.visualize_image)

          prev_meas = curr_meas
        
  def _construct_odom_image(self, gpr_m, we_m, imu_m):
    im = MetricGprImage(self.im_props, gpr_m, imu_m, v_flag=self.visualize_peaks)
    
    prev_meas = None

    with tqdm(total=we_m.measurements.size) as pbar:

      for i in range(we_m.measurements.size):
        pbar.update()
        curr_meas = we_m.measurements[i]

        if prev_meas is None or np.abs(prev_meas - curr_meas) >= self.we_spacing:
          we_time = we_m.times[i]
          time_stamp_elem = str(we_time).split('.')

          if we_time < im.t_begin or we_time > im.t_end:
            continue

          image = im.get_gpr_image(self.im_props, we_time, self.image_range, self.image_resolution)

          if image.size == 0:
            continue

          odom, direction = im.get_odom_and_dir(we_time)

          file_name = (f"{time_stamp_elem[0]}_{time_stamp_elem[1]}"
                      f"_odom_{np.round(odom, 4)}"
                      f"_dir_{direction}"
                      f"_{self.index}")

          im.create_gpr_image(image, self.output_path, file_name, v_flag=self.visualize_image)

          prev_meas = curr_meas

  def _find_position(self, search_direction, ts_idx, desired_range):
    arr_size = self.current_gpr_measurement.ranges.size
    init_time = self.current_ts_measurement.times[ts_idx]
    init_meas = self.current_ts_measurement.measurements[:2,ts_idx]
    init_range = self.current_ts_measurement.ranges[ts_idx]

    current_time = init_time

    low_value = init_range - desired_range/2
    high_value = init_range + desired_range/2

    # Find central GPR measurement index that is closest to total station time.
    idx = np.abs(self.current_gpr_measurement.times - init_time).argmin()
    
    while np.abs(init_time - current_time) < self.max_time:
      
      idx += search_direction

      if idx < 0 or idx >= arr_size:
        return -1, 0

      current_time = self.current_gpr_measurement.times[idx]
      current_range = self.current_gpr_measurement.ranges[idx]

      ts_idx_curr = np.abs(self.current_ts_measurement.ranges - current_range).argmin()
      current_dist = np.linalg.norm(init_meas - self.current_ts_measurement.measurements[:2,ts_idx_curr])
      
      if current_range > high_value and current_dist >= desired_range:
        return idx, 1

      elif current_range < low_value and current_dist >= desired_range:
        return idx, -1
    return -1, 0

  def _construct_images_with_total_station(self, ts_m, gpr_m):
    """Constructs all valid GPR images centered at the total station measurement.

    Args:
      ts_m (:obj:`TotalStationMeasurements`) total station measurements over
        desired period.
      gpr_m (:obj:`GprMeasurements) GPR measurements over desired period.
    """
    self.current_gpr_measurement = gpr_m
    self.current_ts_measurement = ts_m
    prev_position = np.array([])

    for i in range(ts_m.ranges.size):
      
      total_station_time = ts_m.times[i]
      total_station_range = ts_m.ranges[i]
      total_station_measurement = ts_m.measurements[:,i]

      if (prev_position.size != 0 and 
          np.linalg.norm(total_station_measurement-prev_position) 
          < ImageConstructor.MOTION_THRESHOLD):
        continue

      p1_idx, p1_dir = self._find_position(-1, 
                                           i, 
                                           self.image_range/2)
      p2_idx, p2_dir = self._find_position(1, 
                                           i,
                                           self.image_range/2)

      if p1_idx == -1 or p2_idx == -1 or p1_dir * p2_dir != -1:
        continue

      img_gpr_m = GprMeasurements(1)
      img_gpr_m.measurements = gpr_m.measurements[:,p1_idx:p2_idx+1]
      img_gpr_m.times = gpr_m.times[p1_idx:p2_idx+1]
      img_gpr_m.ranges = gpr_m.ranges[p1_idx:p2_idx+1]
      img_gpr_m.directions = gpr_m.directions[p1_idx:p2_idx+1]

      img_gpr = MetricGprImage(self.im_props,
                              img_gpr_m,
                              [total_station_range-self.image_range/2,
                               total_station_range+self.image_range/2],
                              self.image_size)
      
      time_stamp_elem = str(total_station_time).split('.')
      yaw = int(np.degrees(ts_m.directions[i]) + 180)

      filename = (f"{time_stamp_elem[0]}_{time_stamp_elem[1]}"
                  f"_X_{np.round(total_station_measurement[0],4)}" 
                  f"_Y_{np.round(total_station_measurement[1],4)}_T"
                  f"_yaw_{np.round(yaw,1)}_{self.index}")

      img_gpr.write(self.output_path, filename, total_station_measurement, 
                    total_station_time, ts_m.directions[i], flip=self.save_flipped_image)

      prev_position = total_station_measurement