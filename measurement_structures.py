import numpy as np

class GprMeasurements:
  """Structure to store ground penetrating radar data.

  Attributes:
    times: np.array of timestamps (in seconds) of each trace.
    measurements: np.2darray of traces stacked horizontally.
    ranges: interpolated distance (in meters) where each trace was acquired.
  """

  def __init__(self, measurement_length=1):
    self.times = np.array([]) # in implementation, ensure .to_sec is used.
    self.measurements = np.empty((measurement_length, 0)) # Stack of GPR measurements in image form.
    self.ranges = np.array([])

  @classmethod
  def load_from_file(cls, path):    
    data = np.genfromtxt(path, delimiter=',', skip_header=1)

    gpr_meas_struct = cls()
    gpr_meas_struct.times = data[:,0].astype(float)
    gpr_meas_struct.measurements = data[:,1:].astype(float).T

    return gpr_meas_struct

  def __str__(self):
    if self.times.size == 0:
      return "No GPR measurements available."
    out = f"""{self.times.size} GPR Measurements Available."""
    return out

class OdometryMeasurements:
  """Structure to store wheel encoder data.

  Attributes:
    times: np.array of timestamps (in seconds) for each encoder tick.
    measurements: np.array of signed distances (in meters).
  """
  def __init__(self):
    self.times = np.array([])
    self.measurements = np.array([])

  @classmethod
  def load_from_file(cls, path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    
    we_meas_struct = cls()
    we_meas_struct.times = data[:,0].astype(float)
    we_meas_struct.measurements = data[:,1].astype(float)

    return we_meas_struct

  def __str__(self):
    if self.times.size == 0:
      return "No odometry measurements available."
    
    out = f"""{self.times.size} odometry measurements available. Starting at 
              {self.measurements[0]}m and ending at {self.measurements[-1]}m."""
    return out

class TotalStationMeasurements:
  """Structure to store total station data (ground truth).

  Attributes:
    times: np.array of timestamps (in seconds) for each ts measurement.
    measurements: np.2darray of observed total station positions.
  """
  def __init__(self):
    # Time of measurement acquisition.
    self.times = np.array([])

    # X, Y, Z position
    self.measurements = np.empty((3,0))

  @classmethod
  def load_from_file(cls, path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    
    ts_meas_struct = cls()

    if len(data) > 0:
      ts_meas_struct.times = data[:,0].astype(float)
      ts_meas_struct.measurements = data[:,1:].astype(float).T

    return ts_meas_struct

class ImuMeasurements:
  """Structure to store IMU (inertial measurement unit) data.

  Attributes:
    times: np.array of timestamps (in seconds) for each IMU measurement.
    measurements: np.2darray of IMU measurements stacked in the form:
        [ ax  ]
        [ ay  ]
        [ az  ]
        [ gx  ]
        [ gy  ]
        [ gz  ]
        [w_mag]
        [x_mag]
        [y_mag]
        [z_mag]
  """

  # 3 accelerometer + 3 gyroscope + 4 magnetometer = 10.
  NUM_MEASUREMENT_ELEMS = 10

  def __init__(self):
    self.times = np.array([])
    self.measurements = np.empty((ImuMeasurements.NUM_MEASUREMENT_ELEMS, 0))

  @classmethod
  def load_from_file(cls, path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    
    imu_meas_struct = cls()
    imu_meas_struct.times = data[:,0].astype(float)
    imu_meas_struct.measurements = data[:,1:].astype(float).T

    return imu_meas_struct