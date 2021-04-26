import pywt
import numpy as np
import matplotlib.pyplot as plt
import logging
from skimage.restoration import denoise_wavelet, estimate_sigma
from scipy import signal
from scipy.signal import butter, firwin, lfilter, filtfilt, hilbert
from scipy.ndimage.filters import uniform_filter1d

def bw_filter(acc_x, acc_y, acc_z, ang_x, ang_y, ang_z):
  """Third order butterworth bandpass filter for IMU data.

  params:
    acc_x: linear acceleration in x-direction [m/s^2].
    acc_y: linear acceleration in y-direction [m/s^2].
    acc_z: linear acceleration in z-direction [m/s^2].
    ang_x: angular velocity about x-axis [rad/s].
    ang_y: angular velocity about y-axis [rad/s].
    ang_z: angular velocity about z-axis [rad/s].

  """
  b, a = signal.butter(3, 0.002)
  acc_x2 = signal.filtfilt(b,a,acc_x)
  acc_y2 = signal.filtfilt(b,a,acc_y)
  acc_z2 = signal.filtfilt(b,a,acc_z)
  ang_x2 = signal.filtfilt(b,a,ang_x)
  ang_y2 = signal.filtfilt(b,a,ang_y)
  ang_z2 = signal.filtfilt(b,a,ang_z)
  return acc_x2, acc_y2, acc_z2, ang_x2, ang_y2, ang_z2

def dewow(config, gpr_img):
    """Polynomial dewow filter.

    params:
      config: Dewow configuration parameters.
      trace_im: np.2darray of horizontally stacked traces.
    """
  
    first_trace = gpr_img[:,0]
    model = np.polyfit(np.arange(gpr_img.shape[0]), first_trace, config.degree)
    pred = np.polyval(model, gpr_img.shape[0])

    return gpr_img + pred

def discrete_wavelet_transform(trace_1d, threshold=.45):
  """Performs discrete wavelet transforms on 1D waveform.

  params:
    trace_1d: list containing amplitudes of GPR signal.
    threshold: float in [0,1] provided to the wavelet filter.
      More information is available in the pywavelets documentation.
      https://pywavelets.readthedocs.io/en/latest/ref/idwt-inverse-discrete-wavelet-transform.html
  """

  wavelet = pywt.Wavelet('db2')
  coeffs = pywt.wavedec(trace_1d, wavelet, level=3)

  for i in range(len(coeffs)):
    if i == 0: 
      continue
    K = np.round(threshold * len(coeffs[i])).astype(int)
    coeffs[i][K:] = np.zeros(len(coeffs[i]) - K)

  return pywt.waverec(coeffs, wavelet)


def bgr(gpr_img, window=0, verbose=False):
    """Horizontal background removal filter.
    
    params:
      config: AttrDict structure containing window parameters.
      trace_im: np.2darray of horizontally stacked traces. 
    """
    if window == 0:
      return gpr_img

    elif window == -1:
      return gpr_img - np.average(gpr_img, axis=1)[:, np.newaxis]

    else:
      if window < 10:
        logging.warning(f'BGR window of size {window} is short.')

      if (window / 2.0 == int(window / 2)):
        window = window + 1
      gpr_img -= uniform_filter1d(gpr_img, 
                                  size=window, 
                                  mode='constant', 
                                  cval=0.0, 
                                  axis=1)

      return gpr_img

def triangular(config, gpr_img):
    """Triangle FIR bandpass filter.
    
    params:
      config: AttrDict structure containing relevant system freq parameters.
      gpr_img: np.2darray of horizontally stacked traces. 
    """

    filt = firwin(numtaps=int(config.num_taps), 
                  cutoff=[int(config.min_freq), int(config.max_freq)], 
                  window='triangle', 
                  pass_zero='bandpass', 
                  fs=int(config.sampling_freq))

    proc_trace = np.copy(lfilter(filt, 1.0, gpr_img, axis=0))
    proc_trace = lfilter(filt, 1.0, proc_trace[::-1], axis=0)[::-1]

    return proc_trace

def sec_gain(gpr_img, a=0.02, b=1, threshold=90):
  """Spreading and Exponential Compensation (SEC) gain function.

  params:
    gpr_img: np.2darray of horizontally stacked traces. 
    a: Power gain component.
    b: Linear gain component.
    threshold: Cut-off array element where gain is flattened.
  """
  t = np.arange(gpr_img.shape[0])
  gain_fn = t**b * np.exp(t*a)
  gain_fn[threshold:] = gain_fn[threshold]

  return np.multiply(gain_fn[:, np.newaxis], gpr_img)

  




