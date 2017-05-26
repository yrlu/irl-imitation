"""Utility functions for process and visualize images"""
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def process_obs(obs):
  """
  Credits:
    https://github.com/andreimuntean/A3C/blob/master/environment.py

  Transforms the specified observation into a 47x47x1 grayscale image.
  Returns:
      A 47x47x1 tensor with float32 values between 0 and 1.
  """
  # Transform the observation into a grayscale image with values between 0 and 1. Use the simple
  # np.mean method instead of sophisticated luminance extraction techniques since they do not seem
  # to improve training.
  gray_obs = obs.mean(2)

  # Resize grayscale frame to a 47x47 matrix of 32-bit floats.
  resized_obs = misc.imresize(gray_obs, (47, 47)).astype(np.float32)
  return np.expand_dims(resized_obs, 2)


def show_img(img):
  print img.shape, img.dtype
  
  plt.imshow(img[:,:,0])
  plt.ion()
  plt.show()
  raw_input()


def heatmap2d(hm_mat, title=''):
  """
  Display heatmap
  input:
    hm_mat:   mxn 2d np array
  """
  print hm_mat.shape, hm_mat.dtype
  fig = plt.figure()
  plt.clf()
  fig.suptitle(title, fontsize=20)
  plt.imshow(hm_mat, cmap='hot', interpolation='nearest')
  # plt.ion()
  
  plt.colorbar()
  for y in range(hm_mat.shape[0]):
    for x in range(hm_mat.shape[1]):
      plt.text(x, y, '%.2f' % hm_mat[y, x],
               horizontalalignment='center',
               verticalalignment='center',
               )
  plt.show()
  # raw_input()


def heatmap3d(hm_mat, title=''):
  """
  Credit:
    https://stackoverflow.com/questions/14061061/how-can-i-render-3d-histograms-in-python
  """

  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.pyplot as plt
  import numpy as np
  #
  # Assuming you have "2D" dataset like the following that you need
  # to plot.
  #
  data_2d = hm_mat
  #
  # Convert it into an numpy array.
  #
  data_array = np.array(data_2d)
  #
  # Create a figure for plotting the data as a 3D histogram.
  #
  fig = plt.figure()
  plt.clf()
  fig.suptitle(title, fontsize=20)
  ax = fig.add_subplot(111, projection='3d')
  #
  # Create an X-Y mesh of the same dimension as the 2D data. You can
  # think of this as the floor of the plot.
  #
  x_data, y_data = np.meshgrid( np.arange(data_array.shape[1]),
                                np.arange(data_array.shape[0]) )
  #
  # Flatten out the arrays so that they may be passed to "ax.bar3d".
  # Basically, ax.bar3d expects three one-dimensional arrays:
  # x_data, y_data, z_data. The following call boils down to picking
  # one entry from each array and plotting a bar to from
  # (x_data[i], y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
  #
  x_data = x_data.flatten()
  y_data = y_data.flatten()
  z_data = data_array.flatten()
  ax.bar3d( x_data,
            y_data,
            np.zeros(len(z_data)),
            1, 1, z_data )
  #
  # Finally, display the plot.
  #
  plt.show()
  raw_input()