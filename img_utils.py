"""Utility functions for process and visualize images"""
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


def show_img(img):
  print img.shape, img.dtype
  
  plt.imshow(img[:,:,0])
  plt.ion()
  plt.show()
  raw_input()


def heatmap2d(hm_mat, title='', block=True, fig_num=1):
  """
  Display heatmap
  input:
    hm_mat:   mxn 2d np array
  """
  print 'map shape: {}, data type: {}'.format(hm_mat.shape, hm_mat.dtype)

  if block:
    plt.figure(fig_num)
    plt.clf()
  
  plt.imshow(hm_mat, cmap='hot', interpolation='nearest')
  plt.title(title)
  plt.colorbar()
  for y in range(hm_mat.shape[0]):
    for x in range(hm_mat.shape[1]):
      plt.text(x, y, '%.1f' % hm_mat[y, x],
               horizontalalignment='center',
               verticalalignment='center',
               )
  if block:
    plt.ion()
    print 'press any key to continue'
    plt.show()
    raw_input()


def heatmap3d(hm_mat, title=''):
  """
  Credit:
    https://stackoverflow.com/questions/14061061/how-can-i-render-3d-histograms-in-python
  """

  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.pyplot as plt
  import numpy as np
  print 'map shape: {}, data type: {}'.format(hm_mat.shape, hm_mat.dtype)

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
  plt.title(title, fontsize=20)
  # _, ax = plt.subplots()
  ax = fig.add_subplot(111, )
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
  plt.ion()
  print 'press any key to continue'
  plt.show()
  raw_input()