import os

import tensorflow as tf
import numpy as np
import scipy.misc
from io import BytesIO, StringIO
tf.compat.v1.disable_eager_execution()#禁用 Eager Execution
import matplotlib.pyplot as plt
# from scipy.interpolate import spline, interp1d

from core.util import time_seq

plt.rcParams.update({'font.size': 13})
plt.rcParams['figure.figsize'] = 10, 8


class TensorBoardLogger(object):
	"""
    Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    """

	def __init__(self, log_dir):
		"""Create a summary writer logging to log_dir."""
		self.writer = tf.compat.v1.summary.FileWriter(log_dir)

	def scalar_summary(self, tag, step, value):
		"""Log a scalar variable."""
		summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
		self.writer.add_summary(summary, step)

	def image_summary(self, tag, images, step):
		"""Log a list of images."""

		img_summaries = []
		for i, img in enumerate(images):
			# Write the image to a string
			try:
				s = StringIO()
			except:
				s = BytesIO()
			scipy.misc.toimage(img).save(s, format="png")

			# Create an Image object
			img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
			                           height=img.shape[0],
			                           width=img.shape[1])
			# Create a Summary value
			img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

		# Create and write Summary
		summary = tf.Summary(value=img_summaries)
		self.writer.add_summary(summary, step)

	def histo_summary(self, tag, values, step, bins=1000):
		"""Log a histogram of the tensor of values."""

		# Create a histogram using numpy
		counts, bin_edges = np.histogram(values, bins=bins)

		# Fill the fields of the histogram proto
		hist = tf.HistogramProto()
		hist.min = float(np.min(values))
		hist.max = float(np.max(values))
		hist.num = int(np.prod(values.shape))
		hist.sum = float(np.sum(values))
		hist.sum_squares = float(np.sum(values ** 2))

		# Drop the start of the first bin
		bin_edges = bin_edges[1:]

		# Add bin edges and counts
		for edge in bin_edges:
			hist.bucket_limit.append(edge)
		for c in counts:
			hist.bucket.append(c)

		# Create and write Summary
		summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
		self.writer.add_summary(summary, step)
		self.writer.flush()


class Plot:
	def __init__(self, save_path):
		self.Y = []
		self.X = []
		self.ax = None
		self.fig = None
		self.save_path = save_path

	def save(self):
		# list and ',' = list[0]
		line, = self.ax.plot(self.X, self.Y, 'b')
		if self.ax.get_title() != '':
			name = self.ax.get_title().replace(' ', '_')
			self.fig.savefig(self.save_path + name + '.png')
		else:
			name = time_seq()
			self.fig.savefig(self.save_path + name + '.png')


class MatplotlibLogger:

	def __init__(self, save_path):
		self.save_path = save_path
		self.plot_dict = {}

		if self.save_path[-1] != '/':
			self.save_path += '/'

	def add_plot(self, tag: str, x_label, y_label, title=''):
		plot = Plot(self.save_path)
		plot.fig, plot.ax = plt.subplots()
		plot.ax.set_xlabel(x_label)
		plot.ax.set_ylabel(y_label)
		plot.ax.set_title(title)

		self.plot_dict[tag] = plot

	def scalar_summary(self, tag, x, y):
		self.plot_dict[tag].Y.append(y)
		self.plot_dict[tag].X.append(x)
		self.plot_dict[tag].save()







