from libcpp.string cimport string
from libcpp.vector cimport vector
import matplotlib.pyplot as plt
import math
import numpy as np
import operator

cdef int plot_count

cdef public void initialize_plot(int count):
	global plot_count
	plot_count = count

cdef public void plot(int plotNo, const vector[float]& x, const vector[float]& y, const string& plotLabel):
	global plot_count
	colormap = plt.cm.cool
	plt.plot(x, y, label=plotLabel, color=colormap(float(plotNo)/float(plot_count)))

cdef public void plot_image(const vector[float]& x, const string& file_name):
	width = math.sqrt(len(x))
	img = np.reshape(x, (width, width))
	plt.imshow(img, cmap=plt.cm.Greys)
	plt.savefig(file_name, bbox_inches='tight')

cdef public void save_plot(float x1, float x2, float y1, float y2, float xtick, float ytick, const string& xAxis, const string& yAxis, const string& fileName):
	plt.xlim([x1, x2]);
	plt.ylim([y1, y2]);
	plt.xticks(np.arange(x1, x2, xtick))
	plt.yticks(np.arange(y1, y2, ytick))
	plt.title(xAxis + '/' + yAxis + ' plot')
	plt.xlabel(xAxis)
	plt.ylabel(yAxis)
	plt.grid()	
	ax = plt.axes()
	handles, labels = ax.get_legend_handles_labels()
	plt.legend(fancybox=True, shadow=True)
	hl=sorted(zip(handles, labels),
	key=operator.itemgetter(1))
	handles2, labels2 = zip(*hl)
	ax.legend(handles2, labels2)
	figure = plt.gcf()
	figure.set_size_inches(30, 30)
	plt.savefig(fileName, bbox_inches='tight')
