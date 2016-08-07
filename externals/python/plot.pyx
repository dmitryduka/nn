from libcpp.string cimport string
from libcpp.vector cimport vector
import matplotlib.pyplot as plt
import numpy as np
import operator

cdef public void initialize_plot(int plot_count):
	colormap = plt.cm.cool
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, plot_count)])

cdef public void plot(const vector[float]& x, const vector[float]& y, const string& plotLabel):
	plt.plot(x, y, label=plotLabel)

cdef public void save_plot(float x1, float x2, float y1, float y2, const string& xAxis, const string& yAxis, const string& fileName):
	plt.xlim([x1, x2]);
	plt.ylim([y1, y2]);
	plt.xticks(np.arange(x1, x2, 1))
	plt.yticks(np.arange(y1, y2, 0.01))
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
	plt.savefig(fileName)
