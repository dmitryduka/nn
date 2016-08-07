from libcpp.string cimport string
from libcpp.vector cimport vector
import matplotlib.pyplot as plt
import numpy as np

cdef public void plot(const vector[float]& x, const vector[float]& y, float x1, float x2, float y1, float y2, const string& plotLabel):
	plt.plot(x, y, label=plotLabel)
	plt.xlim([x1, x2]);
	plt.ylim([y1, y2]);

cdef public void save_plot(const string& xAxis, const string& yAxis, const string& fileName):
	plt.title(xAxis + '/' + yAxis + ' plot')
	plt.xlabel(xAxis)
	plt.ylabel(yAxis)
	plt.legend()
	figure = plt.gcf()
	figure.set_size_inches(20, 20)
	plt.savefig(fileName)
