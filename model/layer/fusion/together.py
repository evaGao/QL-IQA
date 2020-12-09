import sys
caffe_root='/media/s408/zz/gr/NR-IQA-CNN-master1/'
sys.path.insert(0,caffe_root+'python')
import caffe
import numpy as np
import yaml
import cv2
import os.path as osp
class TogetherLayer(caffe.Layer):
	def setup(self,bottom,top):
		pass
	def reshape(self,bottom,top):
		top[0].reshape(*bottom[0].data.shape)
	def forward(self,bottom,top):
		top[0].data[...]=((bottom[0].data[...])+(bottom[1].data[...]))*0.5
	def backward(self,top,propagate_down,bottom):
		bottom[0].diff[...] = (top[0].diff[...])*0.5
		bottom[1].diff[...] = (top[0].diff[...])*0.5
