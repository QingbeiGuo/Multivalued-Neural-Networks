#pytorch -0.2.1
#python -3.6.2
#torchvision - 0.1.9

import torch
from torch.autograd import Variable
from torchvision import models
#import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def get_kmeans(x, n_clusters, c_range = []):
    X = x.reshape(len(x),1)
    print("X", X)

    # Incorrect number of clusters
    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(X)
    l_pred_class = estimator.labels_
    centroids = estimator.cluster_centers_
    print("centroids", centroids)
    print("min(centroids)", min(centroids))
    print("centroids.index(min(centroids))", centroids.tolist().index(min(centroids)))
    
    l_pred_centroid = np.zeros(l_pred_class.shape)
    for i in range(len(l_pred_class)):
        if l_pred_class[i] in c_range:
            l_pred_centroid[i] = centroids[l_pred_class[i]]
        else:
            l_pred_centroid[i] = X[i]

    l_pred_class = l_pred_class.reshape(x.shape)
    l_pred_centroid = l_pred_centroid.reshape(x.shape)

    print("label_pred_class", l_pred_class)
    print("label_pred_centroid", l_pred_centroid)

    return centroids, l_pred_class, l_pred_centroid

class ModifiedVGG7Model(torch.nn.Module):
	def __init__(self):
		super(ModifiedVGG7Model, self).__init__()

		self.features = nn.Sequential(
		    nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-04, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
		    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
		    nn.MaxPool2d(kernel_size=2),                   
            nn.BatchNorm2d(128, eps=1e-04, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
		    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-04, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
		    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
		    nn.MaxPool2d(kernel_size=2),                   
            nn.BatchNorm2d(256, eps=1e-04, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-04, momentum=0.1, affine=False),
		    nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
		    nn.MaxPool2d(kernel_size=2),              
            nn.BatchNorm2d(512, eps=1e-04, momentum=0.1, affine=False),
            nn.ReLU(inplace=True))

		for param in self.features.parameters():
			  param.requires_grad = True

		self.classifier = nn.Sequential(
		    nn.Linear(4*4*512, 1024),
            nn.BatchNorm1d(1024, eps=1e-04, momentum=0.1, affine=False),
		    nn.ReLU(inplace=True),
		    nn.Linear(1024, 10))

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)

		return x

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--train_path", type = str, default = "D:/Datasets/MNIST/")
    parser.add_argument("--test_path", type = str, default = "D:/Datasets/MNIST/")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = get_args()
	print("args:", args)

	model = torch.load("model2000").cuda()
	#model = torch.load("model_compressing").cuda()
	print("model:", model)

	n_clusters = 2
	c_range = [0, 1]

	for l, (module) in enumerate(model.modules()):
	    if isinstance(module, nn.Conv2d):
	        #print(module.weight)

	        weight_array = []
	        weight_array.extend(module.weight.data.cpu().numpy().flatten())

	        #print("weight_array:", weight_array)
	        print("weight_array:", set(list(weight_array)))
	        print("===============================================================")

	        plt.hist(weight_array, 100)
	        plt.xlabel('Value of weight')
	        plt.ylabel('Scale(log(count))')
	        plt.title('Weight distribution')
	        plt.show()

	        #centroids, label_pred_class, label_pred_centroid = get_kmeans(x = np.array(weight_array), n_clusters = n_clusters, c_range = c_range)

	        #plt.plot(1)
	        #plt.scatter(weight_array, np.zeros(len(weight_array)), c=label_pred_class)
	        #plt.title("Incorrect Number of Blobs")
	        #plt.show()
