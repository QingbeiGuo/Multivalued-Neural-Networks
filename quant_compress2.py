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
import dataset_mnist
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
#from numpy import arange

class CentroidLoss(nn.Module):
    def __init__(self, model, n_partitions, i_partitions, centroids, lambda_l1, lambda_l2, lambda_g, lambda_c):
        super(CentroidLoss, self).__init__()
        self.model = model
        self.n_partitions = n_partitions
        self.i_partitions = i_partitions
        self.centroids = centroids
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_g = lambda_g
        self.lambda_c = lambda_c

    def forward(self, input, target):
        #print("input", input)
        #print("target", target)

        loss = F.cross_entropy(input, target)
        #print("loss", loss)

        #L1 regularization
        l1_regularizer = 0
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                #print("weight", module.weight)
                l1_regularizer = l1_regularizer + torch.sum(torch.abs(module.weight))
                #print("l1_regularizer", l1_regularizer)

        '''#L2 regularization
        l2_regularizer = 0
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                #print("weight", module.weight)
                l2_regularizer = l2_regularizer + torch.sum(module.weight * module.weight)
                #print("l2_regularizer", l2_regularizer)'''
        
        '''#Group LASSO regularization1
        lasso_regularizer1 = 0
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                print("weight", module.weight)
                print("weight.size", module.weight.size())
                print("weight.dim", module.weight.dim())
                print("weight.size(0)", module.weight.size(0))

                for i in range(module.weight.size(0)):
                    lasso_regularizer1 = lasso_regularizer1 + torch.sqrt(torch.sum(module.weight[i, :, :, :] * module.weight[i, :, :, :]))
                #print("lasso_regularizer1", lasso_regularizer1)'''
        
        '''#Group Lasso regularization2
        lasso_regularizer2 = 0
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                print("weight", module.weight)
                print("weight.size", module.weight.size())
                print("weight.dim", module.weight.dim())
                print("weight.size(0)", module.weight.size(0))

                for i in range(module.weight.size(1)):
                    lasso_regularizer2 = lasso_regularizer2 + torch.sqrt(torch.sum(module.weight[:, i, :, :] * module.weight[:, i, :, :]))
                #print("lasso_regularizer2", lasso_regularizer2)'''

        '''#Centroid loss regularization3
        centroid_regularizer = 0
        i_conv_layer = 0
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                weight_view = module.weight.view(-1,1)
                #print("weight_view", weight_view)

                n_partition = self.n_partitions[i_conv_layer]

                pred_centroid = np.zeros((len(weight_view), 1))
                for i in range(n_partition):
                    pred_centroid[self.i_partitions[i_conv_layer][i]] = self.centroids[i_conv_layer][i]
                #print("pred_centroid1", pred_centroid)

                pred_centroid = Variable(torch.from_numpy(pred_centroid).type(torch.FloatTensor))
                if not pred_centroid.is_cuda:
                    pred_centroid = pred_centroid.cuda()
                #print("pred_centroid2", pred_centroid)

                centroid_regularizer = centroid_regularizer + torch.sum((weight_view - pred_centroid) * (weight_view - pred_centroid))
                #print("centroid_regularizer", centroid_regularizer)

                i_conv_layer = i_conv_layer + 1'''

        return loss + self.lambda_l1 * l1_regularizer

class ModifiedLeNetModel(torch.nn.Module):
	def __init__(self):
		super(ModifiedLeNetModel, self).__init__()

		self.features = nn.Sequential(
		    nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32, eps=1e-04, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
		    nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64, eps=1e-04, momentum=0.1, affine=False),
		    nn.ReLU(inplace=True),
		    nn.MaxPool2d(kernel_size=2))                       

		for param in self.features.parameters():
			  param.requires_grad = True

        #parameters initilize

		self.classifier = nn.Sequential(
		    nn.Linear(5*5*64, 512),
            nn.BatchNorm2d(512, eps=1e-04, momentum=0.1, affine=False),
		    nn.ReLU(inplace=True),                      
		    nn.Linear(512, 10))                       

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)

		return x

class TrainingFineTuner_LeNet:
	def __init__(self, train_path, test_path, model):
		self.train_data_loader = dataset_mnist.train_loader(train_path)
		self.test_data_loader = dataset_mnist.test_loader(test_path)

		self.n_partitions = [2, 2];
		self.c_w = [];          #[1,2,0,2,1,1,...] save class
		self.r_partitions = []; #[[1,5],[6,10],[11,13],[14,20]] save edge
		self.w_partitions = []; #[[...],[...],[...],[...]] save weights
		self.i_partitions = []; #[[...],[...],[...],[...]] save index
		self.centroids = [];    #[1,2,3,4] save mean before
		self.centroids_ = [];   #[1,2,3,4] save mean after
		self.contractioncount  = 300; #the number of partition contraction batch_count * epoch_count
		#self.delta = 0.0000001;#the change delta of edge 0.0000001 (without closed in 200 epochs)
		#self.delta = 0.000001; #the change delta of edge 0.000001
		#self.delta = 0.00001;  #the change delta of edge 0.00001
		#self.delta = 0.0001;   #the change delta of edge 0.0001
		#self.delta = 0.001;    #the change delta of edge 0.001
		self.tau   = 1;         #the learning rate of centroids
		self.inittype = 2;
		self.lastepoch = 0;
		self.laststep = 0;
		self.startquantepoch = 0;
		self.contractionepoch  = 300;
		self.learningrate = 0.0001;
		#self.decreasing_lr = [15, 25]

		self.model = model
		#self.criterion = CentroidLoss(self.model, self.n_partitions, self.i_partitions, self.centroids, 0.0001, 0, 0, 0)             #lambda_l1, lambda_l2, lambda_g, lambda_c
		self.criterion = torch.nn.CrossEntropyLoss()
		self.model.train()

		self.accuracys1 = []
		self.accuracys5 = []

	def train(self, optimizer = None, epoches = -1, batches = -1):
        #resume
		if os.path.isfile("epoches_compressing") and os.path.isfile("model_compressing"):
		    list_epoches = torch.load("epoches_compressing")
		    self.model = torch.load("model_compressing")
		    print("model_compressing resume:", self.model)

		    self.accuracys1 = torch.load("accuracys1_compressing")
		    self.accuracys5 = torch.load("accuracys5_compressing")
		    print("accuracys1_compressing resume:", self.accuracys1)
		    print("accuracys5_compressing resume:", self.accuracys5)
		else:
		    list_epoches = list(range(epoches))

		list_ = list_epoches[:]
		#for i in range(epoches):
		for i in list_epoches[:]:
		    print("Epoch: ", i)

		    if i == self.startquantepoch:
		        fine_tuner.init_partitions()

		    #if i in self.decreasing_lr:
		    #    self.learningrate = self.learningrate/10;

		    if optimizer is None:
		        optimizer = \
                    optim.SGD(model.parameters(), lr=self.learningrate, momentum=0.9, weight_decay=0.0001)
                    #optim.SGD(model.classifier.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.9)
                    #optim.SGD(model.features.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.9)

		    if (i >= self.startquantepoch) & (i <= self.startquantepoch + self.contractionepoch):
		        self.quantization_partitions(epoch = i)

		    self.train_epoch(i, batches, optimizer)

		    if (i >= self.startquantepoch) & (i <= self.startquantepoch + self.contractionepoch):
		        self.update_partitions()
		    
		    self.test()

		    list_.remove(i)                                                           #update list_epoches

            #save
		    torch.save(list_, "epoches_compressing")
		    torch.save(self.model, "model_compressing")
		    torch.save(self.accuracys1, "accuracys1_compressing")
		    torch.save(self.accuracys5, "accuracys5_compressing")

		print("Finished.")

	def train_epoch(self, epoch, batches, optimizer = None):
		for step, (batch, label) in enumerate(self.train_data_loader):
			  if (step == batches):
			      break

			  print("Epoch-step: ", epoch, "-", step)

			  self.train_batch(epoch, step, batches, optimizer, batch.cuda(), label.cuda())
			  #self.train_batch(optimizer, batch, label)

		if epoch >= self.startquantepoch:
		    self.judge_partitions(epoch, step)

	def train_batch(self, epoch, step, batches, optimizer, batch, label):
        #Forward
		self.model.zero_grad()
		input = Variable(batch)                                                       #Tensor->Variable

        #output = self.model(input)
		#loss = self.criterion(output, Variable(label))
		#loss.backward()
		self.criterion(self.model(input), Variable(label)).backward()                 #1. output=self.model(input) 2.loss=self.criterion(output, Variable(label)) 3.loss.backward()
		
		if epoch >= self.startquantepoch:
		    self.grad_partitions()

		optimizer.step()                                                              #update parameters

        #Backward
        #None

        #update

	def test(self):
		self.model.eval()

		#correct = 0
		correct1 = 0
		correct5 = 0
		total = 0

		print("Testing...")
		for i, (batch, label) in enumerate(self.test_data_loader):
			  batch = batch.cuda()
			  output = model(Variable(batch))
			  pred = output.data.max(1)[1]
			  #correct += pred.cpu().eq(label).sum()
			  cor1, cor5 = accuracy(output.data, label, topk=(1, 5))                    # measure accuracy top1 and top5
			  correct1 += cor1
			  correct5 += cor5
			  total += label.size(0)

		self.accuracys1.append(float(correct1.numpy()) / total)
		self.accuracys5.append(float(correct5.numpy()) / total)

		print("Accuracy Top1:", float(correct1.numpy()) / total)
		print("Accuracy Top5:", float(correct5.numpy()) / total)

		self.model.train()

	def init_partitions(self): #initialize the weight partition for each layer(uniform, kmeans and log-uniform)
		i_conv_layer = 0
        #for the conv_layer weights
		for module in model.modules():
		    if isinstance(module, nn.Conv2d):
		        weight_array = []
		        weight_array.extend(module.weight.data.cpu().numpy().flatten())
		        w = np.array(weight_array)

		        n_partition = self.n_partitions[i_conv_layer]

##################################################################################################
		        if self.inittype == 0:		            
                    #isometry/np.linspace
		            splitline = np.linspace(min(w), max(w), n_partition+1)
		            if len(self.r_partitions) < i_conv_layer+1:
		                self.r_partitions.append([])
		            for i in range(n_partition):
		                self.r_partitions[i_conv_layer].append([splitline[i], splitline[i+1]])

		            i_conv_layer = i_conv_layer + 1

		            #print("w:", w)
		            #print("r_partitions", self.r_partitions)
		        elif self.inittype == 1:		            
                    #equivalence/np.linspace
		            splitline = np.linspace(0, len(w)-1, n_partition+1)
		            if len(self.r_partitions) < i_conv_layer+1:
		                self.r_partitions.append([])
		            for i in range(n_partition):
		                if int(splitline[i+1]) == splitline[i+1]:
		                    self.r_partitions[i_conv_layer].append([w[math.ceil(splitline[i])], w[int(splitline[i+1])-1]])
		                else:
		                    self.r_partitions[i_conv_layer].append([w[math.ceil(splitline[i])], w[int(splitline[i+1])]])
		            i_conv_layer = i_conv_layer + 1

		            #print("w:", w)
		            #print("r_partitions", self.r_partitions)
		        elif self.inittype == 2:		            
                    #clustering/kmeans
		            w_ = w.reshape(len(w), 1)                #[[1] [2] [3] [4] [5]]
		            estimator = KMeans(n_clusters=n_partition)
		            estimator.fit(w_)
		            pred_class = estimator.labels_

		            if len(self.r_partitions) < i_conv_layer+1:
		                self.r_partitions.append([])
		            for i in range(n_partition):
		                idx = np.where(pred_class == i)
		                print("idx:", idx)
		                print("w[idx]:", w[idx])
		                self.r_partitions[i_conv_layer].append([min(w[idx]), max(w[idx])])

		            self.r_partitions[i_conv_layer] = sorted(self.r_partitions[i_conv_layer])

		            for i in range(n_partition):
		                if i == 0:
		                    self.r_partitions[i_conv_layer][i][1] = (self.r_partitions[i_conv_layer][i][1] + self.r_partitions[i_conv_layer][i+1][0]) /2
		                elif i == n_partition - 1:
		                    self.r_partitions[i_conv_layer][i][0] = (self.r_partitions[i_conv_layer][i-1][1] + self.r_partitions[i_conv_layer][i][0]) /2
		                else:
		                    self.r_partitions[i_conv_layer][i][0] = (self.r_partitions[i_conv_layer][i-1][1] + self.r_partitions[i_conv_layer][i][0]) /2
		                    self.r_partitions[i_conv_layer][i][1] = (self.r_partitions[i_conv_layer][i][1] + self.r_partitions[i_conv_layer][i+1][0]) /2

		            i_conv_layer = i_conv_layer + 1

		            #print("w:", w)
		            #print("r_partitions", self.r_partitions)
		        elif self.inittype == 3:		            
                    #log-uniform/np.logspace
		            splitline = np.logspace(min(w), max(w), n_partition+1)
		            if len(self.r_partitions) < i_conv_layer+1:
		                self.r_partitions.append([])
		            for i in range(n_partition):
		                self.r_partitions[i_conv_layer].append([splitline[i], splitline[i+1]])

		            i_conv_layer = i_conv_layer + 1

		            #print("w:", w)
		            #print("r_partitions", self.r_partitions)
##################################################################################################

	def quantization_partitions(self, epoch): #the quantization of weights
		i_conv_layer = 0
        #for the conv_layer weights
		for module in model.modules():
		    if isinstance(module, nn.Conv2d):
		        weight_array = []
		        weight_array.extend(module.weight.data.cpu().numpy().flatten())
		        w = np.array(weight_array)

		        n_partition = self.n_partitions[i_conv_layer]

		        if len(self.c_w) < i_conv_layer+1:
		            self.c_w.append(-1*np.ones(len(w)))
		        else:
		            self.c_w[i_conv_layer] = -1*np.ones(len(w))
		        if len(self.w_partitions) < i_conv_layer+1:
		            self.w_partitions.append([])
		        else:
		            self.w_partitions[i_conv_layer] = []
		        if len(self.i_partitions) < i_conv_layer+1:
		            self.i_partitions.append([])
		        else:
		            self.i_partitions[i_conv_layer] = []
		        if len(self.centroids) < i_conv_layer+1:
		            self.centroids.append([])
		        else:
		            self.centroids[i_conv_layer] = []

                #  [ )    [ )    [ )   [ ]
                #  ) [ )    [ )    [ )   (
		        for i in range(n_partition):
		            if len(self.w_partitions[i_conv_layer]) < i+1:
		                self.w_partitions[i_conv_layer].append([])
		            if len(self.i_partitions[i_conv_layer]) < i+1:
		                self.i_partitions[i_conv_layer].append([])

		            if self.r_partitions[i_conv_layer][i][0] != self.r_partitions[i_conv_layer][i][1]:
		                if i < n_partition-1:
		                    idx = np.where((w >= self.r_partitions[i_conv_layer][i][0]) & (w < self.r_partitions[i_conv_layer][i][1]))
		                else:
		                    idx = np.where((w >= self.r_partitions[i_conv_layer][i][0]) & (w <= self.r_partitions[i_conv_layer][i][1]))
		                if len(idx[0]) > 0:
		                    self.c_w[i_conv_layer][idx] = i
		                    self.w_partitions[i_conv_layer][i] = w[idx].tolist()
		                    self.i_partitions[i_conv_layer][i] = idx[0].tolist()

		                if i == 0:
		                    idx = np.where(w < self.r_partitions[i_conv_layer][i][0])
		                    if len(idx[0]) > 0:
		                        self.c_w[i_conv_layer][idx] = i
		                        #self.w_partitions[i_conv_layer][i] = self.w_partitions[i_conv_layer][i] + (np.ones(len(idx[0]))*self.centroids_[i_conv_layer][i]).tolist()
		                        self.w_partitions[i_conv_layer][i] = self.w_partitions[i_conv_layer][i] + (np.ones(len(idx[0]))*self.r_partitions[i_conv_layer][i][0]).tolist()
		                        self.i_partitions[i_conv_layer][i] = list(set(self.i_partitions[i_conv_layer][i] + idx[0].tolist()))
		                        #w[idx] = self.centroids_[i_conv_layer][i]
		                        w[idx] = self.r_partitions[i_conv_layer][i][0]

		                elif i == n_partition-1:
		                    idx = np.where(w > self.r_partitions[i_conv_layer][i][1])
		                    if len(idx[0]) > 0:
		                        self.c_w[i_conv_layer][idx] = i
		                        #self.w_partitions[i_conv_layer][i] = self.w_partitions[i_conv_layer][i] + (np.ones(len(idx[0]))*self.centroids_[i_conv_layer][i]).tolist()
		                        self.w_partitions[i_conv_layer][i] = self.w_partitions[i_conv_layer][i] + (np.ones(len(idx[0]))*self.r_partitions[i_conv_layer][i][1]).tolist()
		                        self.i_partitions[i_conv_layer][i] = list(set(self.i_partitions[i_conv_layer][i] + idx[0].tolist()))
		                        #w[idx] = self.centroids_[i_conv_layer][i]
		                        w[idx] = self.r_partitions[i_conv_layer][i][1]

		                if i > 0:
		                    idx = np.where((w >= self.r_partitions[i_conv_layer][i-1][1]) & (w < self.r_partitions[i_conv_layer][i][0]) & (np.abs(w - self.r_partitions[i_conv_layer][i-1][1]) > np.abs(w - self.r_partitions[i_conv_layer][i][0])))
		                    if len(idx[0]) > 0:
		                        self.c_w[i_conv_layer][idx] = i
		                        #self.w_partitions[i_conv_layer][i] = self.w_partitions[i_conv_layer][i] + (np.ones(len(idx[0]))*self.centroids_[i_conv_layer][i]).tolist()
		                        self.w_partitions[i_conv_layer][i] = self.w_partitions[i_conv_layer][i] + (np.ones(len(idx[0]))*self.r_partitions[i_conv_layer][i][0]).tolist()
		                        self.i_partitions[i_conv_layer][i] = list(set(self.i_partitions[i_conv_layer][i] + idx[0].tolist()))
		                        #w[idx] = self.centroids_[i_conv_layer][i]
		                        w[idx] = self.r_partitions[i_conv_layer][i][0]

		                if i < n_partition-1:
		                    idx = np.where((w >= self.r_partitions[i_conv_layer][i][1]) & (w < self.r_partitions[i_conv_layer][i+1][0]) & (np.abs(w - self.r_partitions[i_conv_layer][i][1]) <= np.abs(w - self.r_partitions[i_conv_layer][i+1][0])))
		                    if len(idx[0]) > 0:
		                        self.c_w[i_conv_layer][idx] = i
		                        #self.w_partitions[i_conv_layer][i] = self.w_partitions[i_conv_layer][i] + (np.ones(len(idx[0]))*self.centroids_[i_conv_layer][i]).tolist()
		                        self.w_partitions[i_conv_layer][i] = self.w_partitions[i_conv_layer][i] + (np.ones(len(idx[0]))*self.r_partitions[i_conv_layer][i][1]).tolist()
		                        self.i_partitions[i_conv_layer][i] = list(set(self.i_partitions[i_conv_layer][i] + idx[0].tolist()))
		                        #w[idx] = self.centroids_[i_conv_layer][i]
		                        w[idx] = self.r_partitions[i_conv_layer][i][1]

		                self.centroids[i_conv_layer].append(np.mean(self.w_partitions[i_conv_layer][i]))
		            else:
		                idx = np.where(w == self.r_partitions[i_conv_layer][i][0])
		                if len(idx[0]) > 0:
		                    self.c_w[i_conv_layer][idx] = i
		                    self.w_partitions[i_conv_layer][i] = w[idx].tolist()
		                    self.i_partitions[i_conv_layer][i] = idx[0].tolist()

		                if i == 0:
		                    idx = np.where(w < self.r_partitions[i_conv_layer][i][0])
		                    if len(idx[0]) > 0:
		                        self.c_w[i_conv_layer][idx] = i
		                        #self.w_partitions[i_conv_layer][i] = self.w_partitions[i_conv_layer][i] + (np.ones(len(idx[0]))*self.centroids_[i_conv_layer][i]).tolist()
		                        self.w_partitions[i_conv_layer][i] = self.w_partitions[i_conv_layer][i] + (np.ones(len(idx[0]))*self.r_partitions[i_conv_layer][i][0]).tolist()
		                        self.i_partitions[i_conv_layer][i] = list(set(self.i_partitions[i_conv_layer][i] + idx[0].tolist()))
		                        #w[idx] = self.centroids_[i_conv_layer][i]
		                        w[idx] = self.r_partitions[i_conv_layer][i][0]

		                elif i == n_partition-1:
		                    idx = np.where(w > self.r_partitions[i_conv_layer][i][1])
		                    if len(idx[0]) > 0:
		                        self.c_w[i_conv_layer][idx] = i
		                        #self.w_partitions[i_conv_layer][i] = self.w_partitions[i_conv_layer][i] + (np.ones(len(idx[0]))*self.centroids_[i_conv_layer][i]).tolist()
		                        self.w_partitions[i_conv_layer][i] = self.w_partitions[i_conv_layer][i] + (np.ones(len(idx[0]))*self.r_partitions[i_conv_layer][i][1]).tolist()
		                        self.i_partitions[i_conv_layer][i] = list(set(self.i_partitions[i_conv_layer][i] + idx[0].tolist()))
		                        #w[idx] = self.centroids_[i_conv_layer][i]
		                        w[idx] = self.r_partitions[i_conv_layer][i][1]

		                if i > 0:
		                    idx = np.where((w > self.r_partitions[i_conv_layer][i-1][1]) & (w < self.r_partitions[i_conv_layer][i][0]) & (np.abs(w - self.r_partitions[i_conv_layer][i-1][1]) > np.abs(w - self.r_partitions[i_conv_layer][i][0])))
		                    if len(idx[0]) > 0:
		                        self.c_w[i_conv_layer][idx] = i
		                        #self.w_partitions[i_conv_layer][i] = self.w_partitions[i_conv_layer][i] + (np.ones(len(idx[0]))*self.centroids_[i_conv_layer][i]).tolist()
		                        self.w_partitions[i_conv_layer][i] = self.w_partitions[i_conv_layer][i] + (np.ones(len(idx[0]))*self.r_partitions[i_conv_layer][i][0]).tolist()
		                        self.i_partitions[i_conv_layer][i] = list(set(self.i_partitions[i_conv_layer][i] + idx[0].tolist()))
		                        #w[idx] = self.centroids_[i_conv_layer][i]
		                        w[idx] = self.r_partitions[i_conv_layer][i][0]

		                if i < n_partition-1:
		                    idx = np.where((w > self.r_partitions[i_conv_layer][i][1]) & (w < self.r_partitions[i_conv_layer][i+1][0]) & (np.abs(w - self.r_partitions[i_conv_layer][i][1]) <= np.abs(w - self.r_partitions[i_conv_layer][i+1][0])))
		                    if len(idx[0]) > 0:
		                        self.c_w[i_conv_layer][idx] = i
		                        #self.w_partitions[i_conv_layer][i] = self.w_partitions[i_conv_layer][i] + (np.ones(len(idx[0]))*self.centroids_[i_conv_layer][i]).tolist()
		                        self.w_partitions[i_conv_layer][i] = self.w_partitions[i_conv_layer][i] + (np.ones(len(idx[0]))*self.r_partitions[i_conv_layer][i][1]).tolist()
		                        self.i_partitions[i_conv_layer][i] = list(set(self.i_partitions[i_conv_layer][i] + idx[0].tolist()))
		                        #w[idx] = self.centroids_[i_conv_layer][i]
		                        w[idx] = self.r_partitions[i_conv_layer][i][1]

		                self.centroids[i_conv_layer].append(np.mean(self.w_partitions[i_conv_layer][i]))

		        #print("w1", module.weight.data)
		        module.weight.data = torch.from_numpy(w.reshape(module.weight.data.size())).cuda()
		        #print("w2", module.weight.data)

		        #print("w", w)
		        #print("c_w", self.c_w)
		        #print("w_partitions", self.w_partitions)
		        #print("i_partitions", self.i_partitions)
		        #print("centroids", self.centroids)
		        #print("r_partitions", self.r_partitions)

		        i_conv_layer = i_conv_layer + 1

	def grad_partitions(self):
		i_conv_layer = 0
		for module in model.modules():
		    if isinstance(module, nn.Conv2d):
		        grad_array = []
		        grad_array.extend(module.weight.grad.data.cpu().numpy().flatten())
		        g = np.array(grad_array)

		        n_partition = self.n_partitions[i_conv_layer]

		        g_partitions_ = []
		        for i in range(n_partition):
		            if len(g_partitions_) < i+1:
		                g_partitions_.append([])

		            if self.r_partitions[i_conv_layer][i][0] == self.r_partitions[i_conv_layer][i][1]:
		                g_partitions_[i] = g[self.i_partitions[i_conv_layer][i]]
		                if len(g_partitions_[i]) > 0:
		                    g[self.i_partitions[i_conv_layer][i]] = sum(g_partitions_[i])/len(g_partitions_[i])

		        module.weight.grad.data = torch.from_numpy(g.reshape(module.weight.grad.data.size())).cuda()

		        i_conv_layer = i_conv_layer + 1

	def update_partitions(self): #update the weights, centroids and edge
		i_conv_layer = 0
        #for the conv_layer weights
		for module in model.modules():
		    if isinstance(module, nn.Conv2d):
		        weight_array = []
		        weight_array.extend(module.weight.data.cpu().numpy().flatten())
		        w = np.array(weight_array)

		        n_partition = self.n_partitions[i_conv_layer]

		        w_partitions_ = []
		        for i in range(n_partition):
		            if len(w_partitions_) < i+1:
		                w_partitions_.append([])

		            w_partitions_[i] = w[self.i_partitions[i_conv_layer][i]]
		        #print("w_partitions_", w_partitions_)

		        if len(self.centroids_) < i_conv_layer+1:
		            self.centroids_.append([])
		        else:
		            self.centroids_[i_conv_layer] = []

		        for i in range(n_partition):
		            self.centroids_[i_conv_layer].append(np.mean(w_partitions_[i]))
		        print("new centroids_", self.centroids_[i_conv_layer])
		        print("old centroids", self.centroids[i_conv_layer])

		        centroids_delta = np.array(self.centroids_[i_conv_layer]) - np.array(self.centroids[i_conv_layer])
		        #print("centroids delat", centroids_delta)
		        #centroids = np.array(self.centroids[i_conv_layer]) + self.tau * centroids_delta
		        #print("new centroids", centroids)

		        r_partitions = []
		        for i in range(n_partition):
		            '''if self.r_partitions[i_conv_layer][i][0] + self.tau * centroids_delta[i] + self.delta/self.n_partitions[i_conv_layer] < self.r_partitions[i_conv_layer][i][1] + self.tau * centroids_delta[i] - self.delta/self.n_partitions[i_conv_layer]:
		                r_partitions.append([self.r_partitions[i_conv_layer][i][0] + self.tau * centroids_delta[i] + self.delta/self.n_partitions[i_conv_layer], self.r_partitions[i_conv_layer][i][1] + self.tau * centroids_delta[i] - self.delta/self.n_partitions[i_conv_layer]])
		            else:
		                #r_partitions.append([(self.r_partitions[i_conv_layer][i][0] + self.r_partitions[i_conv_layer][i][1])/2, (self.r_partitions[i_conv_layer][i][0] + self.r_partitions[i_conv_layer][i][1])/2])
		                r_partitions.append([centroids[i], centroids[i]])'''
		            if self.r_partitions[i_conv_layer][i][0] + self.tau * centroids_delta[i] + (self.r_partitions[i_conv_layer][i][1] - self.r_partitions[i_conv_layer][i][0])/(2*self.contractioncount) < self.r_partitions[i_conv_layer][i][1] + self.tau * centroids_delta[i] - (self.r_partitions[i_conv_layer][i][1] - self.r_partitions[i_conv_layer][i][0])/(2*self.contractioncount):
		                r_partitions.append([self.r_partitions[i_conv_layer][i][0] + self.tau * centroids_delta[i] + (self.r_partitions[i_conv_layer][i][1] - self.r_partitions[i_conv_layer][i][0])/(2*self.contractioncount), self.r_partitions[i_conv_layer][i][1] + self.tau * centroids_delta[i] - (self.r_partitions[i_conv_layer][i][1] - self.r_partitions[i_conv_layer][i][0])/(2*self.contractioncount)])
		            else:
		                #r_partitions.append([(self.r_partitions[i_conv_layer][i][0] + self.r_partitions[i_conv_layer][i][1])/2, (self.r_partitions[i_conv_layer][i][0] + self.r_partitions[i_conv_layer][i][1])/2])
		                #r_partitions.append([centroids[i], centroids[i]])
		                #r_partitions.append([(self.r_partitions[i_conv_layer][i][0] + self.tau * centroids_delta[i] + self.r_partitions[i_conv_layer][i][1] + self.tau * centroids_delta[i])/2, (self.r_partitions[i_conv_layer][i][0] + self.tau * centroids_delta[i] + self.r_partitions[i_conv_layer][i][1] + self.tau * centroids_delta[i])/2])
		                r_partitions.append([self.centroids_[i_conv_layer][i], self.centroids_[i_conv_layer][i]])
		                #r_partitions.append([self.r_partitions[i_conv_layer][i][0] + self.tau * centroids_delta[i] + (self.r_partitions[i_conv_layer][i][1] - self.r_partitions[i_conv_layer][i][0])/self.contractioncount, self.r_partitions[i_conv_layer][i][1] + self.tau * centroids_delta[i] - (self.r_partitions[i_conv_layer][i][1] - self.r_partitions[i_conv_layer][i][0])/self.contractioncount])

		        self.r_partitions[i_conv_layer] = r_partitions
		        #print("old r_partitions", self.r_partitions[i_conv_layer])
		        #print("new r_partitions", r_partitions)

		        i_conv_layer = i_conv_layer + 1

		i_conv_layer = 0
        #for the conv_layer weights
		for module in model.modules():
		    if isinstance(module, nn.Conv2d):
		        weight_array = []
		        weight_array.extend(module.weight.data.cpu().numpy().flatten())
		        w = np.array(weight_array)

		        n_partition = self.n_partitions[i_conv_layer]

		        for i in range(n_partition-1):
		            if self.r_partitions[i_conv_layer][i][1] > self.r_partitions[i_conv_layer][i+1][0]:
		                m1 = min(self.r_partitions[i_conv_layer][i][0], self.r_partitions[i_conv_layer][i][1], self.r_partitions[i_conv_layer][i+1][0], self.r_partitions[i_conv_layer][i+1][1])
		                m2 = max(self.r_partitions[i_conv_layer][i][0], self.r_partitions[i_conv_layer][i][1], self.r_partitions[i_conv_layer][i+1][0], self.r_partitions[i_conv_layer][i+1][1])
		                self.r_partitions[i_conv_layer][i][0] = m1
		                self.r_partitions[i_conv_layer][i][1] = (m1+m2)/2
		                self.r_partitions[i_conv_layer][i+1][0] = (m1+m2)/2
		                self.r_partitions[i_conv_layer][i+1][1] = m2

		        i_conv_layer = i_conv_layer + 1

		self.contractioncount = max(self.contractioncount - 1, 1)

	def judge_partitions(self, epoch, step):
		closed = True
		i_conv_layer = 0
        #for the conv_layer weights
		for module in model.modules():
		    if isinstance(module, nn.Conv2d):
		        n_partition = self.n_partitions[i_conv_layer]

		        tags = []
		        for i in range(n_partition):
		            if self.r_partitions[i_conv_layer][i][0] < self.r_partitions[i_conv_layer][i][1]:
		                tags.append(-1)
		                closed = False
		            elif self.r_partitions[i_conv_layer][i][0] > self.r_partitions[i_conv_layer][i][1]:
		                tags.append(1)
		                closed = False
		            else:
		                tags.append(0)

		        print("contractioncount", self.contractioncount)
		        print("learningrate", self.learningrate)
		        print("tags", tags)
		        print("r_partitions", self.r_partitions[i_conv_layer])
		        if self.lastepoch > 0:
		            print("closed Epoch-step: ", self.lastepoch, "-", self.laststep)

		        i_conv_layer = i_conv_layer + 1

		if (closed == True) and (self.learningrate >= 0.0001):
		    self.learningrate = self.learningrate/10
		    self.lastepoch = epoch
		    self.laststep = step

def accuracy(output, target, topk=(1,)):                                               
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)                                          
    pred = pred.t()                                                                     
    correct = pred.cpu().eq(target.view(1, -1).expand_as(pred))                               

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)                                 
        res.append(correct_k)
    return res

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--train_path", type = str, default = "/home/guoqb/work/Datasets/MNIST/")
    parser.add_argument("--test_path", type = str, default = "/home/guoqb/work/Datasets/MNIST/")
    parser.set_defaults(init=False)
    parser.set_defaults(compress=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = get_args()
	print("args:", args)

    #initiate the deep model 
	if args.init:
	    model = ModifiedLeNetModel().cuda()
	    torch.save(model, "model_")
	    #model = torch.load("model_compress").cuda()
	    print("model_:", model)

	    #for the layer weights
	    n_conv2d = 0
	    for module in model.modules():
	        if isinstance(module, nn.Conv2d):
	            n_conv2d = n_conv2d + 1
	            '''weight_array = []
	            weight_array.extend(module.weight.data.cpu().numpy().flatten())
	            print("weight_array:", weight_array)
	            print("----------------------------------------------------------------------------------------------")
	            print("----------------------------------------------------------------------------------------------")
	            print("----------------------------------------------------------------------------------------------")'''

	    print("n_conv2d:", n_conv2d)

    #compress the deep model
	elif args.compress:
		model = torch.load("model_training45").cuda()
		print("model_:", model)

		fine_tuner = TrainingFineTuner_LeNet(args.train_path, args.test_path, model)

		fine_tuner.train(epoches = 600, batches = -1)

		torch.save(model, "model_compress")
		print("model_compress:", model)

