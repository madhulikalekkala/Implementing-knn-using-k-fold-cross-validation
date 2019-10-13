import csv
import random
import math
import operator
import numpy as np
from random import seed
from random import randrange

#Loading the dataset
def LoadingData(file, split,method, trainingSet=[] ,testSet=[],dataset1=[]):
	with open(file) as infile, open('csv.data', 'w') as outfile:
		for line in infile:
			outfile.write(" ".join(line.split()).replace(' ', ','))
			outfile.write("\n")
	with open('csv.data', 'rt') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		if (file=='wdbc.data'):
			for z in dataset:
				z[1], z[31] = z[31], z[1]
		for x in range(len(dataset)-1):
			if (file=='iris.data'):
				for y in range(4):
					m=y
					dataset[x][m] = float(dataset[x][m])
			if (file=='yeast.data'):
				for y in range(8):
					m=y
					dataset[x][m+1] = float(dataset[x][m+1])
			if (file=='wdbc.data'):
				for y in range(30):
					m=y
					dataset[x][m+1] = float(dataset[x][m+1])
			if(method=="randomSplit"):
				if random.random() < split:
					trainingSet.append(dataset[x])
				else:
					testSet.append(dataset[x])
			if(method=="kfold"):
				dataset1.append(dataset[x])

#defining the k-fold cross validation
def cross_validation_split(dataset, folds=3):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split



#defining the Euclidean distance
def euclideanDistance(data,instance1, instance2, length):
	distance = 0
	if (data=='iris.data'):
		for x in range(length):
			distance += pow((instance1[x] - instance2[x]), 2)
	if (data=='yeast.data' or data=='wdbc.data'):
		for x in range(length-1):
			y=x
			distance += pow((instance1[y+1] - instance2[y+1]), 2)
	return math.sqrt(distance)

#defining the polynomial distance
def polynomialKernel(data,instance1, instance2, length):
	distance = 0
	xx=0.0
	yy=0.0
	xy=0.0
	if (data=='iris.data'):
		for x in range(length):
			xy = xy + instance1[x]*instance2[x]
			yy = yy + instance2[x]*instance2[x]
			xx = xx + instance1[x]*instance1[x]
	if (data=='yeast.data' or data=='wdbc.data'):
		for x in range(length-1):
			y=x
			xy = xy + instance1[y+1]*instance2[y+1]
			yy = yy + instance2[y+1]*instance2[y+1]
			xx = xx + instance1[y+1]*instance1[y+1]
	distance=pow(1 +math.sqrt(xx -2*xy + yy),3);
	return distance
                
#defining the radial distance
def radialDistance(data,instance1, instance2, length):
	distance = 0
	xMy=0.0
	sigma=0.97
	if (data=='iris.data'):
		for x in range(length):
			xMy = xMy + abs(instance1[x] - instance2[x])
	if (data=='yeast.data' or data=='wdbc.data'):
		for x in range(length-1):
			y=x
			xMy = xMy + abs(instance1[y+1] - instance2[y+1])
	distance=2-2*math.exp(-(math.pow(xMy,2)/math.pow(sigma,2))) 
	return distance

#defining the sigmoid distance
def sigmoidDistance(data,instance1, instance2, length):
	distance = 0
	xx=0.0
	yy=0.0
	xy=0.0
	if (data=='iris.data'):
		for x in range(length):
			xy = xy + instance1[x]*instance2[x]
			yy = yy + instance2[x]*instance2[x]
			xx = xx + instance1[x]*instance1[x]
	if (data=='yeast.data' or data=='wdbc.data'):
		for x in range(length-1):
			y=x
			xy = xy + instance1[y+1]*instance2[y+1]
			yy = yy + instance2[y+1]*instance2[y+1]
			xx = xx + instance1[y+1]*instance1[y+1]
	distance= np.tanh(0.3*math.sqrt(xx -2*xy + yy)+0.7)
	return distance


def getNeighbors(distance_method,data,trainingSet, testInstance, k):
	distances=[]
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		if(distance_method=="euclidean"):
			dist = euclideanDistance(data,testInstance, trainingSet[x], length)
		if(distance_method=="polynomial"):
			dist = polynomialKernel(data,testInstance, trainingSet[x], length)
		if(distance_method=="radialDistance"):
			dist = radialDistance(data,testInstance, trainingSet[x], length)
		if(distance_method=="sigmoidDistance"):
			dist = sigmoidDistance(data,testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
		return neighbors
    

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	dataset1=[]
	avg_accuracy=0
	seed(1)
	method = input("Method")
	distance_method=input("Distance Method")
	data = input("Enter the dataset name: ")
	k = input("Enter the value of k : ")
	split = 0.67
	if(method=="randomSplit"):
		LoadingData(data, split,method,trainingSet, testSet,dataset1)
		print('Train set: ' + repr(len(trainingSet)))
		print('Test set: ' + repr(len(testSet)))
		predictions=[]
		k=int(k)
		for x in range(len(testSet)):
			neighbors = getNeighbors(distance_method,data,trainingSet, testSet[x], k)
			result = getResponse(neighbors)
			predictions.append(result)
			print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
		accuracy = getAccuracy(testSet, predictions)
		print('Accuracy: ' + repr(accuracy) + '%')
	if(method=="kfold"):
		LoadingData(data, split,method,trainingSet, testSet,dataset1)
		dataset_split=cross_validation_split(dataset1,10)
		for x in range(10):
			for y in range(int(len(dataset1)/10)):
				testSet.append(dataset_split[x][y])
			for z in range(10):
				if(z!=x):
					for m in range(int(len(dataset1)/10)):
						trainingSet.append(dataset_split[z][m])
			predictions=[]
			k=int(k)
			for x in range(len(testSet)):
				neighbors = getNeighbors(distance_method,data,trainingSet, testSet[x], k)
				result = getResponse(neighbors)
				predictions.append(result)
				print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
			accuracy = getAccuracy(testSet, predictions)
			avg_accuracy= avg_accuracy+accuracy
			print('Accuracy: ' + repr(accuracy) + '%')                 
			trainingSet=[]
			testSet=[]
		print('Avg_Accuracy'+repr(avg_accuracy/10)+'%')
main()

