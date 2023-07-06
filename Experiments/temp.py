import os
import csv
import math
from re import A
import time
from pstats import Stats
import numpy as np
import statistics
from queue import PriorityQueue
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KDTree
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import NearestNeighbors

path = "/home/hamza/university_admissions/UCSC/Research/I2I_Search/CLIP_features_caltech256"

kpca = KernelPCA(n_components=64, kernel='cosine')

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.parent = None
        self.leftExplored = False
        self.rightExplored = False
        self.euclideanChecked = False
        self.value = []
        self.label = None

########################## HELPER FUNCTIONS ###########################

def euclideanDistance(x, y):
    return np.linalg.norm(x - y)

def return_worst_neighbor(kNeighbors):
    max_dist = -1
    worst_neighbor = None
    for i in range(len(kNeighbors)):
        if kNeighbors[i][0] > max_dist:
            max_dist = kNeighbors[i][0]
            worst_neighbor = kNeighbors[i]
    return worst_neighbor

def sortData(cd, dataset, labels):
    n = len(dataset)
    for i in range(n-1):
        for j in range(i+1,n):
            if dataset[i][cd] > dataset[j][cd]:
                temp = np.array(dataset[i])
                dataset[i] = np.array(dataset[j])
                dataset[j] = np.array(temp)
                labels[i], labels[j] = labels[j], labels[i]     
    return dataset, labels

def clearExplored(node):
    if node == None:
        return
    node.leftExplored = False
    node.rightExplored = False
    node.euclideanChecked = False
    clearExplored(node.left)
    clearExplored(node.right)

def prepareCaltech256(datapath):
    file_list = os.listdir(datapath)
    file_list.sort()
    labels = []
    labels_count = {}
    caltech256_clip_features = []
    for i in range(len(file_list)):
        curr_path = datapath + "/" + file_list[i]
        curr_features = np.load(curr_path)
        labels_count[i] = 0
        for j in range(len(curr_features)):
            labels.append(i)
            labels_count[i] += 1
        caltech256_clip_features.extend(curr_features)
    caltech256_clip_features = np.asarray(caltech256_clip_features)
    return caltech256_clip_features, labels, labels_count

########################################################################

########################### SETUP ######################################

def Setup(dataset, labels, depth, dim, node):
    cd = depth % dim
    dataset, labels = sortData(cd, dataset, labels)
    median_index = math.ceil(len(dataset)/2) -1
    median_value = dataset[median_index]
    median_label = labels[median_index]
    node.value = median_value
    node.label = median_label
    left_value_array = np.array([]); right_value_array = np.array([]); left_label_array = np.array([]); right_label_array = np.array([])
    left_value_array = dataset[0:median_index]
    left_label_array = labels[0:median_index]
    right_value_array = dataset[median_index+1:]
    right_label_array = labels[median_index+1:]
    
    if len(left_value_array):
        node.left = Node()
        node.left.parent = node
        node.left = Setup(left_value_array, left_label_array, depth+1, dim, node.left)
        
    if len(right_value_array):
        node.right = Node()
        node.right.parent = node
        node.right = Setup(right_value_array, right_label_array, depth+1, dim, node.right)
       
    return node

########################### SEARCHES ###################################

def Search(query, depth, dim, k, node, kNeighbors):
    if node == None:
        return kNeighbors
    cd = depth % dim
    if len(kNeighbors) < k:
        kNeighbors.append((euclideanDistance(query, node.value), node.value, node.label))
        node.euclideanChecked = True
    else:
        worst_neighbor = return_worst_neighbor(kNeighbors)
        if euclideanDistance(query, node.value) < worst_neighbor[0]:
            kNeighbors.remove(worst_neighbor)
            kNeighbors.append((euclideanDistance(query, node.value), node.value, node.label))
        node.euclideanChecked = True
    
    if (not node.leftExplored) and (not node.rightExplored):
        if query[cd] < node.value[cd]:
        
            node.leftExplored = True
            Search(query, depth+1, dim, k, node.left, kNeighbors)
            worst_neighbor = return_worst_neighbor(kNeighbors)
            if abs(query[cd] - node.value[cd]) <= worst_neighbor[0] or len(kNeighbors) < k:
                node.rightExplored = True
                Search(query, depth+1, dim, k, node.right, kNeighbors)
        else:
            node.rightExplored = True
            Search(query, depth+1, dim, k, node.right, kNeighbors)
            worst_neighbor = return_worst_neighbor(kNeighbors)
            if abs(query[cd] - node.value[cd]) <= worst_neighbor[0] or len(kNeighbors) < k:
                node.leftExplored = True
                Search(query, depth+1, dim, k, node.left, kNeighbors)
    
    return kNeighbors

def modifiedSearch(query, depth, dim, k, node, kNeighbors, nodeCount, c, ln):
    if nodeCount >= c*k*ln:
        return kNeighbors, nodeCount-1
    if node == None:
        return kNeighbors, nodeCount-1
    cd = depth % dim
    if len(kNeighbors) < k:
        kNeighbors.append((euclideanDistance(query, node.value), node.value, node.label))
        node.euclideanChecked = True
    else:
        worst_neighbor = return_worst_neighbor(kNeighbors)
        if euclideanDistance(query, node.value) < worst_neighbor[0]:
            kNeighbors.remove(worst_neighbor)
            kNeighbors.append((euclideanDistance(query, node.value), node.value, node.label))
        node.euclideanChecked = True
    
    if (not node.leftExplored) and (not node.rightExplored):
        if query[cd] < node.value[cd]:
            node.leftExplored = True
            # nodeCount += 1
            kNeighbors, nodeCount =  modifiedSearch(query, depth+1, dim, k, node.left, kNeighbors, nodeCount+1, c, ln)
            worst_neighbor = return_worst_neighbor(kNeighbors)
            if abs(query[cd] - node.value[cd]) <= worst_neighbor[0] or len(kNeighbors) < k:
                node.rightExplored = True
                # nodeCount += 1
                kNeighbors, nodeCount = modifiedSearch(query, depth+1, dim, k, node.right, kNeighbors, nodeCount+1, c, ln)
        else:
            node.rightExplored = True
            # nodeCount += 1
            kNeighbors, nodeCount = modifiedSearch(query, depth+1, dim, k, node.right, kNeighbors, nodeCount+1, c, ln)
            worst_neighbor = return_worst_neighbor(kNeighbors)
            if abs(query[cd] - node.value[cd]) <= worst_neighbor[0] or len(kNeighbors) < k:
                node.leftExplored = True
                # nodeCount += 1
                kNeighbors, nodeCount = modifiedSearch(query, depth+1, dim, k, node.left, kNeighbors, nodeCount+1, c, ln)
    
    return kNeighbors, nodeCount

def new_KDTree():
    train_data = np.load("/home/hamza/university_admissions/UCSC/Research/I2I_Search/KD-Trees/Experiments/x_train_caltech256_256.npy")
    train_labels = np.load("/home/hamza/university_admissions/UCSC/Research/I2I_Search/KD-Trees/Experiments/y_train_caltech256_256.npy")
    test_data = np.load("/home/hamza/university_admissions/UCSC/Research/I2I_Search/KD-Trees/Experiments/x_test_caltech256_256.npy")
    test_labels = np.load("/home/hamza/university_admissions/UCSC/Research/I2I_Search/KD-Trees/Experiments/y_test_caltech256_256.npy")

    new_labels_count = {}
    for i in range(len(train_labels)):
        try:
            new_labels_count[train_labels[i]] += 1
        except:
            new_labels_count[train_labels[i]] = 1

    # KD_precision = 0; KD_recall = 0
    new_labels_count_sorted = sorted(new_labels_count.items(), key=lambda x:x[1])
    
    root_node = Node()
    root_node = Setup(train_data, train_labels, 0, len(train_data[0]), root_node)
    # start_time = time.time()

    kNeighbors = []; nodeCount = 0; c = 1; ln =  math.ceil(math.log2(len(train_data)))
    
    # kNeighbors, nodeCount = modifiedSearch(test_data[0], 0, len(train_data[0]), 5, root_node, kNeighbors, nodeCount, c, ln)
    kNeighbors = Search(test_data[0], 0, len(train_data[0]), 5, root_node, kNeighbors)
    print(new_labels_count_sorted)
    # print(nodeCount)
    # for i in range(len(kNeighbors)):
    #     print(kNeighbors[i][2], end = " ")
    # print()

# new_KDTree()

def newTempFunc():
    labels = list(np.load("/home/hamza/university_admissions/UCSC/Research/I2I_Search/caltech256_labels.npy"))
    print(labels.count(250))
    labels_count = {}
    for i in range(len(labels)):
        try:
            labels_count[labels[i]] += 1
        except:
            labels_count[labels[i]] = 1
    
    sorted_labels_count = sorted(labels_count.items(), key = lambda x:x[1])
    print(sorted_labels_count)


newTempFunc()

    



