import os
import csv
import math
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

number_of_points = 30607
train_size = int(math.ceil(0.8*number_of_points))
test_size = int(math.floor(0.2*number_of_points))

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
        if not node.euclideanChecked:
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
    else:
        worst_neighbor = return_worst_neighbor(kNeighbors)
        if abs(query[cd] - node.value[cd]) <= worst_neighbor[0] or len(kNeighbors) < k:
            if not node.leftExplored:
                        
                node.leftExplored = True
                Search(query, depth+1, dim, k, node.left, kNeighbors)
            elif not node.rightExplored:
                        
                node.rightExplored = True
                Search(query, depth+1, dim, k, node.right, kNeighbors)
    return kNeighbors


def brute_search(query, depth, dim, k, node, kNeighbors):
    if node == None:
        return kNeighbors
    cd = depth % dim
    if len(kNeighbors) < k:
        kNeighbors.append((euclideanDistance(query, node.value), node.value, node.label))
        node.euclideanChecked = True
    else:
        if not node.euclideanChecked:
            worst_neighbor = return_worst_neighbor(kNeighbors)
            if euclideanDistance(query, node.value) < worst_neighbor[0]:
                kNeighbors.remove(worst_neighbor)
                kNeighbors.append((euclideanDistance(query, node.value), node.value, node.label))
            node.euclideanChecked = True
    
    if (not node.leftExplored) and (not node.rightExplored):
        node.leftExplored = True
        Search(query, depth+1, dim, k, node.left, kNeighbors)
        node.rightExplored = True
        Search(query, depth+1, dim, k, node.right, kNeighbors)
    else:
        if not node.leftExplored:
            node.leftExplored = True
            Search(query, depth+1, dim, k, node.left, kNeighbors)
        elif not node.rightExplored:
            node.rightExplored = True
            Search(query, depth+1, dim, k, node.right, kNeighbors)
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
        if not node.euclideanChecked:
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
    else:
        worst_neighbor = return_worst_neighbor(kNeighbors)
        if abs(query[cd] - node.value[cd]) <= worst_neighbor[0] or len(kNeighbors) < k:
            if not node.leftExplored:
                node.leftExplored = True
                # nodeCount += 1
                kNeighbors, nodeCount = modifiedSearch(query, depth+1, dim, k, node.left, kNeighbors, nodeCount, c, ln)
            elif not node.rightExplored:
                node.rightExplored = True
                # nodeCount += 1
                kNeighbors, nodeCount = modifiedSearch(query, depth+1, dim, k, node.right, kNeighbors, nodeCount, c, ln)
    return kNeighbors, nodeCount

######################### IMPLEMENTATIONS #######################################
data, labels, labels_count = prepareCaltech256("/home/hamza/university_admissions/UCSC/Research/I2I_Search/CLIP_features_caltech256")
original_x_train, original_x_test, original_y_train, original_y_test = train_test_split(data, labels, test_size= 0.2)
x_train = np.copy(original_x_train[0:train_size])
y_train = np.copy(original_y_train[0:train_size])
new_labels_count = {}
for i in range(len(y_train)):
    try:
        new_labels_count[y_train[i]] += 1
    except:
        new_labels_count[y_train[i]] = 1
x_test = np.copy(original_x_test[0:test_size])
y_test = np.copy(original_y_test[0:test_size])

KD_x_train = np.copy(x_train)
KD_y_train = np.copy(y_train)
KD_x_test = np.copy(x_test)
KD_y_test = np.copy(y_test)

KNN_x_train = np.copy(x_train)
KNN_y_train = np.copy(y_train)
KNN_x_test = np.copy(x_test)
KNN_y_test = np.copy(y_test)


def new_KDTree(k, x_train, y_train, x_test, y_test):
    root_node = Node()
    root_node = Setup(x_train, y_train,0, len(data[0]), root_node)
    KD_precision = 0; KD_recall = 0; KD_f1score = 0
    for i in range(len(y_test)):
        round_precision = 0; round_recall = 0; round_f1score = 0
        kNeighbors = []; predicted_labels = []; correct_label = y_test[i]
        true_positive = 0; false_positive = 0; false_negative = 0; true_negative = 0
        total_correct_labels = 0
        
        try:
            total_correct_labels = new_labels_count[correct_label]
        except:
            pass
        if total_correct_labels == 0:
            false_positive = k
            true_negative = len(x_train) - false_positive
        else:
            kNeighbors = Search(x_test[i], 0, len(data[0]), k,root_node, kNeighbors)
            clearExplored(root_node)
            for j in range(len(kNeighbors)):
                if kNeighbors[j][2] == correct_label:
                    true_positive += 1
                else:
                    false_positive += 1
            false_negative = total_correct_labels - true_positive
            # true_negative = len(x_train) - true_positive - false_positive - false_negative
        
        round_precision = true_positive/(true_positive + false_positive)
        try:
            round_recall  = true_positive/(true_positive+false_negative)
        except:
            pass
        try:
            round_f1score = (2*round_precision*round_recall)/(round_precision+round_recall)
        except:
            pass
        KD_precision += round_precision
        KD_recall += round_recall
        KD_f1score += round_f1score
    KD_precision /= len(y_test)
    KD_recall /= len(y_test)
    KD_f1score /= len(y_test)

    print("KDTree precision: ", KD_precision)
    print("KDTree recall: ", KD_recall)
    # print("KDTree f1score: ", KD_f1score)

def new_KNN(k, x_train, y_train, x_test, y_test):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(x_train)
    KNN_precision = 0; KNN_recall = 0; KNN_f1score = 0
    number_test_points = len(y_test)
    for i in range(number_test_points):
        round_precision = 0; round_recall = 0; round_f1score = 0
        true_positive = 0; false_positive = 0; false_negative = 0; true_negative = 0
        correct_label = y_test[i]
        total_correct_labels = 0
        try:
            total_correct_labels = new_labels_count[correct_label]
        except:
            pass
        if total_correct_labels == 0:
            false_positive = k
            true_negative = len(x_train) - false_positive
        else:
            distances, indices = neigh.kneighbors(x_test[i].reshape(1,-1), return_distance=True)
            # print(indices)
            # break
            distances = distances[0]
            indices = indices[0]
            # predicted_labels = []
            for j in range(len(indices)):
                predicted_label = y_train[indices[j]]
                # predicted_labels.append((predicted_label,distances[j]))
                if predicted_label == correct_label:
                    true_positive += 1
                else:
                    false_positive += 1
            false_negative = total_correct_labels - true_positive
            # true_negative = len(x_train) - true_positive - false_positive - false_negative
            # print(predicted_labels)
            # print("Correct Label: ",correct_label)
            # print("Total Correct Labels: ", total_correct_labels)
            # print("True Positive: ", true_positive)
            # print("False Positive: ", false_positive)
            # print("False Negative", false_negative)
        round_precision = true_positive/(true_positive + false_positive)
        try:
            round_recall  = true_positive/(true_positive+false_negative)
        except:
            pass
        try:
            round_f1score = (2*round_precision*round_recall)/(round_precision+round_recall)
        except:
            pass
        KNN_precision += round_precision
        KNN_recall += round_recall
        KNN_f1score += round_f1score
    KNN_precision /= number_test_points
    KNN_recall /= number_test_points
    KNN_f1score /= number_test_points

    print("KNN precision: ", KNN_precision)
    print("KNN recall: ", KNN_recall)
    # print("KNN f1score: ", KNN_f1score)


def new_modified_KDTree(k,c, x_train, y_train, x_test, y_test):
    root_node = Node()
    root_node = Setup(x_train, y_train,0, len(data[0]), root_node)
    KD_precision = 0; KD_recall = 0; KD_f1score = 0
    number_test_points = len(y_test)
    for i in range(number_test_points):
        round_precision = 0; round_recall = 0; round_f1score = 0
        kNeighbors = []; predicted_labels = []; correct_label = y_test[i]
        true_positive = 0; false_positive = 0; false_negative = 0; true_negative = 0
        total_correct_labels = 0
        
        try:
            total_correct_labels = new_labels_count[correct_label]
        except:
            pass
        if total_correct_labels == 0:
            false_positive = k
            true_negative = len(x_train) - false_positive
        else:
            nodeCount = 0
            kNeighbors, nodeCount = modifiedSearch(x_test[i], 0, len(data[0]), k,root_node, kNeighbors, nodeCount, c, math.ceil(math.log2(len(x_train))))
            clearExplored(root_node)
            
            for j in range(len(kNeighbors)):
                # predicted_labels.append((kNeighbors[j][2], kNeighbors[j][0]))
                if kNeighbors[j][2] == correct_label:
                    true_positive += 1
                else:
                    false_positive += 1
            false_negative = total_correct_labels - true_positive
            # true_negative = len(x_train) - true_positive - false_positive - false_negative
            # predicted_labels.sort(key = lambda x: x[1])
            # print(predicted_labels)
            # print("Correct Label: ",correct_label)
            # print("Total Correct Labels: ", total_correct_labels)
            # print("True Positive: ", true_positive)
            # print("False Positive: ", false_positive)
            # print("False Negative", false_negative)
            # print("Max Node Count Allowed: ", c*k*math.ceil(math.log2(len(x_train))))
            # print("Node Count: ", nodeCount)

        round_precision = true_positive/(true_positive + false_positive)
        try:
            round_recall  = true_positive/(true_positive+false_negative)
        except:
            pass
        try:
            round_f1score = (2*round_precision*round_recall)/(round_precision+round_recall)
        except:
            pass
        KD_precision += round_precision
        KD_recall += round_recall
        KD_f1score += round_f1score
    KD_precision /= number_test_points
    KD_recall /= number_test_points
    KD_f1score /= number_test_points

    print(f"Modified KDTree precision for c = {c}: ", KD_precision)
    print(f"Modified KDTree recall for c = {c}: ", KD_recall)
    # print(f"Modified KDTree f1score for c = {c}: ", KD_f1score)

############################### TESTING #######################################
k = 1

start_time = time.time()
new_KNN(k, np.copy(x_train), np.copy(y_train), np.copy(x_test), np.copy(y_test))
end_time = time.time()
print(f"KNN execution time: {end_time-start_time}")
print("\n")

start_time = time.time()
new_KDTree(k, np.copy(x_train), np.copy(y_train), np.copy(x_test), np.copy(y_test))
end_time = time.time()
print(f"KD Tree execution time: {end_time-start_time}")
print("\n")

start_time = time.time()
new_modified_KDTree(k, 1, np.copy(x_train), np.copy(y_train), np.copy(x_test), np.copy(y_test))
end_time = time.time()
print(f"c = 1 execution time: {end_time-start_time}")
print("\n")

start_time = time.time()
new_modified_KDTree(k, 2, np.copy(x_train), np.copy(y_train), np.copy(x_test), np.copy(y_test))
end_time = time.time()
print(f"c = 2 execution time: {end_time-start_time}")
print("\n")

start_time = time.time()
new_modified_KDTree(k, 4, np.copy(x_train), np.copy(y_train), np.copy(x_test), np.copy(y_test))
end_time = time.time()
print(f"c = 4 execution time: {end_time-start_time}")
print("\n")

start_time = time.time()
new_modified_KDTree(k, 16, np.copy(x_train), np.copy(y_train), np.copy(x_test), np.copy(y_test))
end_time = time.time()
print(f"c = 16 execution time: {end_time-start_time}")
print("\n")

start_time = time.time()
new_modified_KDTree(k, 64, np.copy(x_train), np.copy(y_train), np.copy(x_test), np.copy(y_test))
end_time = time.time()
print(f"c = 64 execution time: {end_time-start_time}")


# print("\n")
# new_KNN(k, np.copy(x_train), np.copy(y_train), np.copy(x_test), np.copy(y_test))
# print("\n")
# new_modified_KDTree(k, c, np.copy(x_train), np.copy(y_train), np.copy(x_test), np.copy(y_test))
# new_KDTree(10)
# print("\n")
# new_KNN(10)
# temp_x_train = np.copy(x_train)
# x_train[0][0] = 0.5
# print(temp_x_train == x_train)

# print(math.log2(205))

# print(KNN_x_train == x_train)