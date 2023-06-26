import os
import csv
import math
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

kpca = KernelPCA(n_components=64, kernel='cosine')
number_of_points = 5000
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

def prepareData(dataPath):
    data = []
    labels = []
    labels_dict = dict({})
    labels_set = set({})
    data_to_labels_dict = dict({})
    with open(dataPath, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            data.append(row[:-1])
            labels.append(row[-1])
    data = np.array(data, dtype= float)
    
    # Normalize the dataset
    for i in range(len(data)):
        data[i] = abs(data[i] - np.mean(data[i]))/np.std(data[i])
    
    # labels_set contains the unique set of labels
    labels_set = set(labels)
    
    # Assigning to each label a unique id from 0 till len(labels)-1
    count = 0
    for item in labels_set:
        labels_dict[item] = count
        count+= 1
    
    # replacing labels with their numeric ids for KNN
    for i in range(len(labels)):
        labels[i] = labels_dict[labels[i]]

    return data,labels,labels_dict

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
        return kNeighbors
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
            nodeCount += 1
            modifiedSearch(query, depth+1, dim, k, node.left, kNeighbors, nodeCount, c, ln)
            worst_neighbor = return_worst_neighbor(kNeighbors)
            if abs(query[cd] - node.value[cd]) <= worst_neighbor[0] or len(kNeighbors) < k:
                node.rightExplored = True
                nodeCount += 1
                modifiedSearch(query, depth+1, dim, k, node.right, kNeighbors, nodeCount, c, ln)
        else:
            node.rightExplored = True
            nodeCount += 1
            modifiedSearch(query, depth+1, dim, k, node.right, kNeighbors, nodeCount, c, ln)
            worst_neighbor = return_worst_neighbor(kNeighbors)
            if abs(query[cd] - node.value[cd]) <= worst_neighbor[0] or len(kNeighbors) < k:
                node.leftExplored = True
                nodeCount += 1
                modifiedSearch(query, depth+1, dim, k, node.left, kNeighbors, nodeCount, c, ln)
    else:
        worst_neighbor = return_worst_neighbor(kNeighbors)
        if abs(query[cd] - node.value[cd]) <= worst_neighbor[0] or len(kNeighbors) < k:
            if not node.leftExplored:
                node.leftExplored = True
                nodeCount += 1
                modifiedSearch(query, depth+1, dim, k, node.left, kNeighbors, nodeCount, c, ln)
            elif not node.rightExplored:
                node.rightExplored = True
                nodeCount += 1
                modifiedSearch(query, depth+1, dim, k, node.right, kNeighbors, nodeCount, c, ln)
    return kNeighbors

######################### IMPLEMENTATIONS #######################################
data, labels, labels_count = prepareCaltech256("/home/hamza/university_admissions/UCSC/Research/I2I_Search/CLIP_features_caltech256")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2)
x_train = x_train[0:train_size]
y_train = y_train[0:train_size]
new_labels_count = {}
for i in range(len(y_train)):
    try:
        new_labels_count[y_train[i]] += 1
    except:
        new_labels_count[y_train[i]] = 1
x_test = x_test[0:test_size]
y_test = y_test[0:test_size]

# print(x_train.shape)
# print(x_test.shape)


def KNN():
    # data, labels, labels_dict = prepareData("DryBeanDataset/Dry_Bean_Dataset.csv")
    # data, labels = prepareCaltech256("/home/hamza/university_admissions/UCSC/Research/I2I_Search/CLIP_features_caltech256")
    # x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2)
    
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    # print("KNN Accuracy Score: ", accuracy_score(y_test, y_pred))
    print("KNN Precision Score: ", precision_score(y_test, y_pred, average = 'micro'))
    # print("KNN recall score: ", recall_score(y_test, y_pred, average='micro'))
    # print("KNN f1 score: ", f1_score(y_test, y_pred, average='micro'))

def _KDTree():
    # data, labels, labels_dict = prepareData("DryBeanDataset/Dry_Bean_Dataset.csv")
    # data, labels = prepareCaltech256("/home/hamza/university_admissions/UCSC/Research/I2I_Search/CLIP_features_caltech256")
    # x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2)
    root_node = Node()
    root_node = Setup(x_train, y_train,0, len(data[0]), root_node)
    y_pred = []
    count = 0
    for i in range(len(y_test)):
        kNeighbors = []; predicted_labels = []
        # kNeighbors = modifiedSearch(x_test[i], 0, len(data[0]), 10,root_node, kNeighbors, 0, 4, math.log2(len(x_train)))
        kNeighbors = Search(x_test[i], 0, len(data[0]), 10,root_node, kNeighbors)

        clearExplored(root_node)
        for j in range(len(kNeighbors)):
            predicted_labels.append(kNeighbors[j][2])
        y_pred.append(statistics.mode(predicted_labels))
        if  y_pred[i] == y_test[i]:
            count += 1
    # print("KDTree accuracy: ", accuracy_score(y_test, y_pred))
    print("KDTree precision: ", precision_score(y_test, y_pred, average = 'micro'))
    # print("KDTree recall: ", recall_score(y_test, y_pred, average='micro'))
    # print("KDTree f1score: ", f1_score(y_test, y_pred, average='micro'))

def scikit_KDTree():
    # data, labels, labels_dict = prepareData("DryBeanDataset/Dry_Bean_Dataset.csv")
    # data, labels = prepareCaltech256("/home/hamza/university_admissions/UCSC/Research/I2I_Search/CLIP_features_caltech256")
    # x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2)
    tree = KDTree(x_train, leaf_size=2)
    dist, ind = tree.query(x_test, k=10, return_distance=True, dualtree=False, breadth_first=False)
    y_pred = []
    for i in range(len(ind)):
        temp_labels = []
        for j in range(len(ind[i])):
            temp_labels.append(y_train[ind[i][j]])
        y_pred.append(statistics.mode(temp_labels))
    # print("scikit_KDTree accuracy: ", accuracy_score(y_test, y_pred))
    print("scikit_KDTree precision: ", precision_score(y_test, y_pred, average = 'micro'))
    # print("scikit_KDTree recall: ", recall_score(y_test, y_pred, average='micro'))  
    # print("scikit_KDTree f1score: ", f1_score(y_test, y_pred, average='micro'))

def _modifiedKDTree(c):
    # data, labels, labels_dict = prepareData("DryBeanDataset/Dry_Bean_Dataset.csv")
    # data, labels = prepareCaltech256("/home/hamza/university_admissions/UCSC/Research/I2I_Search/CLIP_features_caltech256")
    # x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2)
    root_node = Node()
    root_node = Setup(x_train, y_train,0, len(data[0]), root_node)
    y_pred = []
    count = 0
    for i in range(len(y_test)):
        kNeighbors = []; predicted_labels = []
        kNeighbors = modifiedSearch(x_test[i], 0, len(data[0]), 10,root_node, kNeighbors, 0, c, math.log2(len(x_train)))
        # kNeighbors = modifiedSearch(x_test[i], 0, len(data[0]), 10,root_node, kNeighbors)

        clearExplored(root_node)
        for j in range(len(kNeighbors)):
            predicted_labels.append(kNeighbors[j][2])
        y_pred.append(statistics.mode(predicted_labels))
        if  y_pred[i] == y_test[i]:
            count += 1
    # print(f"Modified KDTree accuracy for c = {c}: ", accuracy_score(y_test, y_pred))
    print(f"Modified KDTree precision for c = {c}: ", precision_score(y_test, y_pred, average = 'micro'))
    # print(f"Modified KDTree recall for c = {c}: ", recall_score(y_test, y_pred, average='micro'))
    # print(f"Modified KDTree f1score for c = {c}: ", f1_score(y_test, y_pred, average='micro'))

def neighbors_KDTree():
    data, labels, labels_dict = prepareData("DryBeanDataset/Dry_Bean_Dataset.csv")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2)
    root_node = Node()
    root_node = Setup(x_train, y_train,0, len(data[0]), root_node)
    y_pred = []
    count = 0
    kNeighbors_list = []
    for i in range(len(y_test)):
        kNeighbors = []; predicted_labels = []
        kNeighbors = Search(x_test[i], 0, len(data[0]), 10,root_node, kNeighbors)
        clearExplored(root_node)
        kNeighbors_list.append(kNeighbors)
    all_neighbors = []
    for i in range(len(kNeighbors_list)):
        ten_neighbors = []
        for j in range(len(kNeighbors_list[i])):
            ten_neighbors.append(kNeighbors_list[i][j][1])
        all_neighbors.append(ten_neighbors)
    return all_neighbors

def neighbors_scikitKDTree():
    data, labels, labels_dict = prepareData("DryBeanDataset/Dry_Bean_Dataset.csv")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2)
    tree = KDTree(x_train, leaf_size=2)
    dist, ind = tree.query(x_test, k=10, return_distance=True, dualtree=False, breadth_first=False)
    x_pred = []
    for i in range(len(ind)):
        temp_neighbors = []
        for j in range(len(ind[i])):
            temp_neighbors.append(x_train[ind[i][j]])
        x_pred.append(temp_neighbors)
    return x_pred

def new_KDTree(k):
    root_node = Node()
    root_node = Setup(x_train, y_train,0, len(data[0]), root_node)
    precision = 0; recall = 0; f1score = 0
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
        precision += round_precision
        recall += round_recall
        f1score += round_f1score
    precision /= len(y_test)
    recall /= len(y_test)
    f1score /= len(y_test)

    print("KDTree precision: ", precision)
    print("KDTree recall: ", recall)
    print("KDTree f1score: ", f1score)

def new_KNN(k):
    neigh = NearestNeighbors(n_neighbors=k)
    knn_x_train = np.copy(x_train)
    neigh.fit(knn_x_train)
    precision = 0; recall = 0; f1score = 0
    for i in range(len(y_test)):
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
            indices = indices[0]
            for j in range(len(indices)):
                predicted_label = y_train[indices[j]]
                if predicted_label == correct_label:
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
        precision += round_precision
        recall += round_recall
        f1score += round_f1score
    precision /= len(y_test)
    recall /= len(y_test)
    f1score /= len(y_test)

    print("KNN precision: ", precision)
    print("KNN recall: ", recall)
    print("KNN f1score: ", f1score)

        

############################### TESTING #######################################

# KNN()
# print("")
# scikit_KDTree()
# print("")
# _KDTree()
# print("")
# _modifiedKDTree(1)
# print("")
# _modifiedKDTree(2)
# print("")
# _modifiedKDTree(4)
# print("")
# _modifiedKDTree(16)
# print("")
# _modifiedKDTree(64)
# print(_KDTree())


# new_KDTree(10)
# print("\n")
# new_KDTree(10)
# print("\n")
# new_KNN(10)
# print("\n")
# new_KDTree(10)
# print("\n")
# new_KNN(10)
# print("\n")
# new_KDTree(10)

# print(x_test.dtype)

def compare():
    kd_neighbors = neighbors_KDTree()
    scikit_kd_neighbors = neighbors_scikitKDTree()
    count = 0
    for i in range(len(kd_neighbors)):
        temp_count = 0
        for j in range(len(kd_neighbors[i])):
            comparison = kd_neighbors[i][j] == scikit_kd_neighbors[i][j]
            equal_arrays = comparison.all()
            if equal_arrays:
                temp_count += 1
        if temp_count == len(kd_neighbors[i]):
            count += 1
    return count/len(kd_neighbors)

# data, labels, labels_dict = prepareData("DryBeanDataset/Dry_Bean_Dataset.csv")
# data, labels = prepareCaltech256("/home/hamza/university_admissions/UCSC/Research/I2I_Search/CLIP_features_caltech256")

# data, labels = prepareCaltech256("/home/hamza/university_admissions/UCSC/Research/I2I_Search/CLIP_features_caltech256")
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2)
# knn = KNeighborsClassifier(n_neighbors=10)
# knn.fit(x_train, y_train)
# y_pred = knn.predict(x_test[0].reshape(1, -1))
# print(y_pred)
# print(precision_score(y_test, y_pred, average = 'micro'))




##################### KNN how to get k indices of the labels per query point and use it to see the labels of each individual nearest neighbor returned per query point ######################

'''
neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(x_train)
indices = neigh.kneighbors(x_test[0].reshape(1,-1), return_distance=True)
distances = indices[0][0]
indices = indices[1][0]
predicted_labels = []
true_labels = []
for i in range(len(indices)):
    predicted_labels.append(y_train[indices[i]])
    true_labels.append(y_test[0])
true_positive = 0; true_negative = 0; false_positive = 0; false_negative = 0
total_correct_labels = 0
try:
    total_correct_labels = new_labels_count[true_labels[0]]
except:
    pass
if total_correct_labels > 0:
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == true_labels[i]:
            true_positive += 1
        else:
            false_positive += 1
    false_negative = new_labels_count[true_labels[0]] - true_positive
    true_negative = len(x_train) - true_positive - false_positive - false_negative
else:
    false_positive = len(predicted_labels)
    true_negative = len(x_train) - false_positive

print(predicted_labels, "\n")
print(distances, "\n")
print("True Label:", true_labels[0])
print("Total correct labels: ", total_correct_labels)
print("True Positive: ", true_positive)
print("False Positive: ", false_positive)
print("True Negative: ", true_negative)
print("False Negative: ", false_negative)
print(true_positive+true_negative+false_positive+false_negative)
print(len(x_train))

manual_precision = true_positive/(true_positive+false_positive)
manual_recall = 0
try:
    manual_recall = true_positive/(true_positive+false_negative)
except:
    pass
manual_f1_score = 0
try:
    manual_f1_score = (2*manual_precision*manual_recall)/(manual_precision+manual_recall)
except:
    pass
precision = precision_score(true_labels, predicted_labels, average='micro')
recall = recall_score(true_labels, predicted_labels, average='micro')
f1score = f1_score(true_labels, predicted_labels, average='micro')


print("\nManual precision: ", manual_precision)
print("Manual recall: ", manual_recall)
print("Manual f1 score: ", manual_f1_score)

print("\nprecision: ", precision)
print("recall: ", recall)
print("f1 score: ", f1score)

# print(true_labels)
# print(predicted_labels)
# print("accuracy: ", accuracy)
# print("precision: ", precision)
# print("recall: ", recall)
# print("f1score: ", f1score)

'''



