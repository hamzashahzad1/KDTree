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
from sklearn.neighbors import KDTree


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
            Search(query, depth+1, dim, k, node.left, kNeighbors)
            worst_neighbor = return_worst_neighbor(kNeighbors)
            if abs(query[cd] - node.value[cd]) <= worst_neighbor[0] or len(kNeighbors) < k:
                node.rightExplored = True
                nodeCount += 1
                Search(query, depth+1, dim, k, node.right, kNeighbors)
        else:
            node.rightExplored = True
            nodeCount += 1
            Search(query, depth+1, dim, k, node.right, kNeighbors)
            worst_neighbor = return_worst_neighbor(kNeighbors)
            if abs(query[cd] - node.value[cd]) <= worst_neighbor[0] or len(kNeighbors) < k:
                node.leftExplored = True
                nodeCount += 1
                Search(query, depth+1, dim, k, node.left, kNeighbors)
    else:
        worst_neighbor = return_worst_neighbor(kNeighbors)
        if abs(query[cd] - node.value[cd]) <= worst_neighbor[0] or len(kNeighbors) < k:
            if not node.leftExplored:
                node.leftExplored = True
                nodeCount += 1
                Search(query, depth+1, dim, k, node.left, kNeighbors)
            elif not node.rightExplored:
                node.rightExplored = True
                nodeCount += 1
                Search(query, depth+1, dim, k, node.right, kNeighbors)
    return kNeighbors

######################### IMPLEMENTATIONS #######################################

def KNN():
    data, labels, labels_dict = prepareData("DryBeanDataset/Dry_Bean_Dataset.csv")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print(precision_score(y_test, y_pred, average = 'micro'))

def _KDTree():
    data, labels, labels_dict = prepareData("DryBeanDataset/Dry_Bean_Dataset.csv")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2)
    root_node = Node()
    root_node = Setup(x_train, y_train,0, len(data[0]), root_node)
    y_pred = []
    count = 0
    for i in range(len(y_test)):
        kNeighbors = []; predicted_labels = []
        kNeighbors = modifiedSearch(x_test[i], 0, len(data[0]), 10,root_node, kNeighbors, 0, 4, math.log2(len(x_train)))

        clearExplored(root_node)
        for j in range(len(kNeighbors)):
            predicted_labels.append(kNeighbors[j][2])
        y_pred.append(statistics.mode(predicted_labels))
        if  y_pred[i] == y_test[i]:
            count += 1
    return precision_score(y_test, y_pred, average = 'micro')

def scikit_KDTree():
    data, labels, labels_dict = prepareData("DryBeanDataset/Dry_Bean_Dataset.csv")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2)
    tree = KDTree(x_train, leaf_size=2)
    dist, ind = tree.query(x_test, k=10, return_distance=True, dualtree=False, breadth_first=False)
    y_pred = []
    for i in range(len(ind)):
        temp_labels = []
        for j in range(len(ind[i])):
            temp_labels.append(y_train[ind[i][j]])
        y_pred.append(statistics.mode(temp_labels))
    print(precision_score(y_test, y_pred, average = 'micro'))
        

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
    
############################### TESTING #######################################

print(_KDTree())
KNN()
scikit_KDTree()


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

# all_neighbors = neighbors_scikitKDTree()

# all_neighbors = []

# for i in range(len(kNeighbors_list)):
#     ten_neighbors = []
#     for j in range(len(kNeighbors_list[i])):
#         ten_neighbors.append(kNeighbors_list[i][j][1])
#     all_neighbors.append(ten_neighbors)

# for i in range(len(all_neighbors)):
#     print(len(all_neighbors[i]))

# kd_neighbors = neighbors_KDTree()
# scikit_kd_neighbors = neighbors_scikitKDTree()

# print(len(kd_neighbors))
# print(len(scikit_kd_neighbors))
# for i in range(len(kd_neighbors[0])):

# my_neighbor = kd_neighbors[0][0]
# print(my_neighbor)
# for j in range(len(scikit_kd_neighbors[0])):
#     comparison_neighbor = scikit_kd_neighbors[0][j]
#     print(comparison_neighbor)

    # comparison = my_neighbor == comparison_neighbor
    # if comparison.all():
    #     print(my_neighbor)
    #     print(comparison_neighbor)
    # else:
    #     print(comparison.all())
# print(kd_neighbors[0][0])
# print(scikit_kd_neighbors[0][0])


# print(compare())
# l1 = [1,2,3,4,5,6,7,8,9]

# l2 = l1[4+1:]
# print(l2)