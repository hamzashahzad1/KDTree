import csv
import math
import numpy as np
import statistics
from queue import PriorityQueue
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

def KNN():
    data, labels, labels_dict = prepareData("DryBeanDataset/Dry_Bean_Dataset.csv")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print(accuracy_score(y_test, y_pred))

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.parent = None
        self.leftExplored = False
        self.rightExplored = False
        self.value = []
        self.label = None


def Setup(dataset, labels, depth, dim, node):
    cd = depth % dim
    dataset = dataset[dataset[:,cd].argsort()]
    median_index = math.ceil(len(dataset)/2)-1
    median_value = dataset[median_index]
    median_label = labels[median_index]
    node.value = median_value
    node.label = median_label
    left_value_array = []; right_value_array = []; left_label_array = []; right_label_array = []
    for i in range(median_index):
        left_value_array.append(dataset[i])
        left_label_array.append(labels[i])
    for i in range(median_index+1, len(dataset)):
        right_value_array.append(dataset[i])
        right_label_array.append(labels[i])
    left_value_array = np.array(left_value_array)
    right_value_array = np.array(right_value_array)
    left_label_array = np.array(left_label_array)
    right_label_array = np.array(right_label_array)
    if len(left_value_array):
        node.left = Node()
        node.left.parent = node
        node.left = Setup(left_value_array, left_label_array, depth+1, dim, node.left)
        
    if len(right_value_array):
        node.right = Node()
        node.right.parent = node
        node.right = Setup(right_value_array, right_label_array, depth+1, dim, node.right)
       
    return node

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


def modifiedSearch(query, depth, dim, k, node, maxHeap, section_reached, nodeCount, c, ln):
    if nodeCount >= c*k*ln:
        return maxHeap
    if node == None:
        return maxHeap
    cd = depth % dim
    if not section_reached: # complete required path not traversed
        
        if node.left == None and node.right == None: # Finally complete path traversed
            section_reached = True
        if maxHeap.qsize() < k: # k neighbors not found
            maxHeap.put((-1*euclideanDistance(query, node.value), (node.value, node.label)))
        else:
            try:
                max_point = maxHeap.get()
                if euclideanDistance(query, node.value) < -1*max_point[0]: # farthest neighbor farther away than current node
                    try:
                        maxHeap.put((-1*euclideanDistance(query, node.value), (node.value, node.label)))
                    except:
                        maxHeap.put((max_point))
                else: # Farthest neighbor closer than current node
                    maxHeap.put((max_point))
            except:
                pass
        if not section_reached: # not reached leaf of required path
            if query[cd] < node.value[cd]:
                node.leftExplored = True
                nodeCount += 1
                modifiedSearch(query, depth+1, dim, k, node.left, maxHeap, section_reached, nodeCount, c, ln)
            else:
                node.rightExplored = True
                nodeCount += 1
                modifiedSearch(query, depth+1, dim, k, node.right, maxHeap, section_reached, nodeCount, c, ln)
    
    else:
        if maxHeap.qsize() < k: # k neighbors not found
            maxHeap.put((-1*euclideanDistance(query, node.value), (node.value, node.label)))
        else:
            max_point = maxHeap.get()
            if euclideanDistance(query, node.value) < -1*max_point[0]: # farthest neighbor farther away than current node
                maxHeap.put((-1*euclideanDistance(query, node.value), (node.value, node.label)))
            else: # Farthest neighbor closer than current node
                maxHeap.put((max_point))
        if (node.left == None and node.right == None) or (node.leftExplored and node.rightExplored):
            nodeCount += 1
            modifiedSearch(query, depth-1, dim, k, node.parent, maxHeap, section_reached,nodeCount, c, ln)
        else:
            max_point = maxHeap.get()
            if euclideanDistance(query, node.value) < -1*max_point[0] or maxHeap.qsize() < k: # farthest neighbor farther away than current node
                if node.leftExplored:
                    maxHeap.put((max_point))
                    node.rightExplored = True
                    nodeCount += 1
                    modifiedSearch(query, depth+1, dim, k, node.right, maxHeap, section_reached, nodeCount, c, ln)
                elif node.rightExplored:
                    maxHeap.put((max_point))
                    node.leftExplored = True
                    nodeCount += 1
                    modifiedSearch(query, depth+1, dim, k, node.left, maxHeap, section_reached, nodeCount, c, ln)
                else:
                    maxHeap.put((max_point))
                    node.leftExplored = True
                    nodeCount += 1
                    modifiedSearch(query, depth+1, dim, k, node.left, maxHeap, section_reached, nodeCount, c, ln)
                    node.rightExplored = True
                    nodeCount += 1
                    modifiedSearch(query, depth+1, dim, k, node.right, maxHeap, section_reached, nodeCount, c, ln)
            else:
                maxHeap.put((max_point))
                modifiedSearch(query, depth-1, dim, k, node.parent, maxHeap, section_reached)
                nodeCount += 1        

    return maxHeap
        
def clearTree(node):
    if node == None:
        return
    node.leftExplored = False
    node.rightExplored = False
    clearTree(node.left)
    clearTree(node.right)

def KDTree():
    data, labels, labels_dict = prepareData("DryBeanDataset/Dry_Bean_Dataset.csv")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2)
    root_node = Node()
    root_node = Setup(x_train, y_train,0, len(data[0]), root_node)
    section_reached = False
    predicted_labels = []
    count = 0
    for i in range(len(y_test)):
        # print(i)
        maxHeap = PriorityQueue()
        maxHeap = Search(x_test[i], 0, len(data[0]), 10,root_node, maxHeap, section_reached)
        clearTree(root_node)
        for j in range(len(maxHeap.queue)):
            predicted_labels.append(maxHeap.queue[j][1][1])
        # maxHeap.queue.clear()
        if predicted_labels[i] == y_test[i]:
            count += 1
    return count/len(predicted_labels)

    # maxHeap = Search(data[12000], 0, len(data), 3,root_node, maxHeap, section_reached)

def modifiedKDTree():
    data, labels, labels_dict = prepareData("DryBeanDataset/Dry_Bean_Dataset.csv")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2)
    root_node = Node()
    root_node = Setup(x_train, y_train,0, len(data[0]), root_node)
    section_reached = False
    predicted_labels = []
    count = 0
    for i in range(len(y_test)):
        # print(i)
        maxHeap = PriorityQueue()
        maxHeap = modifiedSearch(x_test[i], 0, len(data[0]), 10,root_node, maxHeap, section_reached, 0, 3, math.log2(len(x_train)))
        clearTree(root_node)
        for j in range(len(maxHeap.queue)):
            predicted_labels.append(maxHeap.queue[j][1][1])
        # maxHeap.queue.clear()
        if predicted_labels[i] == y_test[i]:
            count += 1
    return count/len(predicted_labels)

def main():
    data, labels, labels_dict = prepareData("DryBeanDataset/Dry_Bean_Dataset.csv")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2)
    root_node = Node()
    root_node = Setup(x_train, y_train,0, len(data[0]), root_node)
    maxHeap = PriorityQueue()
    section_reached = False
    maxHeap = modifiedSearch(x_test[149], 0, len(data[0]), 10,root_node, maxHeap, section_reached, 0, 3, math.log(len(x_train)))
    temp_label_list = []
    print(KDTree())
    # for i in range(len(maxHeap.queue)):
    #     temp_label_list.append(maxHeap.queue[i][1][1])
    # print(len(temp_label_list))
    # print(statistics.mode(temp_label_list))
    # print(labels_dict)
    # print(root_node.left.parent.value == root_node.value)

main()

k = [3,2,6,4,9]
q = PriorityQueue()
for idx in range(len(k)):
    q.put((-1*k[idx], idx))


# print(q.qsize())
# print(q.get())
# print(q.get())
# print(q.qsize())

