import csv
import math
import numpy as np
import statistics
from queue import PriorityQueue
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dataset = np.array([[178, 72], [163, 55], [179, 81], [168, 58], [181, 98], [170, 60], [184, 78], [171, 59], [182, 85], [165, 61], [185, 83], [168, 63], [177, 80], [172, 58], [188, 82], [164, 53]])

labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

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
    right_value_array = dataset[median_index+1:,]
    right_label_array = labels[median_index+1:,]
    
    if len(left_value_array):
        node.left = Node()
        node.left.parent = node
        node.left = Setup(left_value_array, left_label_array, depth+1, dim, node.left)
        
    if len(right_value_array):
        node.right = Node()
        node.right.parent = node
        node.right = Setup(right_value_array, right_label_array, depth+1, dim, node.right)
       
    return node

def inOrderTraversal(node):
    if node == None:
        return
    inOrderTraversal(node.left)
    print(node.value)
    inOrderTraversal(node.right)

def preOrderTraversal(node):
    if node == None:
        return
    print(node.value)
    preOrderTraversal(node.left)
    preOrderTraversal(node.right)



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

def printExplored(node):
    if node == None:
        return
    print(node.euclideanChecked)
    printExplored(node.left)
    printExplored(node.right)

def clearTree(node):
    if node == None:
        return
    node.leftExplored = False
    node.rightExplored = False
    node.euclideanChecked = False
    clearTree(node.left)
    clearTree(node.right)

"""
def Search(query, depth, dim, k, node, kNeighbors, section_reached):
    cd = depth % dim
    if node.left == None and node.right == None:
        if len(kNeighbors) < k:
            kNeighbors.append((euclideanDistance(query, node.value), node))
            Search(query, depth-1, dim, k, node.parent, kNeighbors, section_reached)
        else:
            worst_neighbor = return_worst_neighbor(kNeighbors)
            if euclideanDistance(query, node.value) < worst_neighbor[0]:
                kNeighbors.append((euclideanDistance(query, node.value), node))
            else:
                kNeighbors.append(worst_neighbor)
    else:
        if len(kNeighbors) < k:
            kNeighbors.append((euclideanDistance(query, node.value), node))
            if query[cd] < node.value[cd]:
                Search(query, depth+1, dim, k, node.left, maxHeap, section_reached)
            else:
                Search(query, depth+1, dim, k, node.right, maxHeap, section_reached)
        else:
            max_value = maxHeap.get()
            if euclideanDistance(query, node.value) < -1*max_value:
                maxHeap.put((-1*euclideanDistance(query, node.value), node))
            else:
                maxHeap.put((max_value))
                if abs(query[cd]-node.value[cd])
            
"""

def Search(query, depth, dim, k, node, kNeighbors):
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
    
    if node.left == None and node.right == None:
        kNeighbors = Search(query, depth-1, dim, k, node.parent, kNeighbors)
    else:
        if node.leftExplored and node.rightExplored:
            if node.parent is not None:
                kNeighbors = Search(query, depth-1, dim, k, node.parent, kNeighbors)
        elif (not node.leftExplored) and (not node.rightExplored):
            if query[cd] < node.value[cd]:
                node.leftExplored = True
                kNeighbors = Search(query, depth+1, dim, k, node.left, kNeighbors)
            else:
                node.rightExplored = True
                kNeighbors = Search(query, depth+1, dim, k, node.right, kNeighbors)
        else:
            worst_neighbor = return_worst_neighbor(kNeighbors)
            if abs(query[cd] - node.value[cd]) <= worst_neighbor[0]:
                if not node.leftExplored:
                    node.leftExplored = True
                    kNeighbors = Search(query, depth+1, dim, k, node.left, kNeighbors)
                else:
                    node.rightExplored = True
                    kNeighbors = Search(query, depth+1, dim, k, node.right, kNeighbors)
            else:
                if len(kNeighbors) < k:
                    if not node.leftExplored:
                        node.leftExplored = True
                        kNeighbors = Search(query, depth+1, dim, k, node.left, kNeighbors)
                    else:
                        node.rightExplored = True
                        kNeighbors = Search(query, depth+1, dim, k, node.right, kNeighbors)
    return kNeighbors


root_node = Node()
root_node = Setup(dataset, labels, 0, 2, root_node)
kNeighbors = []
# temp = np.array(dataset[0])
# dataset[0] = np.array(dataset[1])
# dataset[1] = np.array(temp)

# dataset, labels = sortData(0, dataset, labels)
# print(dataset)
# # print(dataset)
# print(labels)
# preOrderTraversal(root_node)
# printExplored(root_node)
kNeighbors = Search(np.array([167,57]), 0, 2, 3, root_node, kNeighbors)

# print(kNeighbors)
printExplored(root_node)
clearTree(root_node)
printExplored(root_node)