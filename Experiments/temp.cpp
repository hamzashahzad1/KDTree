#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <map>

using key_type = u_int64_t;

struct dataPoint
{
    std::vector<double> value;
    int label;
    dataPoint& operator=(dataPoint const &other){
        label = other.label;
        value = other.value;
        return *this;
    }
};

// contains the datapoint, distance of datapoint from query, and the key of the node from where this datapoint was retrieved
struct neighborPoint
{
    key_type key;
    dataPoint datapoint;
    double distance;
};

struct Node
{
    key_type key;
    dataPoint datapoint;
    Node* parent;
    Node* left;
    Node* right;
    Node(){
        parent = NULL; left = NULL; right = NULL;
    }

};

double euclideadDistance(std::vector<double> point1, std::vector<double> point2)
{
    std::vector<double> resultVector;
    uint length = point1.size();
    double sum = 0;
    for(uint i = 0; i < length; i++){
        resultVector.push_back(point1[i] - point2[i]);
        resultVector[i] = pow(resultVector[i],2);
        sum += resultVector[i];
    }
    return sqrt(sum);
}

neighborPoint worstNeighbor(std::vector<neighborPoint> &neighbors)
{
    double maxDistance = -1;
    neighborPoint worstPoint;
    uint length = neighbors.size();
    uint index;
    for (uint i = 0; i < length; i++){
        if (neighbors[i].distance > maxDistance){
            maxDistance = neighbors[i].distance;
            worstPoint = neighbors[i];
            index = i;
        }
    }
    neighbors.erase(neighbors.begin() + index);
    return worstPoint;
}

std::vector<dataPoint> sortData(int cd, std::vector<dataPoint> dataset)
{
    uint length = dataset.size();
    for(uint i = 0; i < length-1; i++){
        for(uint j = i+1; j < length; j++){
            if (dataset[i].value[cd] > dataset[j].value[cd]){
                dataPoint temp = dataset[i];
                dataset[i] = dataset[j];
                dataset[j] = temp;
            }
        }
    }
    return dataset;
}

std::vector<dataPoint> slicing(std::vector<dataPoint> arr, int X, int Y)
{
    auto start = arr.begin() + X;
    auto end = arr.begin() + Y + 1;
 
    std::vector<dataPoint> result(Y - X +1);
    copy(start, end, result.begin());
 
    return result;
}

Node* setup(Node* node, std::vector<dataPoint> dataset, int depth, int dim)
{
    int cd = depth % dim;
    dataset = sortData(cd, dataset);
    int medianIndex = dataset.size()/2;
    if(!(dataset.size()%2)) medianIndex -= 1;
    dataPoint medianDatapoint = dataset[medianIndex];
    node->datapoint = medianDatapoint;
    std::vector<dataPoint> leftDataset = slicing(dataset,0, medianIndex-1);
    std::vector<dataPoint> rightDataset = slicing(dataset, medianIndex+1, dataset.size()-1);
    if(leftDataset.size()){
        node->left = new Node;
        node->left->parent = node;
        node->left = setup(node->left, leftDataset, depth+1, dim);
    }
    if(rightDataset.size()){
        node->right = new Node;
        node->right->parent = node;
        node->right = setup(node->right, rightDataset, depth+1, dim);
    }
    
    // std::cout << medianIndex << std::endl;
    // std::cout << leftDataset.size() << std::endl;
    // std::cout << rightDataset.size() << std::endl;
    return node;

}

std::vector<neighborPoint> search(Node* node, std::vector<double> query, std::vector<neighborPoint> &kNeighbors, int depth, int dim, int k)
{
    if(node == NULL) return kNeighbors;
    int cd = depth%dim;
    if(kNeighbors.size() < k){
        neighborPoint point;
        point.key = node->key;
        point.datapoint = node->datapoint;
        point.distance = euclideadDistance(query, node->datapoint.value);
        kNeighbors.push_back(point);
    }
    else{
        neighborPoint worstPoint = worstNeighbor(kNeighbors);
        if(euclideadDistance(query, node->datapoint.value) < worstPoint.distance){
            neighborPoint point;
            point.key = node->key;
            point.datapoint = node->datapoint;
            point.distance = euclideadDistance(query, node->datapoint.value);
            kNeighbors.push_back(point);
        }
        else kNeighbors.push_back(worstPoint);
    }

    if(query[cd] < node->datapoint.value[cd]){
        kNeighbors = search(node->left, query, kNeighbors, depth+1, dim, k);
        neighborPoint worstPoint = worstNeighbor(kNeighbors);
        if( (abs(query[cd] - node->datapoint.value[cd]) <= worstPoint.distance) || kNeighbors.size() < k ){
            kNeighbors.push_back(worstPoint);
            kNeighbors = search(node->right, query, kNeighbors, depth+1, dim, k);
        }
        else kNeighbors.push_back(worstPoint);
    }
    else{
        kNeighbors = search(node->right, query, kNeighbors, depth+1, dim, k);
        neighborPoint worstPoint = worstNeighbor(kNeighbors);
        if( (abs(query[cd] - node->datapoint.value[cd]) <= worstPoint.distance) || kNeighbors.size() < k ){
            kNeighbors.push_back(worstPoint);
            kNeighbors = search(node->left, query, kNeighbors, depth+1, dim, k);
        }
        else kNeighbors.push_back(worstPoint);
    }
    return kNeighbors;
}

std::vector<neighborPoint> modifiedSearch(Node* node, std::vector<double> query, std::vector<neighborPoint> &kNeighbors, int depth, int dim, int k, int &nodeCount, int c, int ln)
{
    if (nodeCount >= c*k*ln){
        nodeCount--;
        return kNeighbors;
    }
    if(node == NULL) {
        nodeCount--;
        return kNeighbors;
    }
    int cd = depth%dim;
    if(kNeighbors.size() < k){
        neighborPoint point;
        point.key = node->key;
        point.datapoint = node->datapoint;
        point.distance = euclideadDistance(query, node->datapoint.value);
        kNeighbors.push_back(point);
    }
    else{
        neighborPoint worstPoint = worstNeighbor(kNeighbors);
        if(euclideadDistance(query, node->datapoint.value) < worstPoint.distance){
            neighborPoint point;
            point.key = node->key;
            point.datapoint = node->datapoint;
            point.distance = euclideadDistance(query, node->datapoint.value);
            kNeighbors.push_back(point);
        }
        else kNeighbors.push_back(worstPoint);
    }

    if(query[cd] < node->datapoint.value[cd]){
        kNeighbors = modifiedSearch(node->left, query, kNeighbors, depth+1, dim, k, ++nodeCount, c, ln);
        neighborPoint worstPoint = worstNeighbor(kNeighbors);
        if( (abs(query[cd] - node->datapoint.value[cd]) <= worstPoint.distance) || kNeighbors.size() < k ){
            kNeighbors.push_back(worstPoint);
            kNeighbors = modifiedSearch(node->right, query, kNeighbors, depth+1, dim, k, ++nodeCount, c, ln);
        }
        else kNeighbors.push_back(worstPoint);
    }
    else{
        kNeighbors = modifiedSearch(node->right, query, kNeighbors, depth+1, dim, k, ++nodeCount, c, ln);
        neighborPoint worstPoint = worstNeighbor(kNeighbors);
        if( (abs(query[cd] - node->datapoint.value[cd]) <= worstPoint.distance) || kNeighbors.size() < k ){
            kNeighbors.push_back(worstPoint);
            kNeighbors = modifiedSearch(node->left, query, kNeighbors, depth+1, dim, k, ++nodeCount, c, ln);
        }
        else kNeighbors.push_back(worstPoint);
    }
    return kNeighbors;
}

void printVector(std::vector<double> myVector)
{
    std::cout << "{";
    for(uint i = 0; i < myVector.size(); i++){
        std::cout << myVector[i] << " ";
    }
    std::cout << "}\n"; 
}

void swap_vectors(std::vector<double> &vecA, std::vector<double> &vecB)
{
    std::vector<double> temp;
    if(vecA[0] > vecB[0]){
        temp = vecA;
        vecA = vecB;
        vecB = temp;
    }
}

void tokenize(std::string const &str, const char delim, std::vector<double> &out) 
{ 
    std::stringstream ss(str); 
 
    std::string s; 
    while (std::getline(ss, s, delim)) { 
        double num = std::stold(s);
        out.push_back(num); 
    } 
} 

std::vector<std::vector<double>> readData(std::string fname)
{
    std::ifstream myFile(fname);
    int count = 0; int rows = 0, cols = 0;
    std::string current_data = "";
    std::vector<std::vector<double>> data;
    if(myFile.is_open()){
        while(myFile){
            std::getline(myFile, current_data);
            if(count == 0){
                rows = std::stoi(current_data);
            } 
            else if(count == 1){
                cols = std::stoi(current_data);
            } 
            else {
                std::vector<double> numbers_vector;
                tokenize(current_data, ' ', numbers_vector);
                data.push_back(numbers_vector);
            }
            count++;
        }
    }
    data.pop_back();
    return data;
}

std::vector<int> readLabels(std::string fname)
{
    std::ifstream myFile(fname);
    int count = 0; int rows = 0, cols = 0;
    std::string current_data = "";
    std::vector<int> labels;
    if(myFile.is_open()){
        while(myFile){
            std::getline(myFile, current_data);
            if(count == 0){
                rows = std::stoi(current_data);
            } 
            else if(count == 1){
                cols = std::stoi(current_data);
            } 
            else {
                std::stringstream ss(current_data); 
                std::string s;
                std::getline(ss, s, ' ');
                int label = std::stoi(s);
                labels.push_back(label);
            }
            count++;
        }
    }
    labels.pop_back();
    return labels;
}

std::vector<dataPoint> createDataset(std::string data_file, std::string labels_file)
{
    std::vector<std::vector<double>> data = readData(data_file);
    std::vector<int> labels = readLabels(labels_file);

    std::vector<dataPoint> dataset;

    for(int i = 0; i < data.size(); i++){
        dataPoint tempPoint;
        tempPoint.label = labels[i];
        tempPoint.value = data[i];
        dataset.push_back(tempPoint);
    }
    return dataset;
}

int main()
{    
   

    std::string train_data_file = "/home/hamza/university_admissions/UCSC/Research/I2I_Search/KD-Trees/Experiments/x_train_caltech256_256.txt";
    std::string train_labels_file = "/home/hamza/university_admissions/UCSC/Research/I2I_Search/KD-Trees/Experiments/y_train_caltech256_256.txt";
    std::string test_data_file = "/home/hamza/university_admissions/UCSC/Research/I2I_Search/KD-Trees/Experiments/x_test_caltech256_256.txt";
    std::string test_labels_file = "/home/hamza/university_admissions/UCSC/Research/I2I_Search/KD-Trees/Experiments/y_test_caltech256_256.txt";

    std::vector<dataPoint> train_dataset = createDataset(train_data_file, train_labels_file);
    std::vector<dataPoint> test_dataset = createDataset(test_data_file, test_labels_file);
    int dim = train_dataset[0].value.size();
    int k = 10;
    
    Node* root = new Node();
    // std::vector<dataPoint> smallDataset = slicing(test_dataset, 0, 2);
    // std::cout << smallDataset.size() << std::endl;
    root = setup(root, train_dataset, 0, dim);

    std::vector<neighborPoint> kNeighbors;

    kNeighbors = search(root, test_dataset[0].value, kNeighbors, 0, dim, k);
    for(int i = 0; i < kNeighbors.size(); i++){
        std::cout << kNeighbors[i].datapoint.label << " ";
    }
    std::cout << std::endl;
    std::cout << test_dataset[0].label << std::endl;

    std::vector<neighborPoint> modifiedKNeighbors;
    int nodeCount = 0; int c = 3; int ln = (int)ceil(log2(train_dataset.size()));
    
    modifiedKNeighbors = modifiedSearch(root, test_dataset[0].value, modifiedKNeighbors, 0, dim, k, nodeCount, c, ln);
    std::cout << nodeCount << std::endl;
    for(int i = 0; i < modifiedKNeighbors.size(); i++){
        std::cout << modifiedKNeighbors[i].datapoint.label << " ";  
    }
    std::cout << std::endl;

    return 0; 
}