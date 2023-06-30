import numpy as np
import random
from sklearn.model_selection import train_test_split

class dataProcessor:
    def __init__(self):
        self.data = []
    
    def readNpy(self,data_path):
        return np.load(data_path)
    
    def npyToTxt(self,npy_data,fname):
        rows = npy_data.shape[0]
        cols = npy_data.shape[1]
        file = open(fname, 'w')
        file.write(str(rows))
        file.write("\n")
        file.write(str(cols))
        file.write("\n")
        for i in range(rows):
            for j in range(cols):
                file.write(str(npy_data[i][j]))
                file.write(" ")
            if i < rows-1:
                file.write("\n")
        file.close()
    
    def txtToNpy(self,fname):
        file = open(fname, 'r')
        file_list = file.readlines()
        file_list = list(map(lambda s: s.strip(), file_list))
        rows = int(file_list.pop(0))
        cols = int(file_list.pop(0))
        data = []
        for i in range(len(file_list)):
            file_list[i] = file_list[i].split(" ")
            file_list[i] = list(map(float, file_list[i]))
        return np.array(file_list, dtype=np.single)
    
    def syntheticDataGenerator(self, rows, cols):
        data = []
        for i in range(rows):
            temp = []
            for j in range(cols):
                temp.append(random.uniform(-1,1))
            data.append(temp)
        data = np.array(data, dtype=np.single)
        

def write_train_test(input_data_path, input_labels_path, train_data_path, train_labels_path, test_data_path, test_labels_path, train_size, test_size):
    data_handler = dataProcessor()
    dataset = data_handler.readNpy(input_data_path)
    labels = data_handler.readNpy(input_labels_path)
    original_x_train, original_x_test, original_y_train, original_y_test = train_test_split(dataset, labels, test_size= 0.2)
    x_train = np.copy(original_x_train[0:train_size])
    y_train = np.copy(original_y_train[0:train_size])
    x_test = np.copy(original_x_test[0:test_size])
    y_test = np.copy(original_y_test[0:test_size])
    np.save("x_train_caltech256_256", x_train)
    np.save("x_test_caltech256_256", x_test)
    np.save("y_train_caltech256_256", y_train)
    np.save("y_test_caltech256_256", y_test)
    data_handler.npyToTxt(x_train, train_data_path)
    data_handler.npyToTxt(x_test, test_data_path)
    data_handler.npyToTxt(y_train.reshape(-1, 1), train_labels_path)
    data_handler.npyToTxt(y_test.reshape(-1, 1), test_labels_path)

if __name__ == "__main__":
    # write_train_test("/home/hamza/university_admissions/UCSC/Research/I2I_Search/caltech256_clip_512.npy", "/home/hamza/university_admissions/UCSC/Research/I2I_Search/caltech256_labels.npy", "x_train_caltech256_256.txt", "y_train_caltech256_256.txt", "x_test_caltech256_256.txt", "y_test_caltech256_256.txt", 205, 51)
    print("hello")
