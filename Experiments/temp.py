import os
import numpy as np
from sklearn.decomposition import KernelPCA
path = "/home/hamza/university_admissions/UCSC/Research/I2I_Search/CLIP_features_caltech256"

# for (root, dirs, file) in os.walk(path):
#     print(file)

kpca = KernelPCA(n_components=64, kernel='cosine')

file_list = os.listdir(path)
file_list.sort()
label = []
caltech256_clip_features = []
for i in range(len(file_list)):
    curr_path = path + "/" + file_list[i]
    curr_features = np.load(curr_path)
    for j in range(len(curr_features)):
        label.append(i)
    caltech256_clip_features.extend(curr_features)
caltech256_clip_features = np.asarray(caltech256_clip_features)
# reduced_array = kpca.fit_transform(caltech256_clip_features)
print(caltech256_clip_features)
# print(reduced_array.shape)
print(len(label))

