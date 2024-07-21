import csv 
import random 
import math
import matplotlib.pyplot as plt
import numpy as np

# loading the dataset

dataset = []
with open('cm_dataset_2.csv', newline='') as file:
    csv_file_reader = csv.reader(file)
    for row in csv_file_reader:
        dataset.append(row)

# converting the dataset into floating points

for i in range(len(dataset)):
    correct_point = [float(dataset[i][0]), float(dataset[i][1])]
    dataset[i] = correct_point

# plotting the dataset
    
plt.plot([point[0] for point in dataset], [point[1] for point in dataset], 'ro')
plt.xlabel('X')
plt.ylabel('Y')

# function to calculate the euclidean distance between 2 points

def euclidean_distance(p, q):
    dist = 0
    for i in range(len(p)):
        dist += (float(p[i]) - float(q[i])) ** 2
    return math.sqrt(dist)

# function to calculate the error in that iteration of the loop

def calculate_error(error, means, clusters):
    e = 0
    total_e = 0
    for i in range(len(means)):
        e = 0
        for point in clusters[i]:
            for j in range(len(point)):
                e += (float(means[i][j]) - float(point[j])) ** 2
        if len(clusters[i]) > 0:
            total_e = total_e + e
    error.append(total_e)

# function that implements k-means clustering
# takes in the initial means, the dataset, and the number of clusters

def k_means_clustering(initial_means, dataset, k):
    means = initial_means
    convergence = False
    error = []
    # break out from the loop when convergence occurs. otherwirse, proceed for another iteration
    while(1):
        clusters = [[] for _ in range(k)]
        # reassignment step - finding which mean the datapoint is closer to
        for i in dataset:
            distances_from_centroids = []
            for j in range(k):
                distances_from_centroids.append([euclidean_distance(i, means[j]), j])
            required_cluster = min(distances_from_centroids, key=lambda x: x[0])[1]
            clusters[required_cluster].append(i)
        # finding the new means of the new clusters
        new_means = []
        for i in range(k):
            if clusters[i]:
                x_mean = sum([point[0] for point in clusters[i]])/len(clusters[i])
                y_mean = sum([point[1] for point in clusters[i]])/len(clusters[i])
                new_means.append([x_mean, y_mean])
            else:
                new_means.append(means[i])
        # checking for convergence. if the old means and new means are equal, then break
        if all(means[i] == new_means[i] for i in range(k)):
            break
        # if not convergence, find the error associated with the current iteration
        calculate_error(error, means, clusters)
        means = new_means 

    return clusters, means, error

# first random initialization - taking first 500 in one cluster and next in another and finding initial means.

initial_cluster1 = dataset[:500]
initial_cluster2 = dataset[500:]
initial_means1 = [0,0]
initial_means1[0] = [sum([point[0] for point in initial_cluster1])/len(initial_cluster1), sum([point[1] for point in initial_cluster1])/len(initial_cluster1)]
initial_means1[1] = [sum([point[0] for point in initial_cluster2])/len(initial_cluster2), sum([point[1] for point in initial_cluster2])/len(initial_cluster2)]
cluster1, means1, error1 = k_means_clustering(initial_means1, dataset, 2)

# second random initialization - odd index points in one cluster and even in another

initial_cluster1 = [dataset[i] for i in range(len(dataset)) if i%2 == 0]
initial_cluster2 = [dataset[i] for i in range(len(dataset)) if i%2 == 1]
initial_means2 = [0,0]
initial_means2[0] = [sum([point[0] for point in initial_cluster1])/len(initial_cluster1), sum([point[1] for point in initial_cluster1])/len(initial_cluster1)]
initial_means2[1] = [sum([point[0] for point in initial_cluster2])/len(initial_cluster2), sum([point[1] for point in initial_cluster2])/len(initial_cluster2)]
cluster2, means2, error2 = k_means_clustering(initial_means2, dataset, 2)

# third random initialization - taking two points from dataset at random to be initial means

initial_means3 = random.sample(dataset, 2)
cluster3, means3, error3 = k_means_clustering(initial_means3, dataset, 2)

# fourth random initialization - those with y coordinate >-5 in one and the rest in another

initial_cluster1 = [dataset[i] for i in range(len(dataset)) if dataset[i][1] < -5.0]
initial_cluster2 = [dataset[i] for i in range(len(dataset)) if dataset[i][1] >= -5.0]
initial_means4 = [0,0]
initial_means4[0] = [sum([point[0] for point in initial_cluster1])/len(initial_cluster1), sum([point[1] for point in initial_cluster1])/len(initial_cluster1)]
initial_means4[1] = [sum([point[0] for point in initial_cluster2])/len(initial_cluster2), sum([point[1] for point in initial_cluster2])/len(initial_cluster2)]
cluster4, means4, error4 = k_means_clustering(initial_means4, dataset, 2)

# fifth random initialization - k-means++ initialization

def max_euclidean_distance(p, dataset):
    max = -1
    furthest_point = None
    for i in dataset:
        if (np.linalg.norm(np.array(p) - np.array(i)) > max):
            max = np.linalg.norm(np.array(p) - np.array(i))
            furthest_point = i
    return furthest_point

initial_means5 = random.sample(dataset, 2)
initial_means5[1] = max_euclidean_distance(initial_means5[0], dataset)
cluster5, means5, error5 = k_means_clustering(initial_means5, dataset, 2)

# plotting the clusters

import matplotlib.pyplot as plt

plt.figure(figsize=(35, 5))
# plt.title('Clusters')
ax = plt.subplot(1,5,1)
plt.plot([point[0] for point in cluster1[0]], [point[1] for point in cluster1[0]], 'ro', label='Cluster 0')
plt.plot(means1[0][0], means1[0][1], 'bo', label='Centroid 0')
plt.plot([point[0] for point in cluster1[1]], [point[1] for point in cluster1[1]], 'go', label='Cluster 1')
plt.plot(means1[1][0], means1[1][1], 'ko', label='Centroid 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization 1')
plt.legend()
plt.grid(True)
# plt.figure(figsize=(8, 6))
ax = plt.subplot(1,5,2)
plt.plot([point[0] for point in cluster2[0]], [point[1] for point in cluster2[0]], 'ro', label='Cluster 0')
plt.plot(means2[0][0], means2[0][1], 'bo', label='Centroid 0')
plt.plot([point[0] for point in cluster2[1]], [point[1] for point in cluster2[1]], 'go', label='Cluster 1')
plt.plot(means2[1][0], means2[1][1], 'ko', label='Centroid 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization 2')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,5,3)
plt.plot([point[0] for point in cluster3[0]], [point[1] for point in cluster3[0]], 'ro', label='Cluster 0')
plt.plot(means3[0][0], means3[0][1], 'bo', label='Centroid 0')
plt.plot([point[0] for point in cluster3[1]], [point[1] for point in cluster3[1]], 'go', label='Cluster 1')
plt.plot(means3[1][0], means3[1][1], 'ko', label='Centroid 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization 3')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,5,4)
plt.plot([point[0] for point in cluster4[0]], [point[1] for point in cluster4[0]], 'ro', label='Cluster 0')
plt.plot(means4[0][0], means4[0][1], 'bo', label='Centroid 0')
plt.plot([point[0] for point in cluster4[1]], [point[1] for point in cluster4[1]], 'go', label='Cluster 1')
plt.plot(means4[1][0], means4[1][1], 'ko', label='Centroid 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization 4')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,5,5)
plt.plot([point[0] for point in cluster5[0]], [point[1] for point in cluster5[0]], 'ro', label='Cluster 0')
plt.plot(means5[0][0], means5[0][1], 'bo', label='Centroid 0')
plt.plot([point[0] for point in cluster5[1]], [point[1] for point in cluster5[1]], 'go', label='Cluster 1')
plt.plot(means5[1][0], means5[1][1], 'ko', label='Centroid 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization 5')
plt.legend()
plt.grid(True)
plt.show()

# plotting the errors associated with each cluster

plt.figure(figsize=(35, 5))
ax = plt.subplot(1,5,1)
plt.plot(error1)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Cluster 1')
ax = plt.subplot(1,5,2)
plt.plot(error2)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Cluster 2')
ax = plt.subplot(1,5,3)
plt.plot(error3)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Cluster 3')
ax = plt.subplot(1,5,4)
plt.plot(error4)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Cluster 4')
ax = plt.subplot(1,5,5)
plt.plot(error5)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Cluster 5')
plt.show()

# part ii

# the random initialization fixed is taking two random points from the dataset 
# to be initial means

initial_means_clusters_2 = random.sample(dataset, 2)
cluster_num_2, means_num_2, error_num_2 = k_means_clustering(initial_means_clusters_2, dataset, 2)

initial_means_clusters_3 = random.sample(dataset, 3)
cluster_num_3, means_num_3, error_num_3 = k_means_clustering(initial_means_clusters_3, dataset, 3)

initial_means_clusters_4 = random.sample(dataset, 4)
cluster_num_4, means_num_4, error_num_4 = k_means_clustering(initial_means_clusters_4, dataset, 4)

initial_means_clusters_5 = random.sample(dataset, 5)
cluster_num_5, means_num_5, error_num_5 = k_means_clustering(initial_means_clusters_5, dataset, 5)

# plotting Voronoi regions for K = 2 and K = 3

plt.figure(figsize=(25, 7))
ax = plt.subplot(1,2,1)
plt.plot([point[0] for point in cluster_num_2[0]], [point[1] for point in cluster_num_2[0]], 'ro', label='Cluster 0')
plt.plot(means_num_2[0][0], means_num_2[0][1], 'bo', label='Centroid 0')
plt.plot([point[0] for point in cluster_num_2[1]], [point[1] for point in cluster_num_2[1]], 'go', label='Cluster 1')
plt.plot(means_num_2[1][0], means_num_2[1][1], 'ko', label='Centroid 1')
arr_means_num_2 = np.array(means_num_2)
contour_x, contour_y = np.meshgrid(np.arange(-16, 16, 0.1), np.arange(-16, 6, 0.1))
contour_assgn = np.zeros_like(contour_x, dtype=int)
for x in range(contour_x.shape[0]):
    for y in range(contour_x.shape[1]):
        k = np.linalg.norm(arr_means_num_2 - [contour_x[x, y], contour_y[x, y]], axis=1)
        contour_assgn[x,y] = np.argmin(k)
ax.contourf(contour_x, contour_y, contour_assgn, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Voronoi Regions Visualization for K = 2')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,2,2)
plt.plot([point[0] for point in cluster_num_3[0]], [point[1] for point in cluster_num_3[0]], 'ro', label='Cluster 0')
plt.plot(means_num_3[0][0], means_num_3[0][1], 'bo', label='Centroid 0')
plt.plot([point[0] for point in cluster_num_3[1]], [point[1] for point in cluster_num_3[1]], 'go', label='Cluster 1')
plt.plot(means_num_3[1][0], means_num_3[1][1], 'ko', label='Centroid 1')
plt.plot([point[0] for point in cluster_num_3[2]], [point[1] for point in cluster_num_3[2]], 'co', label='Cluster 2')
plt.plot(means_num_3[2][0], means_num_3[2][1], 'mo', label='Centroid 2')
arr_means_num_3 = np.array(means_num_3)
contour_x, contour_y = np.meshgrid(np.arange(-16, 16, 0.1), np.arange(-16, 6, 0.1))
contour_assgn = np.zeros_like(contour_x, dtype=int)
for x in range(contour_x.shape[0]):
    for y in range(contour_x.shape[1]):
        k = np.linalg.norm(arr_means_num_3 - [contour_x[x, y], contour_y[x, y]], axis=1)
        contour_assgn[x,y] = np.argmin(k)
ax.contourf(contour_x, contour_y, contour_assgn, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Voronoi Regions Visualization for K = 3')
plt.legend()
plt.grid(True)
plt.show()

# plotting Voronoi regions for K = 4 and K = 5

plt.figure(figsize=(25, 7))
ax = plt.subplot(1,2,1)
plt.plot([point[0] for point in cluster_num_4[0]], [point[1] for point in cluster_num_4[0]], 'ro', label='Cluster 0')
plt.plot(means_num_4[0][0], means_num_4[0][1], 'bo', label='Centroid 0')
plt.plot([point[0] for point in cluster_num_4[1]], [point[1] for point in cluster_num_4[1]], 'go', label='Cluster 1')
plt.plot(means_num_4[1][0], means_num_4[1][1], 'ko', label='Centroid 1')
plt.plot([point[0] for point in cluster_num_4[2]], [point[1] for point in cluster_num_4[2]], 'co', label='Cluster 2')
plt.plot(means_num_4[2][0], means_num_4[2][1], 'mo', label='Centroid 2')
plt.plot([point[0] for point in cluster_num_4[3]], [point[1] for point in cluster_num_4[3]], 'bo', label='Cluster 3')
plt.plot(means_num_4[3][0], means_num_4[3][1], 'yo', label='Centroid 3')
arr_means_num_3 = np.array(means_num_4)
contour_x, contour_y = np.meshgrid(np.arange(-16, 16, 0.1), np.arange(-16, 6, 0.1))
contour_assgn = np.zeros_like(contour_x, dtype=int)
for x in range(contour_x.shape[0]):
    for y in range(contour_x.shape[1]):
        k = np.linalg.norm(arr_means_num_3 - [contour_x[x, y], contour_y[x, y]], axis=1)
        contour_assgn[x,y] = np.argmin(k)
ax.contourf(contour_x, contour_y, contour_assgn, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Vornoi Regions Visualization for K = 4')
plt.legend()
plt.grid(True)
ax = plt.subplot(1,2,2)
plt.plot([point[0] for point in cluster_num_5[0]], [point[1] for point in cluster_num_5[0]], 'ro', label='Cluster 0')
plt.plot(means_num_5[0][0], means_num_5[0][1], 'bo', label='Centroid 0')
plt.plot([point[0] for point in cluster_num_5[1]], [point[1] for point in cluster_num_5[1]], 'go', label='Cluster 1')
plt.plot(means_num_5[1][0], means_num_5[1][1], 'ko', label='Centroid 1')
plt.plot([point[0] for point in cluster_num_5[2]], [point[1] for point in cluster_num_5[2]], 'co', label='Cluster 2')
plt.plot(means_num_5[2][0], means_num_5[2][1], 'mo', label='Centroid 2')
plt.plot([point[0] for point in cluster_num_5[3]], [point[1] for point in cluster_num_5[3]], 'bo', label='Cluster 3')
plt.plot(means_num_5[3][0], means_num_5[3][1], 'yo', label='Centroid 3')
plt.plot([point[0] for point in cluster_num_5[4]], [point[1] for point in cluster_num_5[4]], 'yo', label='Cluster 4')
plt.plot(means_num_5[4][0], means_num_5[4][1], 'ro', label='Centroid 4')
arr_means_num_3 = np.array(means_num_5)
contour_x, contour_y = np.meshgrid(np.arange(-16, 16, 0.1), np.arange(-16, 6, 0.1))
contour_assgn = np.zeros_like(contour_x, dtype=int)
for x in range(contour_x.shape[0]):
    for y in range(contour_x.shape[1]):
        k = np.linalg.norm(arr_means_num_3 - [contour_x[x, y], contour_y[x, y]], axis=1)
        contour_assgn[x,y] = np.argmin(k)
ax.contourf(contour_x, contour_y, contour_assgn, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Voronoi Regions Visualization for K = 5')
plt.legend()
plt.grid(True)
plt.show()

# part iii

# defining different kernel functions

def poly_kernel_function(x, y, d):
    return (np.dot(x,y) + 1) ** d

def radial_kernel_calculator(orig_data, sigma):
    orig_data = np.array(orig_data)
    kernel = np.zeros((len(orig_data), len(orig_data)))
    for i in range(len(orig_data)):
        for j in range(len(orig_data)):
            squared_distance = np.sum((orig_data[i] - orig_data[j]) ** 2)
            kernel[i, j] = np.exp(-squared_distance/ (2 * sigma * sigma))
    return kernel

def kernel_calculator(orig_data, d):
    kernel = np.zeros((len(orig_data), len(orig_data)))
    for i in range(len(orig_data)):
        for j in range(len(orig_data)):
            kernel[i, j] = poly_kernel_function(orig_data[i], orig_data[j], d)
    return kernel

# polynomial kernel

# finding eigenvalues and eigenvectors

kernel = kernel_calculator(dataset, 2)
eigenvalues, eigenvectors = np.linalg.eigh(kernel)

# taking top 2 eigenvectors

w_1 = eigenvectors[:, -1].transpose()
w_2 = eigenvectors[:, -2].transpose()

# constructing H, normalizing

H = np.column_stack((w_1, w_2))
H = H/np.linalg.norm(H, axis=1, keepdims=True)

# converting this np.array into a list

spectral_dataset = []
for i in H:
    spectral_dataset.append(i.tolist())

initial_mean = random.sample(spectral_dataset, 2)
sclusters, smeans, serror = k_means_clustering(initial_mean, spectral_dataset, 2) 

kernel_3 = kernel_calculator(dataset, 3)
eigenvalues_3, eigenvectors_3 = np.linalg.eigh(kernel_3)
w_1_3 = eigenvectors_3[:, -1].transpose()
w_2_3 = eigenvectors_3[:, -2].transpose()
H_3 = np.column_stack((w_1_3, w_2_3))
H_3 = H_3/np.linalg.norm(H_3, axis=1, keepdims=True)
K = []
K.append(H_3[0].tolist())
spectral_dataset_3 = []
for i in H_3:
    spectral_dataset_3.append(i.tolist())
initial_mean_3 = random.sample(spectral_dataset_3, 2)
sclusters_3, smeans_3, serror_3 = k_means_clustering(initial_mean_3, spectral_dataset_3, 2) 

kernel_4 = kernel_calculator(dataset, 4)
eigenvalues_4, eigenvectors_4 = np.linalg.eigh(kernel_4)
w_1_4 = eigenvectors_4[:, -1].transpose()
w_2_4 = eigenvectors_4[:, -2].transpose()
H_4 = np.column_stack((w_1_4, w_2_4))
H_4 = H_4 / np.linalg.norm(H_4, axis=1, keepdims=True)
K = []
K.append(H_4[0].tolist())
spectral_dataset_4 = []
for i in H_4:
    spectral_dataset_4.append(i.tolist())
initial_mean_4 = random.sample(spectral_dataset_4, 2)
sclusters_4, smeans_4, serror_4 = k_means_clustering(initial_mean_4, spectral_dataset_4, 2)

kernel_5 = kernel_calculator(dataset, 5)
eigenvalues_5, eigenvectors_5 = np.linalg.eigh(kernel_5)
w_1_5 = eigenvectors_5[:, -1].transpose()
w_2_5 = eigenvectors_5[:, -2].transpose()
H_5 = np.column_stack((w_1_5, w_2_5))
H_5 = H_5 / np.linalg.norm(H_5, axis=1, keepdims=True)
K = []
K.append(H_5[0].tolist())
spectral_dataset_5 = []
for i in H_5:
    spectral_dataset_5.append(i.tolist())
initial_mean_5 = random.sample(spectral_dataset_5, 2)
sclusters_5, smeans_5, serror_5 = k_means_clustering(initial_mean_5, spectral_dataset_5, 2)

# finding representation of the original data using the spectral clustering result

og_cluster_1 = [[] for _ in range(2)]
for i in range(len(spectral_dataset)):
    if spectral_dataset[i] in sclusters[0]:
        og_cluster_1[0].append(dataset[i])
    else:
        og_cluster_1[1].append(dataset[i])

og_cluster_2 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_3)):
    if spectral_dataset_3[i] in sclusters_3[0]:
        og_cluster_2[0].append(dataset[i])
    else:
        og_cluster_2[1].append(dataset[i])

og_cluster_3 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_4)):
    if spectral_dataset_4[i] in sclusters_4[0]:
        og_cluster_3[0].append(dataset[i])
    else:
        og_cluster_3[1].append(dataset[i])

og_cluster_4 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_5)):
    if spectral_dataset_5[i] in sclusters_5[0]:
        og_cluster_4[0].append(dataset[i])
    else:
        og_cluster_4[1].append(dataset[i])

# plotting the spectral clustering results
        
plt.figure(figsize=(18, 5))
plt.suptitle('Spectral Clustering, Polynomial Kernel', fontsize=16)
ax = plt.subplot(1,4,1)
plt.plot([point[0] for point in sclusters[0]], [point[1] for point in sclusters[0]], 'co', label='Cluster 0')
plt.plot(smeans[0][0], smeans[0][1], 'yo', label='Centroid 0')
plt.plot([point[0] for point in sclusters[1]], [point[1] for point in sclusters[1]], 'mo', label='Cluster 1')
plt.plot(smeans[1][0], smeans[1][1], 'ro', label='Centroid 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization for d = 2')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,2)
plt.plot([point[0] for point in sclusters_3[0]], [point[1] for point in sclusters_3[0]], 'co', label='Cluster 0')
plt.plot(smeans_3[0][0], smeans_3[0][1], 'yo', label='Centroid 0')
plt.plot([point[0] for point in sclusters_3[1]], [point[1] for point in sclusters_3[1]], 'mo', label='Cluster 1')
plt.plot(smeans_3[1][0], smeans_3[1][1], 'ro', label='Centroid 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization for d = 3')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,3)
plt.plot([point[0] for point in sclusters_4[0]], [point[1] for point in sclusters_4[0]], 'co', label='Cluster 0')
plt.plot(smeans_4[0][0], smeans_4[0][1], 'yo', label='Centroid 0')
plt.plot([point[0] for point in sclusters_4[1]], [point[1] for point in sclusters_4[1]], 'mo', label='Cluster 1')
plt.plot(smeans_4[1][0], smeans_4[1][1], 'ro', label='Centroid 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization for d = 4')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,4)
plt.plot([point[0] for point in sclusters_5[0]], [point[1] for point in sclusters_5[0]], 'co', label='Cluster 0')
plt.plot(smeans_5[0][0], smeans_5[0][1], 'yo', label='Centroid 0')
plt.plot([point[0] for point in sclusters_5[1]], [point[1] for point in sclusters_5[1]], 'mo', label='Cluster 1')
plt.plot(smeans_5[1][0], smeans_5[1][1], 'ro', label='Centroid 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization for d = 5')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# plotting original dataset with above clustering

plt.figure(figsize=(25, 5))
plt.suptitle('Original Dataset Clusters, Spectral Clustering, Polynomial Kernel', fontsize=16)
ax = plt.subplot(1,4,1)
plt.plot([point[0] for point in og_cluster_1[0]], [point[1] for point in og_cluster_1[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in og_cluster_1[1]], [point[1] for point in og_cluster_1[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Cluster Visualization for d = 2')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,2)
plt.plot([point[0] for point in og_cluster_2[0]], [point[1] for point in og_cluster_2[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in og_cluster_2[1]], [point[1] for point in og_cluster_2[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Cluster Visualization for d = 3')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,3)
plt.plot([point[0] for point in og_cluster_3[0]], [point[1] for point in og_cluster_3[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in og_cluster_3[1]], [point[1] for point in og_cluster_3[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Cluster Visualization for d = 4')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,4)
plt.plot([point[0] for point in og_cluster_4[0]], [point[1] for point in og_cluster_4[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in og_cluster_4[1]], [point[1] for point in og_cluster_4[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Cluster Visualization for d = 5')
plt.legend()
plt.grid(True)
plt.show()

# radial basis kernel

kernel_r_05 = radial_kernel_calculator(dataset, 0.5)
eigenvalues_r_05, eigenvectors_r_05 = np.linalg.eigh(kernel_r_05)
w_1_r_05 = eigenvectors_r_05[:, -1].transpose()
w_2_r_05 = eigenvectors_r_05[:, -2].transpose()
H_r_05 = np.column_stack((w_1_r_05, w_2_r_05))
H_r_05 = H_r_05/np.linalg.norm(H_r_05, axis=1, keepdims=True)
K = []
K.append(H_r_05[0].tolist())
spectral_dataset_r_05 = []
for i in H_r_05:
    spectral_dataset_r_05.append(i.tolist())
initial_mean_r_05 = random.sample(spectral_dataset_r_05, 2)
sclusters_r_05, smeans_r_05, serror_r_05 = k_means_clustering(initial_mean_r_05, spectral_dataset_r_05, 2) 

kernel_r_1 = radial_kernel_calculator(dataset, 1)
eigenvalues_r_1, eigenvectors_r_1 = np.linalg.eigh(kernel_r_1)
w_1_r_1 = eigenvectors_r_1[:, -1].transpose()
w_2_r_1 = eigenvectors_r_1[:, -2].transpose()
H_r_1 = np.column_stack((w_1_r_1, w_2_r_1))
H_r_1 = H_r_1 / np.linalg.norm(H_r_1, axis=1, keepdims=True)
K = []
K.append(H_r_1[0].tolist())
spectral_dataset_r_1 = []
for i in H_r_1:
    spectral_dataset_r_1.append(i.tolist())
initial_mean_r_1 = random.sample(spectral_dataset_r_1, 2)
sclusters_r_1, smeans_r_1, serror_r_1 = k_means_clustering(initial_mean_r_1, spectral_dataset_r_1, 2)

kernel_r_1000 = radial_kernel_calculator(dataset, 1000)
eigenvalues_r_1000, eigenvectors_r_1000 = np.linalg.eigh(kernel_r_1000)
w_1_r_1000 = eigenvectors_r_1000[:, -1].transpose()
w_2_r_1000 = eigenvectors_r_1000[:, -2].transpose()
H_r_1000 = np.column_stack((w_1_r_1000, w_2_r_1000))
H_r_1000 = H_r_1000/np.linalg.norm(H_r_1000, axis=1, keepdims=True)
K = []
K.append(H_r_1000[0].tolist())
spectral_dataset_r_1000 = []
for i in H_r_1000:
    spectral_dataset_r_1000.append(i.tolist())
initial_mean_r_1000 = random.sample(spectral_dataset_r_1000, 2)
sclusters_r_1000, smeans_r_1000, serror_r_1000 = k_means_clustering(initial_mean_r_1000, spectral_dataset_r_1000, 2) 

kernel_r_10000 = radial_kernel_calculator(dataset, 10000)
eigenvalues_r_10000, eigenvectors_r_10000 = np.linalg.eigh(kernel_r_10000)
w_1_r_10000 = eigenvectors_r_10000[:, -1].transpose()
w_2_r_10000 = eigenvectors_r_10000[:, -2].transpose()
H_r_10000 = np.column_stack((w_1_r_10000, w_2_r_10000))
H_r_10000 = H_r_10000/np.linalg.norm(H_r_10000, axis=1, keepdims=True)
K = []
K.append(H_r_10000[0].tolist())
spectral_dataset_r_10000 = []
for i in H_r_10000:
    spectral_dataset_r_10000.append(i.tolist())
initial_mean_r_10000 = random.sample(spectral_dataset_r_10000, 2)
sclusters_r_10000, smeans_r_10000, serror_r_10000 = k_means_clustering(initial_mean_r_10000, spectral_dataset_r_10000, 2) 

og_cluster_05 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_r_05)):
    if spectral_dataset_r_05[i] in sclusters_r_05[0]:
        og_cluster_05[0].append(dataset[i])
    else:
        og_cluster_05[1].append(dataset[i])

og_cluster_1 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_r_1)):
    if spectral_dataset_r_1[i] in sclusters_r_1[0]:
        og_cluster_1[0].append(dataset[i])
    else:
        og_cluster_1[1].append(dataset[i])

og_cluster_1000 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_r_1000)):
    if spectral_dataset_r_1000[i] in sclusters_r_1000[0]:
        og_cluster_1000[0].append(dataset[i])
    else:
        og_cluster_1000[1].append(dataset[i])

og_cluster_10000 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_r_10000)):
    if spectral_dataset_r_10000[i] in sclusters_r_10000[0]:
        og_cluster_10000[0].append(dataset[i])
    else:
        og_cluster_10000[1].append(dataset[i])

plt.figure(figsize=(20, 5))
plt.suptitle('Spectral Clustering, Radial Basis Kernel', fontsize=16)
ax = plt.subplot(1,4,1)
plt.plot([point[0] for point in sclusters_r_05[0]], [point[1] for point in sclusters_r_05[0]], 'co', label='Cluster 0')
plt.plot(smeans_r_05[0][0], smeans_r_05[0][1], 'yo', label='Centroid 0')
plt.plot([point[0] for point in sclusters_r_05[1]], [point[1] for point in sclusters_r_05[1]], 'mo', label='Cluster 1')
plt.plot(smeans_r_05[1][0], smeans_r_05[1][1], 'ro', label='Centroid 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization for $\\sigma$ = 0.5')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,2)
plt.plot([point[0] for point in sclusters_r_1[0]], [point[1] for point in sclusters_r_1[0]], 'co', label='Cluster 0')
plt.plot(smeans_r_1[0][0], smeans_r_1[0][1], 'yo', label='Centroid 0')
plt.plot([point[0] for point in sclusters_r_1[1]], [point[1] for point in sclusters_r_1[1]], 'mo', label='Cluster 1')
plt.plot(smeans_r_1[1][0], smeans_r_1[1][1], 'ro', label='Centroid 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization for $\\sigma$ = 1')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,3)
plt.plot([point[0] for point in sclusters_r_1000[0]], [point[1] for point in sclusters_r_1000[0]], 'co', label='Cluster 0')
plt.plot(smeans_r_1000[0][0], smeans_r_1000[0][1], 'yo', label='Centroid 0')
plt.plot([point[0] for point in sclusters_r_1000[1]], [point[1] for point in sclusters_r_1000[1]], 'mo', label='Cluster 1')
plt.plot(smeans_r_1000[1][0], smeans_r_1000[1][1], 'ro', label='Centroid 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization for $\\sigma$ = 1000')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,4)
plt.plot([point[0] for point in sclusters_r_10000[0]], [point[1] for point in sclusters_r_10000[0]], 'co', label='Cluster 0')
plt.plot(smeans_r_10000[0][0], smeans_r_10000[0][1], 'yo', label='Centroid 0')
plt.plot([point[0] for point in sclusters_r_10000[1]], [point[1] for point in sclusters_r_10000[1]], 'mo', label='Cluster 1')
plt.plot(smeans_r_10000[1][0], smeans_r_10000[1][1], 'ro', label='Centroid 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization for $\\sigma$ = 10000')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(25, 5))
plt.suptitle('Original Dataset Clusters, Spectral Clustering, Radial Basis Kernel', fontsize=16)
ax = plt.subplot(1,4,1)
plt.plot([point[0] for point in og_cluster_05[0]], [point[1] for point in og_cluster_05[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in og_cluster_05[1]], [point[1] for point in og_cluster_05[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Cluster Visualization for $\\sigma$ = 0.5')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,2)
plt.plot([point[0] for point in og_cluster_1[0]], [point[1] for point in og_cluster_1[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in og_cluster_1[1]], [point[1] for point in og_cluster_1[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Cluster Visualization for $\\sigma$ = 1')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,3)
plt.plot([point[0] for point in og_cluster_1000[0]], [point[1] for point in og_cluster_1000[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in og_cluster_1000[1]], [point[1] for point in og_cluster_1000[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Cluster Visualization for $\\sigma$ = 1000')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,4)
plt.plot([point[0] for point in og_cluster_10000[0]], [point[1] for point in og_cluster_10000[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in og_cluster_10000[1]], [point[1] for point in og_cluster_10000[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Cluster Visualization for $\\sigma$ = 10000')
plt.legend()
plt.grid(True)
plt.show()

kernel_r_100000 = radial_kernel_calculator(dataset, 100000000000000)
eigenvalues_r_100000, eigenvectors_r_100000 = np.linalg.eigh(kernel_r_100000)
w_1_r_100000 = eigenvectors_r_100000[:, -1].transpose()
w_2_r_100000 = eigenvectors_r_100000[:, -2].transpose()
H_r_100000 = np.column_stack((w_1_r_100000, w_2_r_100000))
H_r_100000 = H_r_100000/np.linalg.norm(H_r_100000, axis=1, keepdims=True)
K = []
K.append(H_r_100000[0].tolist())
spectral_dataset_r_100000 = []
for i in H_r_100000:
    spectral_dataset_r_100000.append(i.tolist())
initial_mean_r_100000 = random.sample(spectral_dataset_r_100000, 2)
sclusters_r_100000, smeans_r_100000, serror_r_100000 = k_means_clustering(initial_mean_r_100000, spectral_dataset_r_100000, 2) 

og_cluster_100000 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_r_100000)):
    if spectral_dataset_r_100000[i] in sclusters_r_100000[0]:
        og_cluster_100000[0].append(dataset[i])
    else:
        og_cluster_100000[1].append(dataset[i])

plt.figure(figsize=(20, 5))
plt.plot([point[0] for point in sclusters_r_100000[0]], [point[1] for point in sclusters_r_100000[0]], 'co', label='Cluster 0')
plt.plot(smeans_r_100000[0][0], smeans_r_100000[0][1], 'yo', label='Centroid 0')
plt.plot([point[0] for point in sclusters_r_100000[1]], [point[1] for point in sclusters_r_100000[1]], 'mo', label='Cluster 1')
plt.plot(smeans_r_100000[1][0], smeans_r_100000[1][1], 'ro', label='Centroid 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization for $\\sigma$ = 1e14')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(25, 5))
plt.plot([point[0] for point in og_cluster_100000[0]], [point[1] for point in og_cluster_100000[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in og_cluster_100000[1]], [point[1] for point in og_cluster_100000[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Cluster Visualization for $\\sigma$ = 1e14')
plt.legend()
plt.grid(True)
plt.show()

# part iv

# putting the first datapoint in the first cluster if y coordinate is greater than x coordinate in H

alt_ssclusters = [[] for _ in range(2)]
for i in range(len(spectral_dataset)):
    if spectral_dataset[i][0] > spectral_dataset[i][1]:
        alt_ssclusters[0].append(spectral_dataset[i])
    else:
        alt_ssclusters[1].append(spectral_dataset[i])

alt_ssclusters_3 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_3)):
    if spectral_dataset_3[i][0] > spectral_dataset_3[i][1]:
        alt_ssclusters_3[0].append(spectral_dataset_3[i])
    else:
        alt_ssclusters_3[1].append(spectral_dataset_3[i])

# plotting based on alternative spectral clustering

plt.figure(figsize=(25, 7))
plt.suptitle('Clusters for alternative Spectral Clustering, Polynomial Kernel', fontsize=16)
ax = plt.subplot(1,2,1)
plt.plot([point[0] for point in alt_ssclusters[0]], [point[1] for point in alt_ssclusters[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in alt_ssclusters[1]], [point[1] for point in alt_ssclusters[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization for alternative K = 2')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,2,2)
plt.plot([point[0] for point in alt_ssclusters_3[0]], [point[1] for point in alt_ssclusters_3[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in alt_ssclusters_3[1]], [point[1] for point in alt_ssclusters_3[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization for alternative K = 3')
plt.legend()
plt.grid(True)
plt.show()

# finding clusters for og dataset based on above clustering

alt_sclusters = [[] for _ in range(2)]
for i in range(len(spectral_dataset)):
    if spectral_dataset[i][0] > spectral_dataset[i][1]:
        alt_sclusters[0].append(dataset[i])
    else:
        alt_sclusters[1].append(dataset[i])

alt_sclusters_3 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_3)):
    if spectral_dataset_3[i][0] > spectral_dataset_3[i][1]:
        alt_sclusters_3[0].append(dataset[i])
    else:
        alt_sclusters_3[1].append(dataset[i])

# plotting above clusters
        
plt.figure(figsize=(25, 7))
plt.suptitle('Original Clusters for alternative Spectral Clustering, Polynomial Kernel', fontsize=16)
ax = plt.subplot(1,2,1)
plt.plot([point[0] for point in alt_sclusters[0]], [point[1] for point in alt_sclusters[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in alt_sclusters[1]], [point[1] for point in alt_sclusters[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Cluster Visualization for alternative K = 2')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,2,2)
plt.plot([point[0] for point in alt_sclusters_3[0]], [point[1] for point in alt_sclusters_3[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in alt_sclusters_3[1]], [point[1] for point in alt_sclusters_3[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Cluster Visualization for alternative K = 3')
plt.legend()
plt.grid(True)
plt.show()

# doing the same for radial basis kernel

alt_scluster_05 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_r_05)):
    if spectral_dataset_r_05[i][0] > spectral_dataset_r_05[i][1]:
        alt_scluster_05[0].append(spectral_dataset_r_05[i])
    else:
        alt_scluster_05[1].append(spectral_dataset_r_05[i])

alt_scluster_1 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_r_1)):
    if spectral_dataset_r_1[i][0] > spectral_dataset_r_1[i][1]:
        alt_scluster_1[0].append(spectral_dataset_r_1[i])
    else:
        alt_scluster_1[1].append(spectral_dataset_r_1[i])

alt_scluster_1000 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_r_1000)):
    if spectral_dataset_r_1000[i][0] > spectral_dataset_r_1000[i][1]:
        alt_scluster_1000[0].append(spectral_dataset_r_1000[i])
    else:
        alt_scluster_1000[1].append(spectral_dataset_r_1000[i])

alt_scluster_10000 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_r_10000)):
    if spectral_dataset_r_10000[i][0] > spectral_dataset_r_10000[i][1]:
        alt_scluster_10000[0].append(spectral_dataset_r_10000[i])
    else:
        alt_scluster_10000[1].append(spectral_dataset_r_10000[i])

plt.figure(figsize=(25, 5))
plt.suptitle('Clusters for alternative Spectral Clustering, Radial Basis Kernel', fontsize=16)

ax = plt.subplot(1,4,1)
plt.plot([point[0] for point in alt_scluster_05[0]], [point[1] for point in alt_scluster_05[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in alt_scluster_05[1]], [point[1] for point in alt_scluster_05[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization for alternative $\\sigma$ = 0.5')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,2)
plt.plot([point[0] for point in alt_scluster_1[0]], [point[1] for point in alt_scluster_1[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in alt_scluster_1[1]], [point[1] for point in alt_scluster_1[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization for alternative $\\sigma$ = 1')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,3)
plt.plot([point[0] for point in alt_scluster_1000[0]], [point[1] for point in alt_scluster_1000[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in alt_scluster_1000[1]], [point[1] for point in alt_scluster_1000[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization for alternative $\\sigma$ = 1000')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,4)
plt.plot([point[0] for point in alt_scluster_10000[0]], [point[1] for point in alt_scluster_10000[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in alt_scluster_10000[1]], [point[1] for point in alt_scluster_10000[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Visualization for alternative $\\sigma$ = 10000')
plt.legend()
plt.grid(True)
plt.show()

alt_cluster_05 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_r_05)):
    if spectral_dataset_r_05[i][0] > spectral_dataset_r_05[i][1]:
        alt_cluster_05[0].append(dataset[i])
    else:
        alt_cluster_05[1].append(dataset[i])

alt_cluster_1 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_r_1)):
    if spectral_dataset_r_1[i][0] > spectral_dataset_r_1[i][1]:
        alt_cluster_1[0].append(dataset[i])
    else:
        alt_cluster_1[1].append(dataset[i])

alt_cluster_1000 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_r_1000)):
    if spectral_dataset_r_1000[i][0] > spectral_dataset_r_1000[i][1]:
        alt_cluster_1000[0].append(dataset[i])
    else:
        alt_cluster_1000[1].append(dataset[i])

alt_cluster_10000 = [[] for _ in range(2)]
for i in range(len(spectral_dataset_r_10000)):
    if spectral_dataset_r_10000[i][0] > spectral_dataset_r_10000[i][1]:
        alt_cluster_10000[0].append(dataset[i])
    else:
        alt_cluster_10000[1].append(dataset[i])

plt.figure(figsize=(25, 5))
plt.suptitle('Original Clusters for alternative Spectral Clustering, Radial Basis Kernel', fontsize=16)

ax = plt.subplot(1,4,1)
plt.plot([point[0] for point in alt_cluster_05[0]], [point[1] for point in alt_cluster_05[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in alt_cluster_05[1]], [point[1] for point in alt_cluster_05[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Cluster Visualization for alternative $\\sigma$ = 0.5')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,2)
plt.plot([point[0] for point in alt_cluster_1[0]], [point[1] for point in alt_cluster_1[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in alt_cluster_1[1]], [point[1] for point in alt_cluster_1[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Cluster Visualization for alternative $\\sigma$ = 1')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,3)
plt.plot([point[0] for point in alt_cluster_1000[0]], [point[1] for point in alt_cluster_1000[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in alt_cluster_1000[1]], [point[1] for point in alt_cluster_1000[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Cluster Visualization for alternative $\\sigma$ = 1000')
plt.legend()
plt.grid(True)

ax = plt.subplot(1,4,4)
plt.plot([point[0] for point in alt_cluster_10000[0]], [point[1] for point in alt_cluster_10000[0]], 'co', label='Cluster 0')
plt.plot([point[0] for point in alt_cluster_10000[1]], [point[1] for point in alt_cluster_10000[1]], 'mo', label='Cluster 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Cluster Visualization for alternative $\\sigma$ = 10000')
plt.legend()
plt.grid(True)
plt.show()