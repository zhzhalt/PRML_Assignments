# loading the dataset. here, i combined the train and test datasets into one dataset of 70,000 images.
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset, concatenate_datasets
train_dataset = load_dataset("mnist", split="train")
test_dataset = load_dataset("mnist", split="test")
dataset = concatenate_datasets([train_dataset,test_dataset])

# divide the dataset based on the labels (digit representing the image)
data_per_label = [[] for _ in range(10)]
zeroes = 0
k = 0
for i in range(len(dataset)):
    data_per_label[dataset[i]['label']].append(dataset[i])

# from each label, take a 100 random data points
data = []
for i in data_per_label:
    data = data + random.sample(i, 100)
print(len(data))

# ravel takes the 28x28 image and converts it into a one-dimensional array
image_data = []
for i in data:
    image_data.append(np.ravel(i['image']))
print(len(image_data))

# centering the data and finding covariance matrix
mean = np.mean(image_data, axis=0)
centered_image_data = image_data - mean
centered_image_data = np.array(centered_image_data)
covariance = 0.001 * (np.cov(centered_image_data.transpose()))

# obtaining eigenvectors and eigenvalues and sorting them
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
index = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[index]
eigenvectors = eigenvectors[:, index]
eigenvectors = eigenvectors.transpose()

# finding contribution of each eigenvector to variance
sum_of_eigenvalues = 0
for i in eigenvalues:
    if i>0:
        sum_of_eigenvalues += i
ninetyfive = 95*sum_of_eigenvalues/100
contribution_to_variance = []
ninety_five_max = 0
sum = 0
for i in eigenvalues:
    if i>0:
        contribution_to_variance.append(i/sum_of_eigenvalues)
        sum += i
        if (sum < ninetyfive):
            ninety_five_max += 1
    else:
        contribution_to_variance.append(0)
contribution_to_variance = np.array(contribution_to_variance)

# plot the contr. to var. graph
x = np.arange(1,785)
y = contribution_to_variance
fig, ax = plt.subplots(figsize=(9,5))
ax.bar(x, y, width=1.0)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Contribution to Variance')
ax.axvline(x=ninety_five_max, color='red', linestyle='--')
plt.title('Contribution of principal components to variance')
plt.text(0.95, 0.95, 'Red line: Point of 95% of total variance', fontsize=10, ha='right', va='top', transform=plt.gca().transAxes)
plt.show()

# print the principal components
components = 25
rows = 5
columns = 5
plt.figure(figsize=(13,10))
for i in range(components):
    ax = plt.subplot(rows, columns, i+1)
    plt.imshow(eigenvectors[i].reshape(28,28))
    ax.set_xlabel("Dimensions", fontsize=8)
    ax.set_ylabel("Dimensions", fontsize=8)
    plt.title('Principal Component '+str(i+1), fontsize=8)
    plt.text(0.5, -.660, 'Contribution to variance = '+str(format(contribution_to_variance[i], '.4f')), fontsize=8.5, ha='center', transform=plt.gca().transAxes)
plt.subplots_adjust(wspace=0.75, hspace=1.2)
plt.show()

plt.figure(figsize=(13,10))
for i in range(25,50):
    ax = plt.subplot(rows, columns, i-24)
    plt.imshow(eigenvectors[i].reshape(28,28))
    ax.set_xlabel("Dimensions", fontsize=8)
    ax.set_ylabel("Dimensions", fontsize=8)
    plt.title('Principal Component '+str(i+1), fontsize=8)
    plt.text(0.5, -.660, 'Contribution to variance = '+str(format(contribution_to_variance[i], '.4f')), fontsize=8.5, ha='center', transform=plt.gca().transAxes)
plt.subplots_adjust(wspace=0.75, hspace=1.2)
plt.show()

plt.figure(figsize=(13,10))
for i in range(125,150):
    ax = plt.subplot(rows, columns, i-124)
    plt.imshow(eigenvectors[i].reshape(28,28))
    ax.set_xlabel("Dimensions", fontsize=8)
    ax.set_ylabel("Dimensions", fontsize=8)
    plt.title('Principal Component '+str(i+1), fontsize=8)
    plt.text(0.5, -.660, 'Contribution to variance = '+str(format(contribution_to_variance[i], '.4f')), fontsize=8.5, ha='center', transform=plt.gca().transAxes)

plt.subplots_adjust(wspace=0.75, hspace=1.2)
plt.show()

# part ii

# function that gives the 'd' dimensional representation of the data
def d_dimensional_rep(orig_data, d):
    data_rep = []
    for i in range(len(orig_data)):
        i_th_datapoint = np.empty(784)
        for j in range(d):
            rep = np.dot(orig_data[i], eigenvectors[j]) * eigenvectors[j]
            i_th_datapoint += rep
        data_rep.append(i_th_datapoint)
    return data_rep

ten_dimensional_rep = np.array(d_dimensional_rep(centered_image_data, 10))
twentyfive_dimensional_rep = np.array(d_dimensional_rep(centered_image_data, 25))
fifty_dimensional_rep = np.array(d_dimensional_rep(centered_image_data, 50))
seventyfive_dimensional_rep = np.array(d_dimensional_rep(centered_image_data, 75))
hun_dimensional_rep = np.array(d_dimensional_rep(centered_image_data, 100))
onethirty_dimensional_rep = np.array(d_dimensional_rep(centered_image_data, 130))
onefifty_dimensional_rep = np.array(d_dimensional_rep(centered_image_data, 150))

# plotting different representations of the data

dimensional_reps = [ten_dimensional_rep, twentyfive_dimensional_rep, fifty_dimensional_rep, seventyfive_dimensional_rep, hun_dimensional_rep, onethirty_dimensional_rep, onefifty_dimensional_rep]
dimension = [10, 25, 50, 75, 100, 130, 150]
indices = [48, 148, 248, 348, 448, 548, 648, 748, 878, 948]
plt.figure(figsize=(15, 15))
for i in range(10):
    for j in range(7):
        rep = dimensional_reps[j]
        plt.subplot(10, 7, i*7 + j + 1)
        plt.imshow(rep[indices[i]].reshape(28, 28), cmap='gray')
        plt.title('d = {}'.format(dimension[j]))
        plt.axis('off')
plt.suptitle('Different Dimensional Representations of the data', fontsize=13)
plt.tight_layout()
plt.show()

# plotting different datapoints using d = point of 95% of variance

dimensional_reps = onethirty_dimensional_rep
dimension = [10, 25, 50, 75, 100, 130, 150]
indices = random.sample(range(1000), 25)

plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, 25-i)
    plt.imshow(dimensional_reps[indices[i]].reshape(28, 28), cmap='gray')
    plt.title('d = 130')
    plt.axis('off')

plt.suptitle('Different Dimensional Representations of the data', fontsize=13)
plt.tight_layout()
plt.show()

# part iii

# kernel calculator for polynomial kernel

def kernel_calculator(orig_data, d):
    kernel = np.zeros((len(orig_data), len(orig_data)))
    for i in range(len(orig_data)):
        for j in range(len(orig_data)):
            kernel[i, j] = (np.dot(orig_data[i], orig_data[j]) + 1)**d
    return kernel

# function for centering the kernel

def centering(kernel):    
    n = len(kernel)
    centered_kernel = np.zeros((n,n))
    mean_k_i = np.mean(kernel, axis=1)
    mean_k_j = np.mean(kernel, axis=0)
    tot_mean = np.mean(kernel)
    for i in range(len(kernel)):
        for j in range(len(kernel)):
            centered_kernel[i][j] = kernel[i][j] - mean_k_i[i] - mean_k_j[j] + tot_mean
    return centered_kernel

kernel_image_data = centered_image_data

twodim = kernel_calculator(kernel_image_data, 2)
threedim = kernel_calculator(kernel_image_data, 3)
fourdim = kernel_calculator(kernel_image_data, 4)

twodim_kernel = centering(twodim)
threedim_kernel = centering(threedim)
fourdim_kernel = centering(fourdim)

# finding the eigenvalues and eigenvectors, taking the positive ones, sorting

kernel_eigenvalues, kernel_eigenvectors = np.linalg.eigh(twodim_kernel)
kernel_eigenvectors = kernel_eigenvectors.transpose()

positive_eigenvalues = []
positive_eigenvectors = []
for i in range(len(kernel_eigenvalues)):
    if (kernel_eigenvalues[i] > 0):
        positive_eigenvalues.append(kernel_eigenvalues[i])
        positive_eigenvectors.append(kernel_eigenvectors[i])
kernel_eigenvalues = np.array(positive_eigenvalues)
kernel_eigenvectors = np.array(positive_eigenvectors)

index = kernel_eigenvalues.argsort()[::-1]
kernel_eigenvalues = kernel_eigenvalues[index]
kernel_eigenvectors = kernel_eigenvectors[index]

# finding the weights that are given to represent the eigenvector as a linear combination of the original datapoints

alpha_1 = kernel_eigenvectors[0]/math.sqrt(kernel_eigenvalues[0])
alpha_2 = kernel_eigenvectors[1]/math.sqrt(kernel_eigenvalues[1])

# finding the projections

projections_1 = []
for i in range(len(kernel_image_data)):
    projection = 0
    for j in range(len(kernel_image_data)):
        projection += alpha_1[j] * twodim_kernel[i][j]
    projections_1.append(projection)

projections_2 = []
for i in range(len(kernel_image_data)):
    projection = 0
    for j in range(len(kernel_image_data)):
        projection += alpha_2[j] * twodim_kernel[i][j]
    projections_2.append(projection)

kernel_eigenvalues_3, kernel_eigenvectors_3 = np.linalg.eigh(threedim_kernel)

kernel_eigenvectors_3 = kernel_eigenvectors_3.transpose()
positive_eigenvalues = []
positive_eigenvectors = []
for i in range(len(kernel_eigenvalues_3)):
    if (kernel_eigenvalues_3[i] > 0):
        positive_eigenvalues.append(kernel_eigenvalues_3[i])
        positive_eigenvectors.append(kernel_eigenvectors_3[i])
        

kernel_eigenvalues_3 = np.array(positive_eigenvalues)
kernel_eigenvectors_3 = np.array(positive_eigenvectors)

index = kernel_eigenvalues_3.argsort()[::-1]
kernel_eigenvalues_3 = kernel_eigenvalues_3[index]
kernel_eigenvectors_3 = kernel_eigenvectors_3[index]

alpha_1_3 = kernel_eigenvectors_3[0]/math.sqrt(kernel_eigenvalues_3[0])
alpha_2_3 = kernel_eigenvectors_3[1]/math.sqrt(kernel_eigenvalues_3[1])

projections_1_3 = []
for i in range(len(kernel_image_data)):
    projection = 0
    for j in range(len(kernel_image_data)):
        projection += alpha_1_3[j] * threedim_kernel[i][j]
    projections_1_3.append(projection)

projections_2_3 = []
for i in range(len(kernel_image_data)):
    projection = 0
    for j in range(len(kernel_image_data)):
        projection += alpha_2_3[j] * threedim_kernel[i][j]
    projections_2_3.append(projection)


kernel_eigenvalues_4, kernel_eigenvectors_4 = np.linalg.eigh(fourdim_kernel)

kernel_eigenvectors_4 = kernel_eigenvectors_4.transpose()
positive_eigenvalues = []
positive_eigenvectors = []
for i in range(len(kernel_eigenvalues_4)):
    if (kernel_eigenvalues_4[i] > 0):
        positive_eigenvalues.append(kernel_eigenvalues_4[i])
        positive_eigenvectors.append(kernel_eigenvectors_4[i])
        

kernel_eigenvalues_4 = np.array(positive_eigenvalues)
kernel_eigenvectors_4 = np.array(positive_eigenvectors)

index = kernel_eigenvalues_4.argsort()[::-1]
kernel_eigenvalues_4 = kernel_eigenvalues_4[index]
kernel_eigenvectors_4 = kernel_eigenvectors_4[index]

alpha_1_4 = kernel_eigenvectors_4[0]/math.sqrt(kernel_eigenvalues_4[0])
alpha_2_4 = kernel_eigenvectors_4[1]/math.sqrt(kernel_eigenvalues_4[1])

projections_1_4 = []
for i in range(len(kernel_image_data)):
    projection = 0
    for j in range(len(kernel_image_data)):
        projection += alpha_1_4[j] * fourdim_kernel[i][j]
    projections_1_4.append(projection)

projections_2_4 = []
for i in range(len(kernel_image_data)):
    projection = 0
    for j in range(len(kernel_image_data)):
        projection += alpha_2_4[j] * fourdim_kernel[i][j]
    projections_2_4.append(projection)

# plotting the projections
    
plt.figure(figsize=(15,5))

ax = plt.subplot(1,3,1)
plt.scatter(projections_1, projections_2)
plt.xlabel('Projection on first principal component')
plt.ylabel('Projection on second principal component')
plt.title('Projections of datapoints on first 2 principal components', fontsize=10, pad=25)
plt.text(0.65, 1.05, 'Polynomial kernel, d = 2', fontsize=9.5, ha='right', va='top', transform=plt.gca().transAxes) 
plt.grid()

ax = plt.subplot(1,3,2)
plt.scatter(projections_1_3, projections_2_3)
plt.xlabel('Projection on first principal component')
plt.ylabel('Projection on second principal component')
plt.title('Projections of datapoints on first 2 principal components', fontsize=10, pad=25)
plt.text(0.65, 1.05, 'Polynomial kernel, d = 3', fontsize=9.5, ha='right', va='top', transform=plt.gca().transAxes) 
plt.grid()

ax = plt.subplot(1,3,3)
plt.scatter(projections_1_4, projections_2_4)
plt.xlabel('Projection on first principal component')
plt.ylabel('Projection on second principal component')
plt.title('Projections of datapoints on first 2 principal components', fontsize=10, pad=25)
plt.text(0.65, 1.05, 'Polynomial kernel, d = 4', fontsize=9.5, ha='right', va='top', transform=plt.gca().transAxes) 
plt.grid()

plt.tight_layout()
plt.show()

# radial basis kernel

def radial_kernel_calculator(orig_data, sigma):
    kernel = np.zeros((len(orig_data), len(orig_data)))
    for i in range(len(orig_data)):
        for j in range(len(orig_data)):
            kernel[i, j] = np.exp(-(np.dot(orig_data[i] - orig_data[j] , orig_data[i] - orig_data[j])) / (2 * sigma * sigma))
    return kernel

# finding the kernel and centering

sigma_10 = radial_kernel_calculator(kernel_image_data, 10)
centered_sigma_10 = centering(sigma_10)

# finding the eigenvectors and eigenvalues and sorting

k_eigenvalues_10, k_eigenvectors_10 = np.linalg.eigh(centered_sigma_10)
k_eigenvectors_10 = k_eigenvectors_10.transpose()
positive_eigenvalues = []
positive_eigenvectors = []
for i in range(len(k_eigenvalues_10)):
    if (k_eigenvalues_10[i] > 0):
        positive_eigenvalues.append(k_eigenvalues_10[i])
        positive_eigenvectors.append(k_eigenvectors_10[i])
        

k_eigenvalues_10 = np.array(positive_eigenvalues)
k_eigenvectors_10 = np.array(positive_eigenvectors)

index = k_eigenvalues_10.argsort()[::-1]
k_eigenvalues_10 = k_eigenvalues_10[index]
k_eigenvectors_10 = k_eigenvectors_10[index]

# finding the weights that are given to represent the eigenvector as a linear combination of the original datapoints

a_10_1 = k_eigenvectors_10[0]/math.sqrt(k_eigenvalues_10[0])
a_10_2 = k_eigenvectors_10[1]/math.sqrt(k_eigenvalues_10[1])

# finding the projections

p_1_10 = []
for i in range(len(kernel_image_data)):
    projection = 0
    for j in range(len(kernel_image_data)):
        projection += a_10_1[j] * centered_sigma_10[i][j]
    p_1_10.append(projection)

p_2_10 = []
for i in range(len(kernel_image_data)):
    projection = 0
    for j in range(len(kernel_image_data)):
        projection += a_10_2[j] * centered_sigma_10[i][j]
    p_2_10.append(projection)

# plotting the projections
    
fig = plt.figure(figsize=(6,6))
ax = plt.scatter(p_1_10, p_2_10)
plt.xlabel('Projection on first principal component')
plt.ylabel('Projection on second principal component')
plt.title('Projections of datapoints on first 2 principal components', fontsize=10, pad=25)
plt.text(0.68, 1.05, 'Radial basis kernel, $\\sigma$ = 10', fontsize=9.5, ha='right', va='top', transform=plt.gca().transAxes) 
plt.grid()
plt.show()

sigma_100 = radial_kernel_calculator(kernel_image_data, 150)
centered_sigma_100 = centering(sigma_100)
k_eigenvalues_100, k_eigenvectors_100 = np.linalg.eigh(centered_sigma_100)
k_eigenvectors_100 = k_eigenvectors_100.transpose()
positive_eigenvalues = []
positive_eigenvectors = []
for i in range(len(k_eigenvalues_100)):
    if (k_eigenvalues_100[i] > 0):
        positive_eigenvalues.append(k_eigenvalues_100[i])
        positive_eigenvectors.append(k_eigenvectors_100[i])
        

k_eigenvalues_100 = np.array(positive_eigenvalues)
k_eigenvectors_100 = np.array(positive_eigenvectors)

index = k_eigenvalues_100.argsort()[::-1]
k_eigenvalues_100 = k_eigenvalues_100[index]
k_eigenvectors_100 = k_eigenvectors_100[index]

a_100_1 = k_eigenvectors_100[0]/math.sqrt(k_eigenvalues_100[0])
a_100_2 = k_eigenvectors_100[1]/math.sqrt(k_eigenvalues_100[1])

p_1_100 = []
for i in range(len(kernel_image_data)):
    projection = 0
    for j in range(len(kernel_image_data)):
        projection += a_100_1[j] * centered_sigma_100[i][j]
    p_1_100.append(projection)

p_2_100 = []
for i in range(len(kernel_image_data)):
    projection = 0
    for j in range(len(kernel_image_data)):
        projection += a_100_2[j] * centered_sigma_100[i][j]
    p_2_100.append(projection)

fig = plt.figure(figsize=(6,6))
ax = plt.scatter(p_1_100, p_2_100)
plt.xlabel('Projection on first principal component')
plt.ylabel('Projection on second principal component')
plt.title('Projections of datapoints on first 2 principal components', fontsize=10, pad=25)
plt.text(0.68, 1.05, 'Radial basis kernel, $\\sigma$ = 150', fontsize=9.5, ha='right', va='top', transform=plt.gca().transAxes) 
plt.grid()
plt.show()

sigma_1000 = radial_kernel_calculator(kernel_image_data, 1000)
centered_sigma_1000 = centering(sigma_1000)
k_eigenvalues_1000, k_eigenvectors_1000 = np.linalg.eigh(centered_sigma_1000)
k_eigenvectors_1000 = k_eigenvectors_1000.transpose()
positive_eigenvalues = []
positive_eigenvectors = []
for i in range(len(k_eigenvalues_1000)):
    if (k_eigenvalues_1000[i] > 0):
        positive_eigenvalues.append(k_eigenvalues_1000[i])
        positive_eigenvectors.append(k_eigenvectors_1000[i])
        

k_eigenvalues_1000 = np.array(positive_eigenvalues)
k_eigenvectors_1000 = np.array(positive_eigenvectors)

index = k_eigenvalues_1000.argsort()[::-1]
k_eigenvalues_1000 = k_eigenvalues_1000[index]
k_eigenvectors_1000 = k_eigenvectors_1000[index]

a_1000_1 = k_eigenvectors_1000[0]/math.sqrt(k_eigenvalues_1000[0])
a_1000_2 = k_eigenvectors_1000[1]/math.sqrt(k_eigenvalues_1000[1])

p_1_1000 = []
for i in range(len(kernel_image_data)):
    projection = 0
    for j in range(len(kernel_image_data)):
        projection += a_1000_1[j] * centered_sigma_1000[i][j]
    p_1_1000.append(projection)

p_2_1000 = []
for i in range(len(kernel_image_data)):
    projection = 0
    for j in range(len(kernel_image_data)):
        projection += a_1000_2[j] * centered_sigma_1000[i][j]
    p_2_1000.append(projection)

fig = plt.figure(figsize=(6,6))
ax = plt.scatter(p_1_1000, p_2_1000)
plt.xlabel('Projection on first principal component')
plt.ylabel('Projection on second principal component')
plt.title('Projections of datapoints on first 2 principal components', fontsize=10, pad=25)
plt.text(0.68, 1.05, 'Radial basis kernel, $\\sigma$ = 1000', fontsize=9.5, ha='right', va='top', transform=plt.gca().transAxes) 
plt.grid()
plt.show()

sigma_10000 = radial_kernel_calculator(kernel_image_data, 10000)
centered_sigma_10000 = centering(sigma_10000)
k_eigenvalues_10000, k_eigenvectors_10000 = np.linalg.eigh(centered_sigma_10000)
k_eigenvectors_10000 = k_eigenvectors_10000.transpose()
positive_eigenvalues = []
positive_eigenvectors = []
for i in range(len(k_eigenvalues_10000)):
    if (k_eigenvalues_10000[i] > 0):
        positive_eigenvalues.append(k_eigenvalues_10000[i])
        positive_eigenvectors.append(k_eigenvectors_10000[i])
        

k_eigenvalues_10000 = np.array(positive_eigenvalues)
k_eigenvectors_10000 = np.array(positive_eigenvectors)

index = k_eigenvalues_10000.argsort()[::-1]
k_eigenvalues_10000 = k_eigenvalues_10000[index]
k_eigenvectors_10000 = k_eigenvectors_10000[index]

a_10000_1 = k_eigenvectors_10000[0]/math.sqrt(k_eigenvalues_10000[0])
a_10000_2 = k_eigenvectors_10000[1]/math.sqrt(k_eigenvalues_10000[1])

p_1_10000 = []
for i in range(len(kernel_image_data)):
    projection = 0
    for j in range(len(kernel_image_data)):
        projection += a_10000_1[j] * centered_sigma_10000[i][j]
    p_1_10000.append(projection)

p_2_10000 = []
for i in range(len(kernel_image_data)):
    projection = 0
    for j in range(len(kernel_image_data)):
        projection += a_10000_2[j] * centered_sigma_10000[i][j]
    p_2_10000.append(projection)

fig = plt.figure(figsize=(6,6))
ax = plt.scatter(p_1_10000, p_2_10000)
plt.xlabel('Projection on first principal component')
plt.ylabel('Projection on second principal component')
plt.title('Projections of datapoints on first 2 principal components', fontsize=10, pad=25)
plt.text(0.68, 1.05, 'Radial basis kernel, $\\sigma$ = 10000', fontsize=9.5, ha='right', va='top', transform=plt.gca().transAxes) 
plt.grid()
plt.show()