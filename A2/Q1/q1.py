import csv
import numpy as np
import matplotlib.pyplot as plt
import random

# reading the dataset
data = []
with open('A2Q1.csv', newline='') as file:
    csv_file_reader = csv.reader(file)
    for row in csv_file_reader:
        data.append(row)

# converting the data in strings to int
for i in range(len(data)):
    for j in range(len(data[0])):
        data[i][j] = int(data[i][j])

# part i - bernoulli multivariate distribution

import numpy as np

def initialize_parameters(data, num_clusters, num_features):
    pi = np.ones(num_clusters) / num_clusters
    thetas = np.random.rand(num_clusters, num_features)  # randomly initialize Bernoulli parameters
    return pi, thetas

def bernoulli_pdf(x, theta):
    return np.prod(np.where(x, theta, 1 - theta))

def expectation(data, pi, thetas):
    N, D = data.shape
    K = len(pi)
    responsibilities = np.zeros((N, K))

    for n in range(N):
        for k in range(K):
            responsibilities[n, k] = pi[k] * bernoulli_pdf(data[n], thetas[k])
        responsibilities[n, :] /= np.sum(responsibilities[n, :])

    return responsibilities

def maximization(data, responsibilities):

    N, D = data.shape
    K = responsibilities.shape[1]

    # update mixing coefficients
    pi = np.sum(responsibilities, axis=0) / N

    # update Bernoulli parameters
    thetas = np.zeros((K, D))
    for k in range(K):
        numerator = np.dot(responsibilities[:, k], data)
        denominator = np.sum(responsibilities[:, k])
        thetas[k] = numerator / denominator

    return pi, thetas

def log_likelihood(data, pi, thetas):
    """
    Compute the log-likelihood of the observed data under the Bernoulli mixture model.
    
    Parameters:
        data (ndarray): Observed data points (NxD array).
        pi (ndarray): Mixing coefficients for each cluster (1xK array).
        thetas (ndarray): Bernoulli parameters for each cluster (KxD array).
    
    Returns:
        float: Log-likelihood of the observed data.
    """
    N, D = data.shape
    K = len(pi)
    log_likelihood = 0

    for n in range(N):
        likelihood = 0
        for k in range(K):
            likelihood += pi[k] * bernoulli_pdf(data[n], thetas[k])
        log_likelihood += np.log(likelihood)

    return log_likelihood

def em_algorithm(data, num_clusters, max_iterations=30, tolerance=1e-6):
    """
    Perform the Expectation-Maximization (EM) algorithm for a Bernoulli mixture model.
    
    Parameters:
        data (ndarray): Observed data points (NxD array).
        num_clusters (int): Number of clusters.
        max_iterations (int): Maximum number of iterations (default is 100).
        tolerance (float): Convergence tolerance (default is 1e-6).
    
    Returns:
        tuple: Tuple containing final mixing coefficients, Bernoulli parameters, and log-likelihoods.
    """
    N, D = data.shape
    pi, thetas = initialize_parameters(data, num_clusters, D)
    prev_log_likelihood = None
    log_likelihoods = []

    for iteration in range(max_iterations):
        # E-step
        responsibilities = expectation(data, pi, thetas)
        
        # M-step
        pi, thetas = maximization(data, responsibilities)
        
        # compute log-likelihood
        curr_log_likelihood = log_likelihood(data, pi, thetas)
        log_likelihoods.append(curr_log_likelihood)
        
        # convergence check
        if prev_log_likelihood is not None and np.abs(curr_log_likelihood - prev_log_likelihood) < tolerance:
            break
        
        prev_log_likelihood = curr_log_likelihood

    return pi, thetas, log_likelihoods

data = np.array(data)

pi, thetas, log_likelihoods = em_algorithm(data, 4)

plt.plot(range(1, len(log_likelihoods) + 1), log_likelihoods)
plt.xlabel('Iterations')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood of Bernoulli Multivariate Model as a function of iterations')
plt.grid()
plt.show()

# part ii - gaussian multivariate distribution

def initialize_parameters(data, num_clusters):
    N, D = data.shape  # number of data points and dimensions

    cluster_indices = np.random.choice(num_clusters, size=N)

    means = np.array([data[cluster_indices == k].mean(axis=0) for k in range(num_clusters)])

    covariances = np.array([np.cov(data[cluster_indices == k].T) for k in range(num_clusters)])

    mixing_coefficients = np.array([np.sum(cluster_indices == k) for k in range(num_clusters)]) / N

    return means, covariances, mixing_coefficients


def multivariate_gaussian_pdf(x, mu, cov):
    D = len(x)
    coefficient = 1 / np.sqrt((2 * np.pi) ** D * np.linalg.det(cov))
    exponent = -0.5 * np.dot(np.dot((x - mu).T, np.linalg.inv(cov)), (x - mu))
    pdf_value = coefficient * np.exp(exponent)
    return pdf_value

def expectation(data, means, covariances, mixing_coefficients):
    N = data.shape[0]  # data points
    K = means.shape[0]  # gaussian components
    responsibilities = np.zeros((N, K))

    for n in range(N):
        for k in range(K):
            pdf_value = multivariate_gaussian_pdf(data[n], means[k], covariances[k])
            responsibilities[n, k] = mixing_coefficients[k] * pdf_value
        responsibilities[n, :] /= np.sum(responsibilities[n, :])
    return responsibilities

def maximization(data, responsibilities):

    N, D = data.shape  # no of data points and dimensions
    K = responsibilities.shape[1]  # no of gaussian components

    # update means
    means = np.dot(responsibilities.T, data) / responsibilities.sum(axis=0)[:, np.newaxis]

    # update covariances
    covariances = np.zeros((K, D, D))
    for k in range(K):
        diff = data - means[k]
        covariances[k] = np.dot((responsibilities[:, k] * diff.T), diff) / responsibilities[:, k].sum()

    # update mixing coefficients
    mixing_coefficients = responsibilities.sum(axis=0) / N

    return means, covariances, mixing_coefficients

def em_algorithm(data, num_clusters, max_iterations=30, tolerance=1e-6):
    N, D = data.shape  # no of data points and dimensions
    K = num_clusters

    means, covariances, mixing_coefficients = initialize_parameters(data, num_clusters)

    prev_log_likelihood = None
    loglist = []
    for iteration in range(max_iterations):
        # E-step
        responsibilities = expectation(data, means, covariances, mixing_coefficients)
        
        # M-step
        means, covariances, mixing_coefficients = maximization(data, responsibilities)
        
        # compute log-likelihood
        log_likelihood = 0
        
        for n in range(N):
            likelihood = 0
            for k in range(K):
                likelihood += mixing_coefficients[k] * multivariate_gaussian_pdf(data[n], means[k], covariances[k])
            log_likelihood += np.log(likelihood)        
        # convergence check
        if prev_log_likelihood is not None and np.abs(log_likelihood - prev_log_likelihood) < tolerance:
            break
        prev_log_likelihood = log_likelihood
        loglist.append(prev_log_likelihood)
    
    return means, covariances, mixing_coefficients, loglist
    
data = np.array(data)

means, covariances, mixing_coefficients, loglist = em_algorithm(data, 4)

plt.plot(range(1, len(loglist) + 1), loglist)
plt.xlabel('Iterations')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood of Gaussian Multivariate Model as a function of iterations')
plt.grid()
plt.show()

# part iii - k means clustering


import math

def euclidean_distance(p, q):
    dist = 0
    for i in range(len(p)):
        dist += (float(p[i]) - float(q[i])) ** 2
    return math.sqrt(dist)

def calculate_error(error, means, clusters):
    total_e = 0
    for i in range(len(means)):
        e = 0
        for point in clusters[i]:
            for j in range(len(point)):
                e += (float(means[i][j]) - float(point[j])) ** 2
        total_e += e
    error.append(total_e)

def k_means_clustering(initial_means, dataset, k):
    means = initial_means
    convergence = False
    error = []
    while not convergence:
        clusters = [[] for _ in range(k)]
        for i in dataset:
            distances_from_centroids = []
            for j in range(k):
                distances_from_centroids.append([euclidean_distance(i, means[j]), j])
            required_cluster = min(distances_from_centroids, key=lambda x: x[0])[1]
            clusters[required_cluster].append(i)
            
        new_means = []
        for i in range(k):
            if clusters[i]:
                mean_point = [sum(coords)/len(coords) for coords in zip(*clusters[i])]
                new_means.append(mean_point)
            else:
                new_means.append(means[i])
                
        if all(means[i] == new_means[i] for i in range(k)):
            break

        calculate_error(error, means, clusters)
        means = new_means 

    return clusters, means, error

data = []
with open('A2Q1.csv', newline='') as file:
    csv_file_reader = csv.reader(file)
    for row in csv_file_reader:
        data.append(row)

for i in range(len(data)):
    for j in range(len(data[0])):
        data[i][j] = int(data[i][j])

initial_means = random.sample(data, 4)
clusters, means, error = k_means_clustering(initial_means, data, 4)

plt.plot(range(1, len(error) + 1), error)
plt.xlabel('Iterations')
plt.ylabel('Objective (Error)')
plt.title('Objective of K-means as a function of iterations')
plt.grid()
plt.show()
