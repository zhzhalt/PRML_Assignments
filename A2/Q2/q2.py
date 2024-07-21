# importing relevant libraries
import csv
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold

# reading the data from csv files
train_data = []
with open('A2Q2Data_train.csv', newline='') as file:
    csv_file_reader = csv.reader(file)
    for row in csv_file_reader:
        train_data.append(row)
test_data = []
with open('A2Q2Data_test.csv', newline='') as file:
    csv_file_reader = csv.reader(file)
    for row in csv_file_reader:
        test_data.append(row)

# converting the data in strings to int
for i in range(len(train_data)):
    for j in range(len(train_data[0])):
        train_data[i][j] = float(train_data[i][j])

for i in range(len(test_data)):
    for j in range(len(test_data[0])):
        test_data[i][j] = float(test_data[i][j])

# separating the features and labels in the data
train_y = []
train_x = [[] for _ in range(len(train_data))]

for i in range(len(train_data)):
    train_y.append(train_data[i][100])  
    
    for j in range(len(train_data[0])-1):
        train_x[i].append(train_data[i][j])  

# part i 

# finding the analytical solution for linear regression
train_x_arr = np.array(train_x).transpose()
train_y_arr = np.array(train_y).reshape(-1,1)
covariance = np.dot(train_x_arr, train_x_arr.transpose())
inv_covariance = np.linalg.inv(covariance)
wstar = np.dot(np.dot(inv_covariance, train_x_arr), train_y_arr)

# calculating the mse the analytical solution of linear regression makes 
opt_error = np.mean((np.dot(train_x_arr.T, wstar) - train_y_arr) ** 2)

# plotting wstar
plt.figure(figsize=(8, 6))
plt.plot(wstar, 'bo-')
plt.xlabel('Feature Index')
plt.ylabel('Weight Value')
plt.title('Visualization of Weight Vector w*')
plt.grid(True)
plt.show()

# part ii

# gradient descent algorithm
def gradient_descent(train_x_arr, train_y_arr, step_size, max_iterations, tolerance):
    num_features = train_x_arr.shape[0]  # finding the number of features from the X np array
    w = np.zeros((100,1))                # taking w to be an array of zeros as w0 
    iteration = 1 
    w_it_list = [(w, iteration)]         # list that contains the w that is obtained for that iteration of the algorithm
    error_list = [np.mean((np.dot(train_x_arr.T, w) - train_y_arr) ** 2)]                           # list that stores the value of mse wt makes 
    while iteration < max_iterations:
        gradient = (2*(np.dot(covariance, w).reshape(-1,1) - np.dot(train_x_arr, train_y_arr)))     # finding the gradient
        w_new = w - step_size * gradient                                                            # updation step
        w_it_list.append((w_new, iteration+1))
        mse_w = np.mean((np.dot(train_x_arr.T, w) - train_y_arr) ** 2)
        mse_w_new = np.mean((np.dot(train_x_arr.T, w_new) - train_y_arr) ** 2)
        abs_diff = abs(mse_w_new - mse_w)
        if abs_diff < tolerance:                                                                    # convergence when the difference in error made by wt+1 and wt is less than tolerance
            break
        w = w_new
        iteration += 1
        error_list.append(mse_w_new)
    return w, w_it_list, error_list

# calling the gradient descent algorithm

# fixed no of iterations, different step sizes

grad_w_1, w_it_list_1, error_list_1 = gradient_descent(train_x_arr, train_y_arr, 0.00001, 100, 1e-6)   #1e-5 step size
grad_w_2, w_it_list_2, error_list_2 = gradient_descent(train_x_arr, train_y_arr, 0.000001, 100, 1e-6)  #1e-6 step size
grad_w_3, w_it_list_3, error_list_3 = gradient_descent(train_x_arr, train_y_arr, 0.0000001, 100, 1e-6) #1e-7 step size
grad_w_4, w_it_list_4, error_list_4 = gradient_descent(train_x_arr, train_y_arr, 0.00000001, 100, 1e-6) #1e-8 step size
grad_w_5, w_it_list_5, error_list_5 = gradient_descent(train_x_arr, train_y_arr, 0.000000001, 100, 1e-6) #1e-9 step size
grad_w_6, w_it_list_6, error_list_6 = gradient_descent(train_x_arr, train_y_arr, 0.0000000001, 100, 1e-6) #1e-10 step size

# function to calculate squared distances between wt and wstar
def calculate_squared_distances(w_list, optimal_w):
    squared_distances = []
    for w in w_list:
        squared_distances.append(np.linalg.norm(w[0] - optimal_w) ** 2)
    return squared_distances

# calculate squared distances for each w_it_list
squared_distances_1 = calculate_squared_distances(w_it_list_1, wstar)
squared_distances_2 = calculate_squared_distances(w_it_list_2, wstar)
squared_distances_3 = calculate_squared_distances(w_it_list_3, wstar)
squared_distances_4 = calculate_squared_distances(w_it_list_4, wstar)
squared_distances_5 = calculate_squared_distances(w_it_list_5, wstar)
squared_distances_6 = calculate_squared_distances(w_it_list_6, wstar)

# plot squared distances

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].plot(range(1, len(squared_distances_1) + 1), squared_distances_1)
axs[0, 0].set_title('Squared Distances (Step size: 1e-5)')
axs[0, 0].set_xlabel('Iteration (t)')
axs[0, 0].set_ylabel('|wt - wML|^2')
axs[0, 0].grid(True)

axs[0, 1].plot(range(1, len(squared_distances_2) + 1), squared_distances_2)
axs[0, 1].set_title('Squared Distances (Step size: 1e-6)')
axs[0, 1].set_xlabel('Iteration (t)')
axs[0, 1].set_ylabel('|wt - wML|^2')
axs[0, 1].grid(True)

axs[0, 2].plot(range(1, len(squared_distances_3) + 1), squared_distances_3)
axs[0, 2].set_title('Squared Distances (Step size: 1e-7)')
axs[0, 2].set_xlabel('Iteration (t)')
axs[0, 2].set_ylabel('|wt - wML|^2')
axs[0, 2].grid(True)

axs[1, 0].plot(range(1, len(squared_distances_4) + 1), squared_distances_4)
axs[1, 0].set_title('Squared Distances (Step size: 1e-8)')
axs[1, 0].set_xlabel('Iteration (t)')
axs[1, 0].set_ylabel('|wt - wML|^2')
axs[1, 0].grid(True)

axs[1, 1].plot(range(1, len(squared_distances_5) + 1), squared_distances_5)
axs[1, 1].set_title('Squared Distances (Step size: 1e-9)')
axs[1, 1].set_xlabel('Iteration (t)')
axs[1, 1].set_ylabel('|wt - wML|^2')
axs[1, 1].grid(True)

axs[1, 2].plot(range(1, len(squared_distances_6) + 1), squared_distances_6)
axs[1, 2].set_title('Squared Distances (Step size: 1e-10)')
axs[1, 2].set_xlabel('Iteration (t)')
axs[1, 2].set_ylabel('|wt - wML|^2')
axs[1, 2].grid(True)

plt.tight_layout()
plt.show()


iteration_numbers = range(1, len(error_list_2) + 1)

plt.figure(figsize=(8, 6))

plt.plot(iteration_numbers, squared_distances_2, label='Step Size: 1e-6')
plt.plot(iteration_numbers, squared_distances_3, label='Step Size: 1e-7')
plt.plot(iteration_numbers, squared_distances_4, label='Step Size: 1e-8')
plt.plot(iteration_numbers, squared_distances_5, label='Step Size: 1e-9')
plt.plot(iteration_numbers, squared_distances_6, label='Step Size: 1e-10')

plt.xlabel('Number of Iterations')
plt.ylabel('|wt - wML|^2')
plt.title('|wt - wML|^2 vs. Number of Iterations (100) for Different Step Sizes')
plt.grid(True)
plt.legend()
plt.show()

# plot errors 

iteration_numbers_1 = range(1, len(error_list_1) + 1)
iteration_numbers_2 = range(1, len(error_list_2) + 1)
iteration_numbers_3 = range(1, len(error_list_3) + 1)
opt_line = [opt_error for _ in iteration_numbers_1]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].plot(iteration_numbers_1, error_list_1, label='Step Size: 1e-5', linestyle='-')
axes[0].set_xlabel('Number of Iterations')
axes[0].set_ylabel('Error')
axes[0].set_title('Error vs. Number of Iterations (Step Size: 1e-5)')
axes[0].plot(iteration_numbers_1, opt_line, linestyle='--')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(iteration_numbers_2, error_list_2, label='Step Size: 1e-6', linestyle='-')
axes[1].set_xlabel('Number of Iterations')
axes[1].set_ylabel('Error')
axes[1].set_title('Error vs. Number of Iterations (Step Size: 1e-6)')
axes[1].plot(iteration_numbers_2, opt_line, linestyle='--')
axes[1].grid(True)
axes[1].legend()

axes[2].plot(iteration_numbers_3, error_list_3, label='Step Size: 1e-7', linestyle='-')
axes[2].set_xlabel('Number of Iterations')
axes[2].set_ylabel('Error')
axes[2].set_title('Error vs. Number of Iterations (Step Size: 1e-7)')
axes[2].plot(iteration_numbers_3, opt_line, linestyle='--')
axes[2].grid(True)
axes[2].legend()

plt.suptitle('Error vs. Number of Iterations for 100 Iterations')

plt.tight_layout()
plt.show()


iteration_numbers_4 = range(1, len(error_list_4) + 1)
iteration_numbers_5 = range(1, len(error_list_5) + 1)
iteration_numbers_6 = range(1, len(error_list_6) + 1)
opt_line = [opt_error for _ in iteration_numbers_4]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].plot(iteration_numbers_4, error_list_4, label='Step Size: 1e-8', linestyle='-')
axes[0].set_xlabel('Number of Iterations')
axes[0].set_ylabel('Error')
axes[0].set_title('Error vs. Number of Iterations (Step Size: 1e-8)')
axes[0].plot(iteration_numbers_4, opt_line, linestyle='--')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(iteration_numbers_5, error_list_5, label='Step Size: 1e-9', linestyle='-')
axes[1].set_xlabel('Number of Iterations')
axes[1].set_ylabel('Error')
axes[1].set_title('Error vs. Number of Iterations (Step Size: 1e-9)')
axes[1].plot(iteration_numbers_5, opt_line, linestyle='--')
axes[1].grid(True)
axes[1].legend()

axes[2].plot(iteration_numbers_6, error_list_6, label='Step Size: 1e-10', linestyle='-')
axes[2].set_xlabel('Number of Iterations')
axes[2].set_ylabel('Error')
axes[2].set_title('Error vs. Number of Iterations (Step Size: 1e-10)')
axes[2].plot(iteration_numbers_6, opt_line, linestyle='--')
axes[2].grid(True)
axes[2].legend()

plt.suptitle('Error vs. Number of Iterations for 100 Iterations')

plt.tight_layout()
plt.show()


iteration_numbers = range(1, len(error_list_2) + 1)

plt.figure(figsize=(8, 6))

plt.plot(iteration_numbers, error_list_2, label='Step Size: 1e-6')
plt.plot(iteration_numbers, error_list_3, label='Step Size: 1e-7')
plt.plot(iteration_numbers, error_list_4, label='Step Size: 1e-8')
plt.plot(iteration_numbers, error_list_5, label='Step Size: 1e-9')
plt.plot(iteration_numbers, error_list_6, label='Step Size: 1e-10')

plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Error vs. Number of Iterations (100) for Different Step Sizes')
plt.grid(True)
plt.legend()
plt.show()

grad_w_7, w_it_list_7, error_list_7 = gradient_descent(train_x_arr, train_y_arr, 0.00001, 1000, 1e-6)   #1e-5 step size
grad_w_8, w_it_list_8, error_list_8 = gradient_descent(train_x_arr, train_y_arr, 0.000001, 1000, 1e-6)  #1e-6 step size
grad_w_9, w_it_list_9, error_list_9 = gradient_descent(train_x_arr, train_y_arr, 0.0000001, 1000, 1e-6) #1e-7 step size
grad_w_10, w_it_list_10, error_list_10 = gradient_descent(train_x_arr, train_y_arr, 0.00000001, 1000, 1e-6) #1e-8 step size
grad_w_11, w_it_list_11, error_list_11 = gradient_descent(train_x_arr, train_y_arr, 0.000000001, 1000, 1e-6) #1e-9 step size
grad_w_12, w_it_list_12, error_list_12 = gradient_descent(train_x_arr, train_y_arr, 0.0000000001, 1000, 1e-6) #1e-10 step size

squared_distances_7 = calculate_squared_distances(w_it_list_7, wstar)
squared_distances_8 = calculate_squared_distances(w_it_list_8, wstar)
squared_distances_9 = calculate_squared_distances(w_it_list_9, wstar)
squared_distances_10 = calculate_squared_distances(w_it_list_10, wstar)
squared_distances_11 = calculate_squared_distances(w_it_list_11, wstar)
squared_distances_12 = calculate_squared_distances(w_it_list_12, wstar)


fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].plot(range(1, len(squared_distances_7) + 1), squared_distances_7)
axs[0, 0].set_title('Squared Distances (1e-5)')
axs[0, 0].set_xlabel('Iteration (t)')
axs[0, 0].set_ylabel('|wt - wML|^2')
axs[0, 0].grid(True)

axs[0, 1].plot(range(1, len(squared_distances_8) + 1), squared_distances_8)
axs[0, 1].set_title('Squared Distances (1e-6)')
axs[0, 1].set_xlabel('Iteration (t)')
axs[0, 1].set_ylabel('|wt - wML|^2')
axs[0, 1].grid(True)

axs[0, 2].plot(range(1, len(squared_distances_9) + 1), squared_distances_9)
axs[0, 2].set_title('Squared Distances (1e-7)')
axs[0, 2].set_xlabel('Iteration (t)')
axs[0, 2].set_ylabel('|wt - wML|^2')
axs[0, 2].grid(True)

axs[1, 0].plot(range(1, len(squared_distances_10) + 1), squared_distances_10)
axs[1, 0].set_title('Squared Distances (1e-8)')
axs[1, 0].set_xlabel('Iteration (t)')
axs[1, 0].set_ylabel('|wt - wML|^2')
axs[1, 0].grid(True)

axs[1, 1].plot(range(1, len(squared_distances_11) + 1), squared_distances_11)
axs[1, 1].set_title('Squared Distances (1e-9)')
axs[1, 1].set_xlabel('Iteration (t)')
axs[1, 1].set_ylabel('|wt - wML|^2')
axs[1, 1].grid(True)

axs[1, 2].plot(range(1, len(squared_distances_12) + 1), squared_distances_12)
axs[1, 2].set_title('Squared Distances (1e-10)')
axs[1, 2].set_xlabel('Iteration (t)')
axs[1, 2].set_ylabel('|wt - wML|^2')
axs[1, 2].grid(True)

plt.tight_layout()
plt.show()


iteration_numbers = range(1, len(error_list_8) + 1)

plt.figure(figsize=(8, 6))

plt.plot(iteration_numbers, squared_distances_8, label='Step Size: 1e-6')
plt.plot(iteration_numbers, squared_distances_9, label='Step Size: 1e-7')
plt.plot(iteration_numbers, squared_distances_10, label='Step Size: 1e-8')
plt.plot(iteration_numbers, squared_distances_11, label='Step Size: 1e-9')
plt.plot(iteration_numbers, squared_distances_12, label='Step Size: 1e-10')

plt.xlabel('Number of Iterations')
plt.ylabel('|wt - wML|^2')
plt.title('|wt - wML|^2 vs. Number of Iterations (1000) for Different Step Sizes')
plt.grid(True)
plt.legend()
plt.show()


iteration_numbers_7 = range(1, len(error_list_7) + 1)
iteration_numbers_8 = range(1, len(error_list_8) + 1)
iteration_numbers_9 = range(1, len(error_list_9) + 1)
opt_line = [opt_error for _ in iteration_numbers_7]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].plot(iteration_numbers_7, error_list_7, label='Step Size: 1e-5', linestyle='-')
axes[0].set_xlabel('Number of Iterations')
axes[0].set_ylabel('Error')
axes[0].set_title('Error vs. Number of Iterations (Step Size: 1e-5)')
axes[0].plot(iteration_numbers_7, opt_line, linestyle='--')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(iteration_numbers_8, error_list_8, label='Step Size: 1e-6', linestyle='-')
axes[1].set_xlabel('Number of Iterations')
axes[1].set_ylabel('Error')
axes[1].set_title('Error vs. Number of Iterations (Step Size: 1e-6)')
axes[1].plot(iteration_numbers_8, opt_line, linestyle='--')
axes[1].grid(True)
axes[1].legend()

axes[2].plot(iteration_numbers_9, error_list_9, label='Step Size: 1e-7', linestyle='-')
axes[2].set_xlabel('Number of Iterations')
axes[2].set_ylabel('Error')
axes[2].set_title('Error vs. Number of Iterations (Step Size: 1e-7)')
axes[2].plot(iteration_numbers_9, opt_line, linestyle='--')
axes[2].grid(True)
axes[2].legend()

plt.suptitle('Error vs. Number of Iterations for 1000 Iterations')

plt.tight_layout()
plt.show()



iteration_numbers_10 = range(1, len(error_list_10) + 1)
iteration_numbers_11 = range(1, len(error_list_11) + 1)
iteration_numbers_12 = range(1, len(error_list_12) + 1)
opt_line = [opt_error for _ in iteration_numbers_10]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].plot(iteration_numbers_10, error_list_10, label='Step Size: 1e-8', linestyle='-')
axes[0].set_xlabel('Number of Iterations')
axes[0].set_ylabel('Error')
axes[0].set_title('Error vs. Number of Iterations (Step Size: 1e-8)')
axes[0].plot(iteration_numbers_10, opt_line, linestyle='--')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(iteration_numbers_11, error_list_11, label='Step Size: 1e-9', linestyle='-')
axes[1].set_xlabel('Number of Iterations')
axes[1].set_ylabel('Error')
axes[1].set_title('Error vs. Number of Iterations (Step Size: 1e-9)')
axes[1].plot(iteration_numbers_11, opt_line, linestyle='--')
axes[1].grid(True)
axes[1].legend()

axes[2].plot(iteration_numbers_12, error_list_12, label='Step Size: 1e-10', linestyle='-')
axes[2].set_xlabel('Number of Iterations')
axes[2].set_ylabel('Error')
axes[2].set_title('Error vs. Number of Iterations (Step Size: 1e-10)')
axes[2].plot(iteration_numbers_12, opt_line, linestyle='--')
axes[2].grid(True)
axes[2].legend()

plt.suptitle('Error vs. Number of Iterations for 1000 Iterations')

plt.tight_layout()
plt.show()


iteration_numbers = range(1, len(error_list_8) + 1)

plt.figure(figsize=(8, 6))

plt.plot(iteration_numbers, error_list_8, label='Step Size: 1e-6')
plt.plot(iteration_numbers, error_list_9, label='Step Size: 1e-7')
plt.plot(iteration_numbers, error_list_10, label='Step Size: 1e-8')
plt.plot(iteration_numbers, error_list_11, label='Step Size: 1e-9')
plt.plot(iteration_numbers, error_list_12, label='Step Size: 1e-10')

plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Error vs. Number of Iterations (1000) for Different Step Sizes')
plt.grid(True)
plt.legend()
plt.show()

grad_w_13, w_it_list_13, error_list_13 = gradient_descent(train_x_arr, train_y_arr, 0.000001, 2500, 1e-6)  #1e-6 step size
grad_w_14, w_it_list_14, error_list_14 = gradient_descent(train_x_arr, train_y_arr, 0.0000001, 2500, 1e-6) #1e-7 step size
grad_w_15, w_it_list_15, error_list_15 = gradient_descent(train_x_arr, train_y_arr, 0.00000001, 2500, 1e-6) #1e-8 step size
grad_w_16, w_it_list_16, error_list_16 = gradient_descent(train_x_arr, train_y_arr, 0.000000001, 2500, 1e-6) #1e-9 step size
grad_w_17, w_it_list_17, error_list_17 = gradient_descent(train_x_arr, train_y_arr, 0.0000000001, 2500, 1e-6) #1e-10 step size

squared_distances_13 = calculate_squared_distances(w_it_list_13, wstar)
squared_distances_14 = calculate_squared_distances(w_it_list_14, wstar)
squared_distances_15 = calculate_squared_distances(w_it_list_15, wstar)
squared_distances_16 = calculate_squared_distances(w_it_list_16, wstar)
squared_distances_17 = calculate_squared_distances(w_it_list_17, wstar)



fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].plot(range(1, len(squared_distances_13) + 1), squared_distances_13)
axs[0, 0].set_title('Squared Distances (1e-6)')
axs[0, 0].set_xlabel('Iteration (t)')
axs[0, 0].set_ylabel('|wt - wML|^2')
axs[0, 0].grid(True)

axs[0, 1].plot(range(1, len(squared_distances_14) + 1), squared_distances_14)
axs[0, 1].set_title('Squared Distances (1e-7)')
axs[0, 1].set_xlabel('Iteration (t)')
axs[0, 1].set_ylabel('|wt - wML|^2')
axs[0, 1].grid(True)

axs[0, 2].plot(range(1, len(squared_distances_15) + 1), squared_distances_15)
axs[0, 2].set_title('Squared Distances (1e-8)')
axs[0, 2].set_xlabel('Iteration (t)')
axs[0, 2].set_ylabel('|wt - wML|^2')
axs[0, 2].grid(True)                  

axs[1, 0].plot(range(1, len(squared_distances_16) + 1), squared_distances_16)
axs[1, 0].set_title('Squared Distances (1e-9)')
axs[1, 0].set_xlabel('Iteration (t)')
axs[1, 0].set_ylabel('|wt - wML|^2')
axs[1, 0].grid(True)

axs[1, 1].plot(range(1, len(squared_distances_17) + 1), squared_distances_17)
axs[1, 1].set_title('Squared Distances (1e-10)')
axs[1, 1].set_xlabel('Iteration (t)')
axs[1, 1].set_ylabel('|wt - wML|^2')
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()


iteration_numbers = range(1, len(error_list_14) + 1)

plt.figure(figsize=(8, 6))

plt.plot(iteration_numbers, squared_distances_14, label='Step Size: 1e-7')
plt.plot(iteration_numbers, squared_distances_15, label='Step Size: 1e-8')
plt.plot(iteration_numbers, squared_distances_16, label='Step Size: 1e-9')
plt.plot(iteration_numbers, squared_distances_17, label='Step Size: 1e-10')

plt.xlabel('Number of Iterations')
plt.ylabel('|wt - wML|^2')
plt.title('|wt - wML|^2 vs. Number of Iterations (2500) for Different Step Sizes')
plt.grid(True)
plt.legend()
plt.show()

# 13 converges at 1995 only, so separate graph for it

iteration_numbers = range(1, len(error_list_13) + 1)
opt_line = [opt_error for _ in iteration_numbers]
plt.plot(iteration_numbers, error_list_13, linestyle='-')
plt.plot(iteration_numbers, opt_line, linestyle='-')
# plt.plot(len(error_list_13), 'ro')
# plt.plot(iteration_numbers, opt_error, 'ro')
plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Step size: 1e-6')
plt.grid(True)
plt.show()


iteration_numbers = range(1, len(error_list_14) + 1)

plt.figure(figsize=(8, 6))

plt.plot(iteration_numbers, error_list_14, label='Step Size: 1e-7')
plt.plot(iteration_numbers, error_list_15, label='Step Size: 1e-8')
plt.plot(iteration_numbers, error_list_16, label='Step Size: 1e-9')
plt.plot(iteration_numbers, error_list_17, label='Step Size: 1e-10')

plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Error vs. Number of Iterations (2500) for Different Step Sizes')
plt.grid(True)
plt.legend()
plt.show()

# gradient descent on taking step sizes in order of a sequence
def gradient_descent_sequence(train_x_arr, train_y_arr, max_iterations, tolerance):
    num_features = train_x_arr.shape[0] 
    w = np.zeros((100,1))
    iteration = 1
    w_it_list = [(w, iteration)]
    step_sizes = [1 / (iter) for iter in range(1, max_iterations+1)]       # defining the sequence
    error_list = [np.mean((np.dot(train_x_arr.T, w) - train_y_arr) ** 2)]
    while iteration < max_iterations:
        gradient = (2*(np.dot(covariance, w).reshape(-1,1) - np.dot(train_x_arr, train_y_arr)))
        w_new = w - step_sizes[iteration] * gradient
        w_it_list.append((w_new, iteration+1))
        mse_w = np.mean((np.dot(train_x_arr.T, w) - train_y_arr) ** 2)
        mse_w_new = np.mean((np.dot(train_x_arr.T, w_new) - train_y_arr) ** 2)
        abs_diff = abs(mse_w_new - mse_w)
        if abs_diff < tolerance:
            break
        w = w_new
        iteration += 1
        error_list.append(mse_w_new)
    return w, w_it_list, error_list

grad_w_1_seq, w_it_list_1_seq, error_list_1_seq = gradient_descent_sequence(train_x_arr, train_y_arr, 100, 1e-6)

iteration_numbers = range(1, len(error_list_1_seq) + 1)
opt_line = [opt_error for _ in iteration_numbers]
plt.plot(iteration_numbers, error_list_1_seq, linestyle='-')
plt.plot(iteration_numbers, opt_line, linestyle='-')
plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Error vs. Number of Iterations')
plt.grid(True)
plt.show()


# plotting the difference between different ws and wstar
fig, axs = plt.subplots(1, 1, figsize=(10, 7))

axs.plot(wstar, 'bo-', label='Analytical Solution w*')
axs.plot(grad_w_2, 'ro-', label='Step Size: 1e-6')
axs.plot(grad_w_3, 'go-', label='Step Size: 1e-7')
axs.plot(grad_w_4, 'ko-', label='Step Size: 1e-8')
axs.plot(grad_w_5, 'yo-', label='Step Size: 1e-9')
axs.plot(grad_w_6, 'mo-', label='Step Size: 1e-10')
axs.set_xlabel('Feature Index')
axs.set_ylabel('Weight Value')
axs.set_title('Gradient Descent Solutions')
axs.grid(True)
axs.legend()

plt.suptitle('No of Iterations = 100')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(10, 7))

axs.plot(wstar, 'bo-', label='Analytical Solution w*')
axs.plot(grad_w_8, 'ro-', label='Step Size: 1e-6')
axs.plot(grad_w_9, 'go-', label='Step Size: 1e-7')
axs.plot(grad_w_10, 'ko-', label='Step Size: 1e-8')
axs.plot(grad_w_11, 'yo-', label='Step Size: 1e-9')
axs.plot(grad_w_12, 'mo-', label='Step Size: 1e-10')
axs.set_xlabel('Feature Index')
axs.set_ylabel('Weight Value')
axs.set_title('Gradient Descent Solutions')
axs.grid(True)
axs.legend()

plt.suptitle('No of Iterations = 1000')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(10, 7))

axs.plot(wstar, 'bo-', label='Analytical Solution w*')
axs.plot(grad_w_13, 'ro-', label='Step Size: 1e-6')
axs.plot(grad_w_14, 'go-', label='Step Size: 1e-7')
axs.plot(grad_w_15, 'ko-', label='Step Size: 1e-8')
axs.plot(grad_w_16, 'yo-', label='Step Size: 1e-9')
axs.plot(grad_w_17, 'mo-', label='Step Size: 1e-10')
axs.set_xlabel('Feature Index')
axs.set_ylabel('Weight Value')
axs.set_title('Gradient Descent Solutions')
axs.grid(True)
axs.legend()

plt.suptitle('No of Iterations = 2500')
plt.tight_layout()
plt.show()

# the best w from gradient descent
grad_w_111, w_it_list_111, error_list_111 = gradient_descent(train_x_arr, train_y_arr, 0.000001, 6000, 1e-10)  #1e-6 step size
iteration_numbers = range(1, len(error_list_111) + 1)
opt_line = [opt_error for _ in iteration_numbers]
plt.plot(iteration_numbers, error_list_111, linestyle='-')
plt.plot(iteration_numbers, opt_line, linestyle='-')
# plt.plot(iteration_numbers, opt_error, 'ro')
plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Error vs. Number of Iterations')
plt.grid(True)
plt.show()

squared_distances_111 = calculate_squared_distances(w_it_list_111, wstar)
plt.plot(range(len(squared_distances_111)), squared_distances_111, linestyle='-')
plt.xlabel('Iteration (t)')
plt.ylabel('|wt - wML|²')
plt.title('Squared Euclidean Distance from Optimal Weight Vector')
plt.grid(True)
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(10, 7))

axs.plot(wstar, 'bo-', label='Analytical Solution w*')
axs.plot(grad_w_111, 'ro-', label='Step Size: 1e-6')
axs.set_xlabel('Feature Index')
axs.set_ylabel('Weight Value')
axs.set_title('Gradient Descent Solutions')
axs.grid(True)
axs.legend()

plt.suptitle('No of Iterations = 5004')
plt.tight_layout()
plt.show()

# part iii - stochastic gradient descent

# stochastic gradient descent algorithm
def stochastic_gradient_descent(train_data, step_size, max_iterations, tolerance):
    w_new = None
    w = np.zeros((100, 1))
    opt_w = w
    w_list = []
    w_list.append(w)
    error_list = []
    for iteration in range(max_iterations):               
        batch = random.sample(train_data, k=100)               # taking a batch of 100 from the data
        batch_train_y = []
        batch_train_x = [[] for _ in range(len(batch))]
        
        for i in range(len(batch)):
            batch_train_y.append(batch[i][100])  
            
            for j in range(len(batch[0])-1):
                batch_train_x[i].append(batch[i][j])
                
        train_x_arr = np.array(batch_train_x).transpose()        # doing gradient descent on this batch
        train_y_arr = np.array(batch_train_y)
        train_y_arr = train_y_arr.reshape(-1, 1)
        covariance = np.dot(train_x_arr, train_x_arr.T)
        num_features = train_x_arr.shape[0]
        num_samples = train_x_arr.shape[1]
        error_list.append(np.mean((np.dot(train_x_arr.copy().T, w) - train_y_arr) ** 2))
        gradient = (2*(np.dot(covariance, w).reshape(-1,1) - np.dot(train_x_arr, train_y_arr)))
        # update weights
        w_new = w - step_size * gradient
        # opt_w += w_new
        mse_w = np.mean((np.dot(train_x_arr.T, w) - train_y_arr) ** 2)
        mse_w_new = np.mean((np.dot(train_x_arr.T, w_new) - train_y_arr) ** 2)
        abs_diff = abs(mse_w_new - mse_w)
        if abs_diff < tolerance:
            break
        w = w_new
        w_list.append(w)
        
    return w_list, error_list


# fixed no of iterations, different step sizes

w_list_1, stoc_error_list_1 = stochastic_gradient_descent(train_data, 0.0001, 100, 1e-6)   #1e-4 step size
w_list_2, stoc_error_list_2 = stochastic_gradient_descent(train_data, 0.00001, 100, 1e-6)  #1e-5 step size
w_list_3, stoc_error_list_3 = stochastic_gradient_descent(train_data, 0.000001, 100, 1e-6) #1e-6 step size
w_list_4, stoc_error_list_4 = stochastic_gradient_descent(train_data, 0.0000001, 100, 1e-6) #1e-7 step size
w_list_5, stoc_error_list_5 = stochastic_gradient_descent(train_data, 0.00000001, 100, 1e-6) #1e-8 step size
w_list_6, stoc_error_list_6 = stochastic_gradient_descent(train_data, 0.000000001, 100, 1e-6) #1e-9 step size


# function to calculate squared distances
def stoc_calculate_squared_distances(w_list, optimal_w):
    squared_distances = []
    for w in w_list:
        squared_distances.append(np.linalg.norm(w - optimal_w) ** 2)
    return squared_distances

# calculate squared distances for each w_it_list
squared_distances_1 = stoc_calculate_squared_distances(w_list_1, wstar)
squared_distances_2 = stoc_calculate_squared_distances(w_list_2, wstar)
squared_distances_3 = stoc_calculate_squared_distances(w_list_3, wstar)
squared_distances_4 = stoc_calculate_squared_distances(w_list_4, wstar)
squared_distances_5 = stoc_calculate_squared_distances(w_list_5, wstar)
squared_distances_6 = stoc_calculate_squared_distances(w_list_6, wstar)


fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].plot(range(1, len(squared_distances_1) + 1), squared_distances_1)
axs[0, 0].set_title('Squared Distances (Step size: 1e-4)')
axs[0, 0].set_xlabel('Iteration (t)')
axs[0, 0].set_ylabel('|wt - wML|^2')
axs[0, 0].grid(True)

axs[0, 1].plot(range(1, len(squared_distances_2) + 1), squared_distances_2)
axs[0, 1].set_title('Squared Distances (Step size: 1e-5)')
axs[0, 1].set_xlabel('Iteration (t)')
axs[0, 1].set_ylabel('|wt - wML|^2')
axs[0, 1].grid(True)

axs[0, 2].plot(range(1, len(squared_distances_3) + 1), squared_distances_3)
axs[0, 2].set_title('Squared Distances (Step size: 1e-6)')
axs[0, 2].set_xlabel('Iteration (t)')
axs[0, 2].set_ylabel('|wt - wML|^2')
axs[0, 2].grid(True)

axs[1, 0].plot(range(1, len(squared_distances_4) + 1), squared_distances_4)
axs[1, 0].set_title('Squared Distances (Step size: 1e-7)')
axs[1, 0].set_xlabel('Iteration (t)')
axs[1, 0].set_ylabel('|wt - wML|^2')
axs[1, 0].grid(True)

axs[1, 1].plot(range(1, len(squared_distances_5) + 1), squared_distances_5)
axs[1, 1].set_title('Squared Distances (Step size: 1e-8)')
axs[1, 1].set_xlabel('Iteration (t)')
axs[1, 1].set_ylabel('|wt - wML|^2')
axs[1, 1].grid(True)

axs[1, 2].plot(range(1, len(squared_distances_6) + 1), squared_distances_6)
axs[1, 2].set_title('Squared Distances (Step size: 1e-9)')
axs[1, 2].set_xlabel('Iteration (t)')
axs[1, 2].set_ylabel('|wt - wML|^2')
axs[1, 2].grid(True)

plt.tight_layout()
plt.show()

# plot errors


iteration_numbers = range(1, len(stoc_error_list_2) + 1)

plt.figure(figsize=(10, 8))

plt.plot(iteration_numbers, stoc_error_list_1, label='Step Size: 1e-4')
plt.plot(iteration_numbers, stoc_error_list_2, label='Step Size: 1e-5')
plt.plot(iteration_numbers, stoc_error_list_3, label='Step Size: 1e-6')
plt.plot(iteration_numbers, stoc_error_list_4, label='Step Size: 1e-7')
plt.plot(iteration_numbers, stoc_error_list_5, label='Step Size: 1e-8')
plt.plot(iteration_numbers, stoc_error_list_6, label='Step Size: 1e-9')

plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Error vs. Number of Iterations (100) for Different Step Sizes')
plt.grid(True)
plt.legend()
plt.show()

w_list_7, stoc_error_list_7 = stochastic_gradient_descent(train_data, 0.0001, 1000, 1e-6)   #1e-4 step size
w_list_8, stoc_error_list_8 = stochastic_gradient_descent(train_data, 0.00001, 1000, 1e-6)  #1e-5 step size
w_list_9, stoc_error_list_9 = stochastic_gradient_descent(train_data, 0.000001, 1000, 1e-6) #1e-6 step size
w_list_10, stoc_error_list_10 = stochastic_gradient_descent(train_data, 0.0000001, 1000, 1e-6) #1e-7 step size
w_list_11, stoc_error_list_11 = stochastic_gradient_descent(train_data, 0.00000001, 1000, 1e-6) #1e-8 step size
w_list_12, stoc_error_list_12 = stochastic_gradient_descent(train_data, 0.000000001, 1000, 1e-6) #1e-9 step size

squared_distances_7 = stoc_calculate_squared_distances(w_list_7, wstar)
squared_distances_8 = stoc_calculate_squared_distances(w_list_8, wstar)
squared_distances_9 = stoc_calculate_squared_distances(w_list_9, wstar)
squared_distances_10 = stoc_calculate_squared_distances(w_list_10, wstar)
squared_distances_11 = stoc_calculate_squared_distances(w_list_11, wstar)
squared_distances_12 = stoc_calculate_squared_distances(w_list_12, wstar)


fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].plot(range(1, len(squared_distances_7) + 1), squared_distances_7)
axs[0, 0].set_title('Squared Distances (1e-4)')
axs[0, 0].set_xlabel('Iteration (t)')
axs[0, 0].set_ylabel('|wt - wML|^2')
axs[0, 0].grid(True)

axs[0, 1].plot(range(1, len(squared_distances_8) + 1), squared_distances_8)
axs[0, 1].set_title('Squared Distances (1e-5)')
axs[0, 1].set_xlabel('Iteration (t)')
axs[0, 1].set_ylabel('|wt - wML|^2')
axs[0, 1].grid(True)

axs[0, 2].plot(range(1, len(squared_distances_9) + 1), squared_distances_9)
axs[0, 2].set_title('Squared Distances (1e-6)')
axs[0, 2].set_xlabel('Iteration (t)')
axs[0, 2].set_ylabel('|wt - wML|^2')
axs[0, 2].grid(True)

axs[1, 0].plot(range(1, len(squared_distances_10) + 1), squared_distances_10)
axs[1, 0].set_title('Squared Distances (1e-7)')
axs[1, 0].set_xlabel('Iteration (t)')
axs[1, 0].set_ylabel('|wt - wML|^2')
axs[1, 0].grid(True)

axs[1, 1].plot(range(1, len(squared_distances_11) + 1), squared_distances_11)
axs[1, 1].set_title('Squared Distances (1e-8)')
axs[1, 1].set_xlabel('Iteration (t)')
axs[1, 1].set_ylabel('|wt - wML|^2')
axs[1, 1].grid(True)

axs[1, 2].plot(range(1, len(squared_distances_12) + 1), squared_distances_12)
axs[1, 2].set_title('Squared Distances (1e-9)')
axs[1, 2].set_xlabel('Iteration (t)')
axs[1, 2].set_ylabel('|wt - wML|^2')
axs[1, 2].grid(True)

plt.tight_layout()
plt.show()



iteration_numbers = range(1, len(stoc_error_list_7) + 1)

plt.figure(figsize=(10, 8))

plt.plot(iteration_numbers, stoc_error_list_7, label='Step Size: 1e-4')
plt.plot(iteration_numbers, stoc_error_list_8, label='Step Size: 1e-5')
plt.plot(iteration_numbers, stoc_error_list_9, label='Step Size: 1e-6')
plt.plot(iteration_numbers, stoc_error_list_10, label='Step Size: 1e-7')
plt.plot(iteration_numbers, stoc_error_list_11, label='Step Size: 1e-8')
plt.plot(iteration_numbers, stoc_error_list_12, label='Step Size: 1e-9')


plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Error vs. Number of Iterations (1000) for Different Step Sizes')
plt.grid(True)
plt.legend()
plt.show()

w_list_13, stoc_error_list_13 = stochastic_gradient_descent(train_data, 0.0001, 2500, 1e-6)   #1e-4 step size
w_list_14, stoc_error_list_14 = stochastic_gradient_descent(train_data, 0.00001, 2500, 1e-6)  #1e-5 step size
w_list_15, stoc_error_list_15 = stochastic_gradient_descent(train_data, 0.000001, 2500, 1e-6) #1e-6 step size
w_list_16, stoc_error_list_16 = stochastic_gradient_descent(train_data, 0.0000001, 2500, 1e-6) #1e-7 step size
w_list_17, stoc_error_list_17 = stochastic_gradient_descent(train_data, 0.00000001, 2500, 1e-6) #1e-8 step size
w_list_18, stoc_error_list_18 = stochastic_gradient_descent(train_data, 0.000000001, 2500, 1e-6) #1e-9 step size

squared_distances_13 = stoc_calculate_squared_distances(w_list_13, wstar)
squared_distances_14 = stoc_calculate_squared_distances(w_list_14, wstar)
squared_distances_15 = stoc_calculate_squared_distances(w_list_15, wstar)
squared_distances_16 = stoc_calculate_squared_distances(w_list_16, wstar)
squared_distances_17 = stoc_calculate_squared_distances(w_list_17, wstar)
squared_distances_18 = stoc_calculate_squared_distances(w_list_18, wstar)


fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].plot(range(1, len(squared_distances_13) + 1), squared_distances_13)
axs[0, 0].set_title('Squared Distances (1e-4)')
axs[0, 0].set_xlabel('Iteration (t)')
axs[0, 0].set_ylabel('|wt - wML|^2')
axs[0, 0].grid(True)

axs[0, 1].plot(range(1, len(squared_distances_14) + 1), squared_distances_14)
axs[0, 1].set_title('Squared Distances (1e-5)')
axs[0, 1].set_xlabel('Iteration (t)')
axs[0, 1].set_ylabel('|wt - wML|^2')
axs[0, 1].grid(True)

axs[0, 2].plot(range(1, len(squared_distances_15) + 1), squared_distances_15)
axs[0, 2].set_title('Squared Distances (1e-6)')
axs[0, 2].set_xlabel('Iteration (t)')
axs[0, 2].set_ylabel('|wt - wML|^2')
axs[0, 2].grid(True)

axs[1, 0].plot(range(1, len(squared_distances_16) + 1), squared_distances_16)
axs[1, 0].set_title('Squared Distances (1e-7)')
axs[1, 0].set_xlabel('Iteration (t)')
axs[1, 0].set_ylabel('|wt - wML|^2')
axs[1, 0].grid(True)

axs[1, 1].plot(range(1, len(squared_distances_17) + 1), squared_distances_17)
axs[1, 1].set_title('Squared Distances (1e-8)')
axs[1, 1].set_xlabel('Iteration (t)')
axs[1, 1].set_ylabel('|wt - wML|^2')
axs[1, 1].grid(True)

axs[1, 2].plot(range(1, len(squared_distances_18) + 1), squared_distances_18)
axs[1, 2].set_title('Squared Distances (1e-9)')
axs[1, 2].set_xlabel('Iteration (t)')
axs[1, 2].set_ylabel('|wt - wML|^2')
axs[1, 2].grid(True)

plt.tight_layout()
plt.show()


iteration_numbers = range(1, len(stoc_error_list_13) + 1)

plt.figure(figsize=(10, 8))

plt.plot(iteration_numbers, stoc_error_list_13, label='Step Size: 1e-4')
plt.plot(iteration_numbers, stoc_error_list_14, label='Step Size: 1e-5')
plt.plot(iteration_numbers, stoc_error_list_15, label='Step Size: 1e-6')
plt.plot(iteration_numbers, stoc_error_list_16, label='Step Size: 1e-7')
plt.plot(iteration_numbers, stoc_error_list_17, label='Step Size: 1e-8')
plt.plot(iteration_numbers, stoc_error_list_18, label='Step Size: 1e-9')

plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Error vs. Number of Iterations (2500) for Different Step Sizes')
plt.grid(True)
plt.legend()
plt.show()

# list to store the averaged w values
averaged_ws = []

# iterate over all w_list variables
for w_list in [w_list_1, w_list_2, w_list_3, w_list_4, w_list_5, w_list_6, 
               w_list_7, w_list_8, w_list_9, w_list_10, w_list_11, w_list_12,
               w_list_13, w_list_14, w_list_15, w_list_16, w_list_17, w_list_18]:
    averaged_w = np.zeros((100, 1))
    
    for w in w_list:
        averaged_w += w
    
    averaged_w /= len(w_list)
    
    averaged_ws.append(averaged_w)



fig, axs = plt.subplots(1, 1, figsize=(10, 7))

axs.plot(wstar, 'bo-', label='Analytical Solution w*')
axs.plot(averaged_ws[0], 'ro-', label='Step Size: 1e-4')
axs.plot(averaged_ws[1], 'go-', label='Step Size: 1e-5')
axs.plot(averaged_ws[2], 'ko-', label='Step Size: 1e-6')
axs.plot(averaged_ws[3], 'yo-', label='Step Size: 1e-7')
axs.plot(averaged_ws[4], 'mo-', label='Step Size: 1e-8')
axs.plot(averaged_ws[5], 'mo-', label='Step Size: 1e-9')
axs.set_xlabel('Feature Index')
axs.set_ylabel('Weight Value')
axs.set_title('SGD Solutions')
axs.grid(True)
axs.legend()

plt.suptitle('No of Iterations = 100')
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1, 1, figsize=(10, 7))

axs.plot(wstar, 'bo-', label='Analytical Solution w*')
axs.plot(averaged_ws[6], 'ro-', label='Step Size: 1e-4')
axs.plot(averaged_ws[7], 'go-', label='Step Size: 1e-5')
axs.plot(averaged_ws[8], 'ko-', label='Step Size: 1e-6')
axs.plot(averaged_ws[9], 'yo-', label='Step Size: 1e-7')
axs.plot(averaged_ws[10], 'mo-', label='Step Size: 1e-8')
axs.plot(averaged_ws[11], 'mo-', label='Step Size: 1e-9')
axs.set_xlabel('Feature Index')
axs.set_ylabel('Weight Value')
axs.set_title('SGD Solutions')
axs.grid(True)
axs.legend()

plt.suptitle('No of Iterations = 1000')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(10, 7))

axs.plot(wstar, 'bo-', label='Analytical Solution w*')
axs.plot(averaged_ws[12], 'ro-', label='Step Size: 1e-4')
axs.plot(averaged_ws[13], 'go-', label='Step Size: 1e-5')
axs.plot(averaged_ws[14], 'ko-', label='Step Size: 1e-6')
axs.plot(averaged_ws[15], 'yo-', label='Step Size: 1e-7')
axs.plot(averaged_ws[16], 'mo-', label='Step Size: 1e-8')
axs.plot(averaged_ws[17], 'mo-', label='Step Size: 1e-9')
axs.set_xlabel('Feature Index')
axs.set_ylabel('Weight Value')
axs.set_title('SGD Solutions')
axs.grid(True)
axs.legend()

plt.suptitle('No of Iterations = 2500')
plt.tight_layout()
plt.show()

# finding the best result on doing stochastic gradient descent
w_list_111, stoc_error_list_111 = stochastic_gradient_descent(train_data, 0.0001, 40000, 1e-6)

best_w_111 = np.zeros((100,1))
for i in w_list_111:
    best_w_111 += i
best_w_111 /= len(w_list_111)

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

axs[0].plot(wstar, 'bo-', label='Analytical Solution w*')
axs[0].plot(best_w_111, 'ro-', label='Step Size: 1e-4')
axs[0].set_xlabel('Feature Index')
axs[0].set_ylabel('Weight Value')
axs[0].set_title('SGD Solutions (No of Iterations = {})'.format(len(w_list_111)-1))
axs[0].grid(True)
axs[0].legend()

iteration_numbers = range(1, len(stoc_error_list_111) + 1)
opt_line = [opt_error for _ in iteration_numbers]
axs[1].plot(iteration_numbers, stoc_error_list_111, linestyle='-', label='Stochastic Gradient Descent Error')
axs[1].plot(iteration_numbers, opt_line, linestyle='-', label='Optimal Error')
axs[1].set_xlabel('Number of Iterations')
axs[1].set_ylabel('Error')
axs[1].set_title('Error vs. Number of Iterations')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()


squared_distances_111 = stoc_calculate_squared_distances(w_list_111, wstar)
plt.plot(range(len(squared_distances_111)), squared_distances_111, linestyle='-')
plt.xlabel('Iteration (t)')
plt.ylabel('|wt - wML|²')
plt.title('Squared Euclidean Distance from Optimal Weight Vector')
plt.grid(True)
plt.show()

# part iv - ridge regression

# ridge regression gradient descent
def ridge_gradient_descent(train_x_arr, train_y_arr, step_size, lamdas, max_iterations, tolerance, k_fold=5):
    num_features = train_x_arr.shape[0]
    kf = KFold(n_splits=k_fold)               # getting the split of the data into different sets for cross validation
    errors_per_lambda = {}
    for lamda in lamdas:
        mean_error = 0
        errors = []
        for train_index, val_index in kf.split(train_x_arr.T):
            train_x_fold, val_x_fold = train_x_arr[:, train_index], train_x_arr[:, val_index]        # extracting the training set and validation set
            train_y_fold, val_y_fold = train_y_arr[train_index], train_y_arr[val_index]

            w = np.zeros((num_features,1))                                                             # doing gradient descent

            iteration = 1

            while iteration < max_iterations:
                gradient = (2*(np.dot(np.dot(train_x_fold, train_x_fold.T), w).reshape(-1,1) - np.dot(train_x_fold, train_y_fold)))
                w_new = w - step_size * gradient

                mse_w = np.mean((np.dot(train_x_fold.T, w) - train_y_fold) ** 2) + lamda * np.linalg.norm(w) ** 2
                mse_w_new = np.mean((np.dot(train_x_fold.T, w_new) - train_y_fold) ** 2) + lamda * np.linalg.norm(w_new) ** 2
                abs_diff = abs(mse_w_new - mse_w)

                if abs_diff < tolerance:
                    break

                w = w_new
                iteration += 1    

            val_error = np.mean((np.dot(val_x_fold.T, w) - val_y_fold) ** 2) + lamda * np.linalg.norm(w) ** 2
            mean_error += val_error
    
        mean_error /= k_fold
        errors_per_lambda[lamda] = mean_error
    
    return errors_per_lambda

errors_per_lambda_1 = ridge_gradient_descent(train_x_arr, train_y_arr, step_size=0.000001, lamdas=[1, 10, 100, 1000], max_iterations=1000, tolerance=1e-6, k_fold=5)
errors_per_lambda_2 = ridge_gradient_descent(train_x_arr, train_y_arr, step_size=0.000001, lamdas=[0.025, 0.05, 0.075, 0.1], max_iterations=1000, tolerance=1e-6, k_fold=5)
errors_per_lambda_3 = ridge_gradient_descent(train_x_arr, train_y_arr, step_size=0.000001, lamdas=[0.00025, 0.00050, 0.00075, 0.0001], max_iterations=1000, tolerance=1e-6, k_fold=5)
errors_per_lambda_4 = ridge_gradient_descent(train_x_arr, train_y_arr, step_size=0.000001, lamdas=[0.000001, 0.00001], max_iterations=1000, tolerance=1e-6, k_fold=5)


# plotting the results of doing cross calidation with diff values of lambda

lambdas_1 = list(errors_per_lambda_1.keys())
errors_1 = list(errors_per_lambda_1.values())

lambdas_2 = list(errors_per_lambda_2.keys())
errors_2 = list(errors_per_lambda_2.values())

lambdas_3 = list(errors_per_lambda_3.keys())
errors_3 = list(errors_per_lambda_3.values())

lambdas_4 = list(errors_per_lambda_4.keys())
errors_4 = list(errors_per_lambda_4.values())

fig, axs = plt.subplots(2, 2, figsize=(18, 15))

axs[0, 0].plot(lambdas_1, errors_1, marker='x', linestyle='-')
axs[0, 0].axhline(y=opt_error, color='r', linestyle='--', label='Optimal Error')
axs[0, 0].set_xlabel('Lambda')
axs[0, 0].set_ylabel('Error')
axs[0, 0].set_title('Errors per Lambda (Set 1)')
axs[0, 0].grid(True)
axs[0, 0].legend()

axs[0, 1].plot(lambdas_2, errors_2, marker='x', linestyle='-')
axs[0, 1].axhline(y=opt_error, color='r', linestyle='--', label='Optimal Error')
axs[0, 1].set_xlabel('Lambda')
axs[0, 1].set_ylabel('Error')
axs[0, 1].set_title('Errors per Lambda (Set 2)')
axs[0, 1].grid(True)
axs[0, 1].legend()

axs[1, 0].plot(lambdas_3, errors_3, marker='x', linestyle='-')
axs[1, 0].axhline(y=opt_error, color='r', linestyle='--', label='Optimal Error')
axs[1, 0].set_xlabel('Lambda')
axs[1, 0].set_ylabel('Error')
axs[1, 0].set_title('Errors per Lambda (Set 3)')
axs[1, 0].grid(True)
axs[1, 0].legend()

axs[1, 1].plot(lambdas_4, errors_4, marker='x', linestyle='-')
axs[1, 1].axhline(y=opt_error, color='r', linestyle='--', label='Optimal Error')
axs[1, 1].set_xlabel('Lambda')
axs[1, 1].set_ylabel('Error')
axs[1, 1].set_title('Errors per Lambda (Set 4)')
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# best lambda, on checking the error function : the error function is MSE . 1e-5
Rlambda = 0.00001

# analytical solution for ridge regression

identity = np.identity(100)
covariance = np.dot(train_x_arr, train_x_arr.transpose()) + Rlambda * identity
inv_covariance = np.linalg.inv(covariance)
wRstar = np.dot(np.dot(inv_covariance, train_x_arr) , train_y_arr)

# plotting w* and wR*

plt.plot(wstar, 'bo-', label='Analytical Solution w*')
plt.plot(wRstar, 'ro-', label='Analytical Solution w_R*')
plt.xlabel('Feature Index')
plt.ylabel('Weight Value')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# testing on test data

test_y = []
test_x = [[] for _ in range(len(test_data))]

for i in range(len(test_data)):
    test_y.append(test_data[i][100])  
    
    for j in range(len(test_data[0])-1):
        test_x[i].append(test_data[i][j])  

test_x_arr = np.array(test_x).transpose()
test_y_arr = np.array(test_y).reshape(-1,1)

opt_test_error_R = np.mean((np.dot(test_x_arr.T, wRstar) - test_y_arr) ** 2)
opt_test_error = np.mean((np.dot(test_x_arr.T, wstar) - test_y_arr) ** 2)

print(opt_test_error - opt_test_error_R)
