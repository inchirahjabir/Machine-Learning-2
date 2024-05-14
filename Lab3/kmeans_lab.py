
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read the input data
data = pd.read_csv("./data_kmeans.txt", sep=" ", header=None)
X = pd.DataFrame(data).to_numpy()

# hyper-parameters
np.random.seed(seed=3)
K = 3
ITERA_MAX = 20
I = X.shape[0]

# find the bounds of the data coordinates
x1_min = np.min(X[:,0])
x1_max = np.max(X[:,0])
x2_min = np.min(X[:,1])
x2_max = np.max(X[:,1])


# plot data
plt.figure()
plt.plot(X[:,0],X[:,1],'ko')
plt.xlabel('x1')
plt.ylabel('x2')

# initialize randomly cluster centroids
mu = np.random.rand(K,2) * np.array([[x1_max - x1_min, x2_max - x2_min]]) + np.array([[x1_min, x2_min]])

# repeat until convergence
for I in range (ITERA_MAX):
    for K in range (ITERA_MAX):
        d = np.sqrt(((X - mu[:, np.newaxis])**2).sum(axis=2))
        y_pred = np.argmin(d, axis=0)

        # Update centroids by choosing the data point closest to the current centroid
        new_mu = np.array([X[y_pred == k].min(axis=0) for k in range(K)])

        # Check for convergence
        if np.allclose(mu, new_mu):
            break

        # Update centroids
        mu = new_mu

# test your model with some new data
data_test = np.random.rand(50, 2) * np.array([[x1_max - x1_min, x2_max - x2_min]]) + np.array([[x1_min, x2_min]])

# predict clusters for the new test data
d_test = np.sqrt(((data_test - mu[:, np.newaxis])**2).sum(axis=2))
y_test_pred = np.argmin(d_test, axis=0)

# Plot both training and test results in a 2D graph
plt.figure()
for k in range(K):
    plt.plot(X[y_pred == k][:, 0], X[y_pred == k][:, 1], 'o', label=f'Training Cluster {k+1}')
    plt.plot(data_test[y_test_pred == k][:, 0], data_test[y_test_pred == k][:, 1], 's', label=f'Test Cluster {k+1}')
plt.plot(mu[:, 0], mu[:, 1], 'rs', markersize=10, label='Final Centroids')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Training and Test Results')
plt.legend()
plt.show()