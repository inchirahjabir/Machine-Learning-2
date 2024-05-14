import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

name_file = './data_ffnn_3classes.txt'
columns = ['x1','x2','y']
data = pd.read_csv(name_file, names=columns,sep='\t',header=0)
data_np = pd.DataFrame(data).to_numpy()
#data = pd.read_csv(name_file, names=columns,sep='\t')
x1 = np.asarray(data['x1'])
x2 = np.asarray(data['x2'])
y = np.asarray(data['y'])

I = data_np.shape[0]
N = 2
J = 3 
K = 3 # number of hidden neurons; you can change this value
itera_max = 1000 # you can change this value
rate_learning = 1e-2 # you can change this value

X = data_np[:,0:N]
y = data_np[:,N]

Y = np.zeros((I,J))
for j in range(J):
    Y[y==j,j] = 1

plt.figure()
plt.plot(X[y==0,0],X[y==0,1],'ro')
plt.plot(X[y==1,0],X[y==1,1],'go')
plt.plot(X[y==2,0],X[y==2,1],'bo')
plt.show()


# Some code that may be helpful to work on the lab work
# At some point if you want to concatenate two vectors or matrices, you can do something like: 
#c = np.concatenate((a,b),axis=1)

# If you want to scalar product (a.k.a. dot product) of two vectors: 
# c = np.dot(a,b)

# If you want to transpose a vector: 
# a_t = np.transpose(a)

# If you want to invert a matrix: 
# a_inv = np.linalg.inv(a)

# If you want to invert a matrix elementwisely: 
# a_inv_element = a**(-1)

# If you want to create a vector of size 10 with value one: 
# a = np.ones((10,1))

# If you want to create a vector of size 10 with value zero: 
# a = np.zeros((10,1))

# Find X bar
X_bar = np.concatenate((X,np.ones((I,1))),axis=1)

# Initialize error, threshold and iteration
delta_E = 1e5
threshold = 1e-2
itera = 0

# Initialize randomly V and W
V = np.random.randn(N+1, K)
W = np.random.randn(K+1, J) 

# Do Forward Propagation

# Find X_barbar
X_barbar = np.dot(X_bar,V)
# Find F
F = 1 / (1 + np.exp(-X_barbar))
# Find F_bar
F_bar = np.concatenate((F,np.ones((I,1))),axis=1)
# Find F_barbar
F_barbar = np.dot(F_bar,W)
# Find G
G = 1 / (1 + np.exp(-F_barbar))
# Find E
E = 0.5 * np.sum((G - Y)**2)

# Do Back Propagation
#
gradient_error_G = G * (1 - G) * (G - Y) 
gradient_error_W = G * (1 - G) * (G - Y) 

# Repeat until convergence
while(np.abs(delta_E) >= threshold and itera<itera_max):
    # Do Forward Propagation
    X_barbar = np.dot(X_bar,V)
    F = 1 / (1 + np.exp(-X_barbar))
    F_bar = np.concatenate((F,np.ones((I,1))),axis=1)
    F_barbar = np.dot(F_bar,W)
    G = 1 / (1 + np.exp(-F_barbar))

    # Find the new value of error
    E_next = 0.5 * np.sum((G - Y)**2)

    # Find the new delta E
    delta_E = E_next - E

    # Update E
    E = E_next
    
    # Update iteration
    itera += 1

    # Print the iterations and the errors associated with each iteration
    print(f"Iteration {itera}: Error = {E}")

