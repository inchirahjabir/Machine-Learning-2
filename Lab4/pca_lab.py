
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read the input data
data = pd.read_csv("./data_pca.txt", sep=" ", header=None)
X = pd.DataFrame(data).to_numpy()

plt.figure(1)
plt.plot(X[:,0],X[:,1],'ko')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')







