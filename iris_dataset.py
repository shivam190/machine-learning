# importing packages
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# predefined style in matplotlib
plt.style.use('ggplot')

# this link provides documentation
iris = datasets.load_iris()

# type of iris dataset
type(iris)

# number of rows and columns in iris dataset
print(iris.keys())

# data in iris
type(iris.data)

# shape of iris data set
iris.data.shape

# species of iris dataset
iris.target_names

X = iris.data

# response in iris dataset
y = iris.target

# creating dataframe using iris featur_names
df = pd.DataFrame(X, columns=iris.feature_names)

# plotting scatter matrix of iris
gh = pd.plotting.scatter_matrix(df, c=y, figsize= [8,8], s=100, marker = 'D')

# show the graph
plt.show()
