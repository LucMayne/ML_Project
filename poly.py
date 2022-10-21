import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Data sets for cancer cases in men according to age in the UK per 100,000 population

# Training set
x_train = [[25], [35], [45], [55], [60], [65], [70]]
y_train = [[1062], [1872], [4832], [14664], [20399], [30121], [34179]]

# Testing set
x_test = [[75], [80], [85]] 
y_test = [[29378], [22306], [13363]]

ls = np.linspace(20, 75, 100)

# The degree for the polynomial regression model, 4 looks better on the graph than 2
poly_feat = PolynomialFeatures(degree=4)

# Transforms into a matrix of the given degree
x_train_quad = poly_feat.fit_transform(x_train)
x_test_quad = poly_feat.transform(x_test)

# Train and test the model
quadratic_regressor = LinearRegression()
quadratic_regressor.fit(x_train_quad, y_train)
quadratic = poly_feat.transform(ls.reshape(ls.shape[0], 1))

# line displayed on graph in blue
plt.plot(ls, quadratic_regressor.predict(quadratic), c='blue', linestyle='dashed')

# text displayed on graph
plt.title("Men affected by Cancer")
plt.xlabel('Age')
plt.ylabel('Incidence Rate per 100,000')

# points displayed on graph in red
plt.scatter(x_train, y_train, c='red')

# define the axis
plt.axis([20, 75, 0, 38000])
plt.show()