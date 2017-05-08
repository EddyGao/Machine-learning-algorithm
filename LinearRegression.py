from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X , y = datasets.make_regression(n_samples=100 , n_features=1 , n_targets=1 , noise=2.5)#make datasets for myself
plt.scatter(X , y , color = 'red')

model = LinearRegression()
model.fit(X , y)

plt.plot(X , model.predict(X))
plt.show()
