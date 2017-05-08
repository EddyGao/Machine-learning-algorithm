from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3)#split the train_datasets and test_datesets

knn = KNeighborsClassifier()
knn.fit(X_train , y_train)

print knn.predict(X_test)
print y_test# compared
