from sklearn import svm

# X = [[0,0] , [1,1] , [1,0]]
# y = [0,1,1]
#
# clf = svm.SVC()#classify
# clf.fit(X , y)
#
# print clf.predict([2,2])

X = [[0,0] , [1,1]]
y = [0.5 , 1.5]
clf = svm.SVR()#regression
clf.fit(X , y)

print clf.predict([2,2])
