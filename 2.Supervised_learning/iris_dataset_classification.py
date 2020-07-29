from sklearn.datasets import load_iris
# scikit-learn 내부 iris 데이터셋 로드 
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy  as np
#import csv
#f = open('')

iris = load_iris() # sample dataset load
# load_iris() 가 반환한 객체는 파이썬의 딕셔너리와 유사한 객체이다.
print('iris feature',iris.data)
# data 에는 각 꽃의 특성(feature)가 담겨있다. 
print(iris.keys)
print("iris_dataset의 키: \n{}".format(iris.keys()))
features = iris.data
features_names = iris.feature_names
target = iris.target 
target_names = iris.target_names

x_index = 0
y_index = 1

formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
# 람다식
plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()
plt.show()

