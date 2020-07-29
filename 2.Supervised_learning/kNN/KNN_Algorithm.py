# 최근접 이웃법 
# 유유상종 이론 
# 새로운 데이터를 입력 받았을 때 가장 가까이 있는것이 무엇이냐를 중심으로 새로운 데이터의ㅣ
# 종류를 정해주는 알고리즘이다. 
# 단순히 주변에 무엇이 가장 가까이 있는가를 보고 판단하는것이 아닌
# 주변에 있는 몇개를 같이봐서 가장 많은것으로 판단한다.
# knn에 대해 알기 전 분류(classification)와 군집화(clustering)의 차이점에 대해 알 필요가 있다.
# 분류는 소속집단의 정보를 이미 알고 있는 상태에서 비슷한 집단으로 묶는 방법이다.
# 선별기준(label)이 있는 상태에서 데이터를 나눈다(지도학습의 일종)
# 군집화는 소속집단의 정보가 없고 모르는 상태에서 비슷한 집단으로 묶는다
# 선별기준(label)없이 데이터를 군집한다. (비지도학습의 일종)

# knn의 기본적인 로직으로는 새로운 샘플에서 가까운 거리에 있는 몇가지 라벨을 함께 본다. 
# 그리고 가장 빈도가 높(가까운 데이터가 많)은 것을 통해 분류한다. 
# 분류를 할때 새로운 샘플과 가까운 데이터(라벨)가 몇개인지에 따라서 분류되는것이 다를 수 있다. 
# 여기서 kNN 알고리즘의 뜻을 알아보면 
# k 개의
# Nearest 가장 가까운
# Neighbor 이웃 
# 이라는 뜻이다. 
# 어떤 임의의 새 데이터 t 가 있을 때 기존의 데이터들 중에서 t까지의 거리가 가장 가까운 
# k 개의 데이터를 가까운 순서대로 선택하여 그 선택받은 데이터들의 라벨에 따라 t의 라벨이 결정된다. 
# 따라서 k를 어떻게 정의하는가가 중요하다. 
# 하이퍼 파라미터란 일반적으로 어떤 임의의 모델을 학습시킬 때 사람이 직접 설정해주어야 하는 변수를 의미한다.
# kNN 모델에서 사람이 직접 설정해 주어야 하는 변수는 크게
# 데이터간 거리를 나타낼 기준 Distance 
# 모델에서 지정할 K(새로운 데이터와 거리를 측정할 label 데이터의 개수)
# 거리를 구하는 방법으로 가장 많이 사용하는 두가지는 
# 유클리디안 거리, 맨하탄 거리이다. 
# Euclidean Distance
# n차원에서 두 점사이의 거리를 구하는 공식
# k값을 어떻게 설정하느냐에 따라 결과가 달라지는데 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15
iris = datasets.load_iris()

X = iris.data[:,:2]
y = iris.target
h = .02

cmap_light = ListedColormap(['orange','cyan','cornflowerblue'])
cmap_bold = ListedColormap(['darkorange','c','darkblue'])

for weights in['uniform','distance']:
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights = weights)
    clf.fit(X,y)
    x_min, x_max = X[:,0].min() -1, X[:,0].max()+1
    y_min, y_max = X[:,1].min() -1, X[:,1].max()+1
    xx,yy = np.meshgrid(np.arange(x_min, x_max,h),
                        np.arange(y_min, y_max,h))
    z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    z = z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx,yy,z,cmap=cmap_light)
    plt.scatter(X[:,0],X[:,1],c=y, cmap = cmap_bold,edgecolors='k',s=20)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class clasifiction (k=%i,weights='%s')" %(n_neighbors,weights))
plt.show()