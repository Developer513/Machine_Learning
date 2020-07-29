#의사결정트리는 전형적인 분류모델이며 매우 직관적인 방법 중 하나이다. 
# 다른 모델들과느 다르게 결과물이 시각적을 얽히기 쉬운 형태로 나타나는 것이 장점,
# 전문가 시스템의 모델이 되기도 한다
# 신용평가, 독버섯 분류, 고장진단 시스템, 등 실질적으로 분류하는 경우에 자주사용함
# 의사결정트리는 특정 조건에 의해 분기(노드)가 나누어져 최종 결정(리프노드)에 이르는데 
# 어떤 노드를 가장 위에 놓아야 하는것을 고르는데 정보획득량과 엔트로피라는개념이 필요하다.
# 정보획득량이란 어떤사건이 얼마만큼의 정보를 줄 수 있는지를 수치화한 값이다.
# 정보함수는 정보의 가치를 반환하는데 발생할 확률이 작은 사건일수록 정보의 가치가 크고
# 반대로 발생할 확률이 큰 사건일수록 정보의 가치가 작다. 
# 엔트로피는 무질서도를 정량화해서 표현한 값이다. 
# 어떤집합의 엔트로피가 높을수록 그 집단의 특징을 찾는것이 어렵다.
# 따라서 의사결정트리의 잎 노드들의 엔트로피가 최소가 되는 방향으로 분류해 나가는 갓이 최적의 방법으로 분류한 것이라 할 수 있다.



from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt
# sklearn.metrics 는 모델평가에 사요되는 모듈이다.
# classification_report 는 주요 분류측정항목을 보여주는 보고서 모듈이다.
# confusion_matrics 는 분류의 정확성을 평가하기 위한 오차행렬 계산 모듈이다.
from sklearn.model_selection import train_test_split
# sklearn.model_section은 클래스를 나눌 때,, 그리고 함수를 통해 train/test
# 셋을 나눌 때  모델 검증에 사용되는 서브 패키지이다. 
# train_test_split은 배열 또는 행렬을 임의의 훈련 및 테스트 셋으로 분할하는 모듈이다
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 
# sklearn.tree scikit-learn 패키지 중  분류 및 회귀를 위한 의사결정 트리기반 모델이 있는 서브 패킺
# DecisionTreeClassifire 는 의사결정 트리 분류 모듈
from IPython.display import Image
# IPython.display 는 IPython 내에 정보를 보여주는 도구용도의 공용 API
# API모듈 중 Image 는 원시 데이터가 있는 png, jpec 등의 이미지 객체를 만드는 모듈
import numpy as np
import pandas as pd
# 판다는 데이터를 구조화된 형식으로 가공 및 분석할 수 있도록 자료구조를 제공하는 패키지
import pydotplus
# pydotplus는 그래프를 생성하는 graphviz 의 dot 언어를 파이썬 인터페이스를 제공하는 모듈
import os
# 운영체제와 상호작용하기 위한 기본적인 기능이 제공되는 모듈 
if __name__ == '__main__':
    iris = load_iris()
    x = iris.data[:, [2,3]]
    y = iris.target
    # 학습데이터와 테스트 데이터 분리 
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)
    ml = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    ml.fit(x_train_std,y_train)
    y_pred = ml.predict(x_test_std)
    print('총 테스트 개수:%d, 오류개수:%d' %(len(y_test), (y_test != y_pred).sum()))
    print('정확도: %.2f' %accuracy_score(y_test, y_pred))
    x_combined_std = np.vstack((x_train_std, x_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X=x_combined_std,y=y_combined, clf=ml,
                        res = range(105, 150))
    plt.show()
