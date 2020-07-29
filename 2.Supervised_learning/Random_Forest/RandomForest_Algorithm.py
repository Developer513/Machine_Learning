# 랜덤 포레스트란 훈련 과정에서 구성한 ㄷ다수의 결정트리로부터 분류 또는 평균예측치를 출력함을써 동작한다
# 결정트리를 엮어서 forest 를 마늚으로써 더 좋은 예측을 하게 만드는 기법
# 랜덤포레스트는 여러개의 결정트리가 랜덤으로 생성된다. 그리고 각자의 방식으로 데이터를 샘플링하여 
# 개별적으로 학습한다. 그리고 최종적으로 voting 을 통해 데이터에 대한 예측을 수행한다. 이러한 방법을 ensemble(앙상블)이라고 한다.

#장점

#    알고리즘이 굉장히 간단하다. Decision Tree를 어떻게 만드는지만 알고 있다면, 이들의 결과를 단순히 voting 등을 통해 종합하기만 하면 되기 때문에 굉장히 쉽게 이해하고 구현할 수 있다.
#    Overfitting이 잘 되지 않는다. 새로운 데이터에 대해 굉장히 잘 generalize되는 편이다.
#    Training이 빠르다. 

#단점

#    Memory 사용량이 굉장히 많다. Decision Tree를 만드는 것 자체가 memory를 많이 사용하는데, 이들을 여러 개 만들어 종합해야 하기 때문에 memory consumption이 많다.
#    training data의 양이 증가해도 급격한 성능의 향상이 일어나지 않는다. 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
# 훈련데이터와 테스트 데이터 분류 
# 데이터를 분류할때는 아래와 같은 모듈을 import 한다.
# train_test_split 함수는 아래와 같은 파라미터를 가진다. 
# from sklearn.model_selection import train_test_split
# train_test_split(arrays(분할시킬데이터셋),
#                  test_size(테스트 데이터셋의 비율이나 갯수(기본값0.25)),
#                  train_size(학습데이터셋의 비율이나 갯수(기본값:test_size 의 나머지)),
#                  random_state(데이터 분할 시 셔플이 이루어지는데 이를 위한 랜덤변수 시드값),
#                  shuffle(셔플여부 결정(기본값:true)),
#                  stratify(지정한 데이터의 비율유지 예를 들어, Label Set인 Y가 25%의 0과 75%의 1로 이루어진 Binary Set일 때, stratify=Y로 설정하면 나누어진 데이터셋들도 0과 1을 각각 25%, 75%로 유지한 채 분할된다.
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)
# 데이터를 무작위로 혼합한 후 training data와 test data 를 일정한비율로분리
n_feature = cancer.data.shape[1]
print(cancer.data.shape) # 데이터셋의 모양(행과 열을 출력함 )
print(cancer.data.shape[1])
print(cancer.DESCR)
# 레코드 수(569), 컬럼개수(30)
# data.shape 는 데이터의 차원을 확인할 수 있다. 
# data.shape[0] 데이터의 행을 구할 때 
# data.shape[1] 데이터의 열을 구할 때 
score_n_tr_est = []
score_n_te_est = []
score_m_tr_mft = []
score_m_te_mft = []
# 1~ 31 열의 개수만큼 30번 반복
for i in np.arange(1, n_feature+1): # n_estimators와 mat_features는 모두 0보다 큰 정수여야 하므로 1부터 시작합니다.

    params_n = {'n_estimators':i, 'max_features':'auto', 'n_jobs':-1} # **kwargs parameter
    params_m = {'n_estimators':10, 'max_features':i, 'n_jobs':-1}
# RadomforestClassifier(n_estimators(생성할 트리의 수(기본값:100)),
#                       min_samples_split(노드를 분할하기 위한 최소한의 샘플 데이터 수 
#  만일 10이라면 노드의 샘플 갯수가 10이상이 되어야 분기를 한다(기본값:2)),
#                       min_samples_leaf(리프노드가 되기 위해 필요한 최소한의 샘플데이터 수),
#                       max_features(분기를 할 때  고려하는 2라면 속성을 랜덤하게 2개고른ㄷ feature개수(기본값: auto)),
#                       max_depth(트리의 최대깊이(기본값:완벽하게 클래스 값이 결정될 때 까지 분할),
#                       max_leaf_nodes(리프노드 최대개수)))
#                       n_jobs(병렬처리되는 스레드 갯수,(1이면 하나의 코어를 100% 사용, -1이면 cpu 전부100% 사용))
    forest_n = RandomForestClassifier(**params_n).fit(x_train, y_train)
    forest_m = RandomForestClassifier(**params_m).fit(x_train, y_train)

    score_n_tr = forest_n.score(x_train, y_train)
    score_n_te = forest_n.score(x_test, y_test)
    score_m_tr = forest_m.score(x_train, y_train)
    score_m_te = forest_m.score(x_test, y_test)

    score_n_tr_est.append(score_n_tr)
    score_n_te_est.append(score_n_te)
    score_m_tr_mft.append(score_m_tr)
    score_m_te_mft.append(score_m_te)

index = np.arange(len(score_n_tr_est))
plt.plot(index, score_n_tr_est, label='n_estimators train score', color='lightblue', ls='--') # ls: linestyle
plt.plot(index, score_m_tr_mft, label='max_features train score', color='orange', ls='--')
plt.plot(index, score_n_te_est, label='n_estimators test score', color='lightblue')
plt.plot(index, score_m_te_mft, label='max_features test score', color='orange')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
           ncol=2, fancybox=True, shadow=False) # fancybox: 박스모양, shadow: 그림자
plt.xlabel('number of parameter', size=15)
plt.ylabel('score', size=15)
plt.show()
index2 = np.arange(n_feature)
forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
forest.fit(x_train, y_train)
plt.barh(index2, forest.feature_importances_, align='center')
plt.yticks(index2, cancer.feature_names)
plt.ylim(-1, n_feature)
plt.xlabel('feature importance', size=15)
plt.ylabel('feature', size=15)
plt.show()