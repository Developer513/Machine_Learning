from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pandas as pd

iris = load_iris()
# data는 아이리스 데이터셋에서 피처데이터를 가진다.
iris_data = iris.data
# iris.target은 아이리스 데이터 셋에서 레이블(결정값) 데이터를 가진다. 
iris_label = iris.target
print('iris  target값: ', iris_label)
print('iris target 명:', iris.target_names)

iris_df = pd.DataFrame(data = iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
print(iris_df.head(3)) # 각 품종(레이블)에 해당하는 피쳐값
# 학습데이터와 테스트 데이터 분할 전테 테스트 데이터는 데이터의 20퍼센트
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size = 0.2, random_state =11)
# 결정트리 분류기 호출(학습용데이터 랜덤 시드값 11)
dt_clf = DecisionTreeClassifier(random_state=11)
# fit으로 훈련데이터 학습
dt_clf.fit(X_train, y_train)
# 피처 테스트셋으로 학습된 모델로 예측 
pred = dt_clf.predict(X_test)
print(pred)
# 예측정확도 측정
from sklearn.metrics import accuracy_score
# accuracy_score(y_test,pred) 로 예측값과 실제값 대조 
print('예측정확도 : {0:.4f}',format(accuracy_score(y_test,pred)))