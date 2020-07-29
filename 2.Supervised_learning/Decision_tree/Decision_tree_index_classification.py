from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image

iris = datasets.load_iris()
#sklearn.dataset의 load_iris()가 반환하는 객체는 Bunch 객체인데,
#  이는 파이썬의 Dictionary와 거의 유사합니다
#iris = load_iris() 에서 iris는 이 Bunch 객체이고, 이 중에 iris.data는
#  피처가 이미 Numpy형태로 데이터가 들어가 있습니다.
#  iris.target 역시 레이블(예측하려는 타겟)이 Numpy 형태로 들어가 있으므로 
# 이것을 그대로 가져와서 학습과 예측에 사용하게 됩니다.
# 피처는 데이터셋의 일반속성이다.
# 레이블, 클래스, 타겟, 결정 (다 같은말)
# 타겟값 또는 결정값은 지도학습시 데이터의 학습을 위해 주어지는 정답 데이터
# 지도학습 중 분류의 겨우에는 이 결정값을 레이블 또는 클래스라고 한다. 
# 아이리스의 피처(품종을 가르는 특징)는 sepal length, sepal width, petal length, patal width
# 아이리스 데이터 품종(레이블) setosa, virginica, 
# 분류(classification)는 피처데이터의 패턴을 인식해서 패턴의 결정값이 무엇인지 학습한다. 
# 지도학습은 명확한 정답이 주어진 데이터를 먼저 학습한 뒤 미지의 데이터셋의 예측한다. 
# 데이터셋 분리(학습용,테스트용) -> 학습용데이터를 이용하여 모델학습-> 예측수행-> 예측 결과값과 실제 결과값 비교 해서 모델 성능 평가
X = iris.data[:,[2,3]]
print(X)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3,
random_state = 0)

sc = StandardScaler()
sc.fit(X_train)

x_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

iris_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3,
random_state = 0)
iris_tree.fit(X_train, y_train)


y_pred_tr = iris_tree.predict(X_test)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_tr))



dot_data = export_graphviz(iris_tree, out_file=None, feature_names=['petal length', 'petal width'],
                          class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png("iris.png")