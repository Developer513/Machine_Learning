from sklearn.datasets import load_iris
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy  as np

data = load_iris()
# 아이리스 데이터셋 로드 
print(data,'\n')
# 'data':array([[]]) 2차원 배열 형태의 실수들이 저장되어 있다. 
print(data.keys())
