import numpy
import numpy as np

a = np.array([1,4,5,8], int)
# 라이브러린 내 배열의 원소와 그 자료형을 선언함 
print(a)
print(type(a))
# 배열 a의 타입 조회 
# numpy.ndarray
print(a[:2])
# 2번째 인덱스 부터 출력
a = np.array([[1,2,3], [4,5,6]],float)
# 2차원 배열 선언 
print(a[1,:])
# 1번째 행 전부 출력
# 456
print(a[:,2])
# 2번째 열 전부 출력
# 36
print(a[-1:,-2:])
# -1번째 행, -2번째열 시작으로 전부 출력 
# 56
print(a.shape)
# 배열 차원의 크기를 튜플로 출력
# (2,3) 2행 3열 
print(len(a))
# 배열의 길이 출력 2
a = np.array(range(5),float)
print(a)
# 1차원 배열을 선언하는데 여러 방법이 있다.
a = np.array([1,2,3],float)
b=a
c=a.copy()
a[0] = 0
print(a)
print(b)
print(c) # c는 a의 주소를 참조하는게 아니라 
# 직접 데이터를 복사하여 저장한다
# call by value
a = np.array([1,2,3],float)
# 배열을 리스트로 변환
print(a.tolist())
print(list(a))
a.tostring()
# 문자열로도 변환 할 수 있다.
a = np.array([[1,2,3],[4,5,6]],float)
print(a)
# 2차원 배열을 1줄로 출력할 때
print(a.flatten())
# 서로 다른 배열을 이어 붙일수도 있다.
a = np.array([1,2],float)
b = np.array([3,4,5,6],float)
c = np.array([7,8,9],float)
print(np.concatenate((a,b,c)))
# conatenate 를 사용할 때는 각 배열을 
# 튜플 형태로 입력한다.
a = np.array([[1,2],[3,4]],float)
b = np.array([[5,6],[7,8]],float)
print(np.concatenate((a,b)))
# 2차원 배열을 이어붙일수도 있다 
print(np.concatenate((a,b), axis = 0))
# axis=0 이면 한 행에 한 원소씩 출력한다.
print(np.concatenate((a,b), axis = 1))
# 1이면 한행에 하나의 배열을 출력한다.
print(np.ones((2,3), dtype = float))
# np.ones((row,col), dtype = int)
# 행과 열을 그리고 데이터 타입을 입력받아서 
# ones 는 배열내부를 전부 1로 채운다. 
print(np.ones((2,3), dtype = float))
# np.zeros(2,3) 이면 0으로 채운다  
a = np.array([[1,2,3],[4,5,6]])
print(np.ones_like(a))
print(np.zeros_like(a))
# 이미 선언되어있는 배열을 1로 변경하여 출력함
print(np.identity(4, dtype=float))
# np.identity(행열의 수, 데이터 타입)
# 단위행열 생성
print(np.eye(4,k=1, dtype = float))

