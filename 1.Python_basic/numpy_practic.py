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
a= np.array([1,2,3], float)
s = a.tostring()
print(np.fromstring(s))
# 행렬 연산 (열이 맞아야 함)
a = np.array([[2,3], [3,4]], float)
b = np.array([[2,0], [1,3]], float)
print(a*b)

a =  np.array([1,4,9], float)
print(np.sqrt(a))  # 제곱근

a = np.array([1.1, 1.5, 1.9], float)
print(np.floor(a)) # 내림
print(np.ceil(a))  # 올림
print(np.rint(a))  # 반올림 
# 원주율
print(np.pi)
# 자연상수
print(np.e)

a = np.array([1,4,5], int)
for x in a: # x 이터레이터 x 배열을 돌면서  배열 출력 
    print(x)
a = np.array([[1,2], [3,4], [5,6]],float)
for x in a:
    print(x)    
# 배열의 행을 출력할 때 
a = np.array([[1,2],[3,4],[5,6]],float)
for x,y in a:
    print(x*y)
# 배열행만 곱해서 출력할 수도 있다. 
a = np.array([2,4,3],float)
print(a.sum())# 배열원소 합계
print(a.prod())# 배열원소 곱
print(np.sum(a))# 다른형식으로 표현가능 
print(np.prod(a))

a = np.array([2,1,9],float)
print(a.mean()) # 평균
print(a.var())  # 분산
print(a.std())  # 표준편차
print(a.min())
print(a.max())
print(a.argmin())
print(a.argmax())

a = np.array([6,2,5,-1,9],float)
print(sorted(a)) # 정렬
a.sort() # 정수형태로 정렬 
print(a)

a = np.array([1,3,0],float)
b = np.array([0,3,2],float)
print(a>b)# 배열끼리 비교도 가능 각 원소들 끼리 비교하여 
# 참, 거짓으로 리턴 
print(a==b)
print(a<=b)
c = a > b
print(a)
print(a>2)

c = np.array([True, False, False],bool)
print(any(c)) # 불린 형태의 배열 OR 연산
print(all(c)) # 불린 형태의 배열 AND 연산 
a = np.array([1,3,0],float)
# 논리연산을 수행하는데 괄호안의 조건으로 배열의 각 요소를 비교하여 결과값을 배열형태로 반환 
print(np.logical_and(a>0,a<3))
b = np.array([True, False, True],bool)
print(np.logical_not(b))
c = np.array([False, True, False],bool)
print(np.logical_or(b,c))
# where 는 배열항목이 주어진 조건을 만족하는 항목을 가저옴
a = np.array([1,3,0],float)
print(np.where(a!=0,1/a,a))
print(np.where(a>0,3,2))# 뭔지 잘 모르겠음 

a = np.array([[0,1],[3,0]], float)
print(a.nonzero())# 행렬이 0이아닌 부분은 1 

a = np.array([1, np.NaN, np.Inf],float)
print(a)
print(np.isnan(a)) # isnan 가 숫자인지 판별 
print(np.isfinite(a))

a = np.array([[6,4],[5,9]],float)
print(a>=6)
print(a[a>=6])
sel = (a>=6)
print(a[sel])
a[np.logical_and(a>5,a<9)]
print(np.array([6.])) 

a = np.array([[0,1],[2,3]],float)
b = np.array([2,3],float)
c = np.array([[1,1],[4,0]],float)
print(a)
# 두 배열의 내적 곱 
print(np.dot(b,a))
# 두 배열의 행렬 곱
print(np.matmul(b,a))
print('------------')
print(np.dot(a,b))
print(np.dot(a,c))
print(np.dot(c,a))

a = np.array([[4,2,0,1],[9,3,7,4],[1,2,1,5],[1,2,3,4]],float)
print(a)
# 행렬식
# 정사각행렬에 수를 대응시키는 함수
print(np.linalg.det(a)) # linalg 는 선형대수 함수를 가지고 있음
# 행렬 벡터의 고유값 
vals, vecs = np.linalg.eig(a)
print(vals)
print(vecs)
a = np.array([[1,3,4], [5,2,3]],float)
# 특이값 분해 
U,s,Vh = np.linalg.svd(a)
print(U)
print(s)
print(Vh)
print(np.poly([-1,1,1,10]))
print(np.roots([1,4,-2,3]))
print(np.polyder([1./4.,1./3.,1./2.,1.,0.]))
print(np.polyval([1,-2,0,2],4))
# 피어슨 상관관계 계수 
a = np.array([[1,2,1,3],[5,3,1,8]],float)
c = np.corrcoef(a)
print(c)
# 공분산111
print(np.cov(a))
# 정규분포 
print(np.random.normal(1.5,4.0))
print(np.random.normal())
print(np.random.normal(size = 5))