import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
# pylab 은 matplotlib의 서브패키지로
# matplotlib의 수치해석 시각화 며여을 그대로 사용할 수 있도록
# API를 포장한 명령어 집합을 제공한다. 

plt.title("plot") # 표 타이틀 설정 
plt.plot([1,4,9,16],[1,2,3,4], c="b",lw=5, ls="--", marker = "s",ms =15, mec ="g", mew=5, mfc = "r")# 출력할 데이터 리스트 
#plot  c = 선색깔 , lw = 선 굵기, ls= 선 스타일, maker= 마커종류, ms = 마커크기, mec =마커 선 색깔, mew = 마커선 굵기, mfc=  마커매부 색깔 
# 데ㅇ이터의 위치를 나타내는 기호를 마커라고 한다. 마커의 종류는 검색해 볼 것

font1 = {'family': 'NanumMyeongjo', 'size':24,'color': 'black'}
# 각 축의 이름을 정할수도 있다. 
# 축의 이름에 대한 텍스트 설정을 정할때 위와같이 미리 딕셔너리에 넣어서 
# fontdict이라는 파라미터로 추가한다. 
plt.xlabel("엑스 축",fontdict = font1)
plt.ylabel('와이 축', fontdict=font1)
plt.show()#  시각화 명령을 실제 차트로 렌더링 


X = np.linspace(-np.pi, np.pi, 256)
# x 축 범위설정 -원주율 ~ 원주율을 256 단계로 쪼개서 설정  
c = np.cos(X) # x의 범위에 대한 코사인값 
plt.title("x축과 y축의 tick label 설정",fontdict=font1)
plt.plot(X,c)
plt.xticks([-np.pi,-np.pi / 2,0, np.pi / 2, np.pi],
           [r'$-\pi', r'$-\pi/2$',r'$0$',r'$+\pi/2$',r'$+\pi$'])
           # $$ 사이에 latex 표기법에 의해 수식도 넣을 수 있다. 
# x축상에서 위치표시지점
plt.yticks([-0.2,0,0.5])
# 위에서 지정한x틱 y틱 교차지점에 그래프에 표시된다.
plt.show()
# 그래프 위에 그리드 표시 
t = np.arange(0.,5.,0.2)# 시작, 끝, 스텝 
plt.title("라인 플롯에서 여러개의 선 그리기",fontdict=font1)
plt.plot(t,t,'r--', t,0.5*t**2,'bs:', t,0.2*t**3,'g^-')
# x축, y축 t , 붉은 점선으로 표시, x축 t y축 t제곱*0.5 파란
plt.show()

X = np.arange(2)
Y = np.random.randint(0,20,20)
S = np.abs(np.random.randn(20))*100
C = np.random.randint(0,20,20)

scatter = plt.scatter(X,Y,s=S,c=C, label='A')
plt.xlim(X[0]-1,X[-1]+1)
plt.ylim(np.min(Y-1),np.max(Y+1))

plt.title('scatter',pad=10)
plt.xlabel('X axis',labelpad=10)
plt.ylabel('Y axis',labelpad=10)
plt.xticks(np.linspace)
