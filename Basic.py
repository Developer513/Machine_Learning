# python data structure
# 파이썬을 쓰는 이유, 개발속도가 빠름 자료구조, 문자열 처리 등이 언어레벨로 구현되어 있음 
# 리스트, 튜플 = 스퀀스 구조 = 순서를 가짐 
# 리스트(수정가능), 튜플(수정불가)
# 리스트 표현방식 
# a = [] #a라는 빈공간의 리스트 생성 
# empty_list = list() # list 함수로 리스트 생성 리스트가 아닌자료를 리스트로 변환 가능 
# list('cat') # [c,a,t]
# 문자열을 쪼개는 split은 쪼갠 문자열을 리스트 형태로 반환한다.
# a = 'she is nice'
# a.split(' ') # [she, is, nice]
# 리스트 오프셋은 해당범위에 따라 리스트로 반환한다. 하나를 출력해도 리스트로 반환한다.
# a=['apple','banana','cat']
# a[0:1] # [apple]
# 파이썬에서는 하나의 리스트안에 자료형을 구분하지 않고 넣을 수 있다. 심지어 리스트 원소로 리스트, 튜플 등 자료구조도 넣을 수 있다. 
# apppend() 리스트 끝에 새로운 요소 추가
# insert(인덱스,자료)
# del 리스트명[인덱스]
# 튜플 : 리스트와 동일하지만 요소 추가,수정, 삭제가 불가능하다. 상수배열로 사용하기 좋ㄷ다.
# 튜플은 리스트에 비해 메모리공간을 덜 사용한다(추가,수정,삭제가 불가능하기 때문)
# 딕셔너리 : 리스트와 비슷함, 값에 상응하는 고유의 키값이 다름, 딕셔너리는 순서가 따로 존재하지 않는다. 해시맵이네 
# 파이썬 이터레이터: 복수개의 데이터가 들어있는 
##################################################################################################################################
# 조건문 반복문 숙달 연습문제 start를 1씩 증가시켜 guess_me 와 매칭되면 found it! 을 출력하고 종료 
guess_me = 7                # 변수선언
start = 1
while guess_me >= start:    # 7번 반복
    if start < guess_me:    # start가 guess_me 보다 작을 때 
        print('too low')
    elif start > guess_me:  # start가 guess_me 보다 클 때
        print('Oops')
        break               # 루프 탈출
    else:                   # 두 변수 값이 같을 ㄸ 
        print('found it!')
        break               # 루프탈출
    start+=1                # start 값 1 증가 
# 반복자는 포인터와 개념이 유사하다 컨테이너와 알고리즘이 하나로 동작하도록 해주는데 알고리즘이 컨테이너의 값을 간편하게 참조 할 수 있다.
##################################################################################################################################
# for loop iterator practice examples
num_list = [1,2,3,4]        # make 'num_list' that is list containing number 
for number in num_list:     # set iterator 'number' in num_list. iterator change the value to accending order num_list offset while looping
    print(number)           # print iterator value

# using for loop iterator at dictionary 
accusation = {'room': 'ballroom', 'weapon': 'leap pipe','person': 'Col. Mustard'}
for card in accusation:     # set iterator 'card'
    print(card)             # iterator return only key 

# how to traversal value in dictionary
accusation = {'room': 'ballroom', 'weapon': 'leap pipe','person': 'Col. Mustard'}
for card in accusation.values():     # set iterator 'card' which traversal aaccusation dictionarty's value
    print(card)             # iterator return only key 

# method to traversal all dictionary value -> for card in accusaion.items():

# another cool iteration method is zip() 
# if you need to iteration multiple list at the same time 
# example

days = ['Monday','Tuesday','Wednesday']
fruites = ['banana', 'orange','peach']
drinks = ['coffee','tea','beer']








