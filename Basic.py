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
days    = ['Monday','Tuesday','Wednesday']
fruites = ['banana', 'orange','peach']
drinks  = ['coffee','tea','beer']
desserts =['tramisu', 'ice cream', 'pie', 'pudding','nugat']
for day, fruit, drink, dessert in zip(days,fruites,drinks,desserts): # we can each list value to using iteraters at the same time
    print(day, ": drink", drink, ' - eat', fruit, "- enjoy", dessert)
#########################################################################33
# comprehension
# 컴프리헨션 (함축) 은 하나 이상의 이터레이터로 부터 파이썬의 자료구조를 만드는 방밥
# numlist.append 리스트 수가 많아야 할때 리스트에 요소 추가를 이터레이터로 자동화 한다. 
# number_list = [number for number in range(1,10000)] 리스트에 요소를 다 넣지 않아도 가능 1~10000까지 수가 다 들어간다. 
# 복잡한 수열도 조건만 맞추면 컴프리헨션으로 리스트에 저장이 가능하다. 
# multiple_list = [number for in range(1,10000)]
#   if number % 3 == 0 :
#        number_list.append(number)
# rows = range(1,4)
# cols = range(1,3)
# cells = [(row, col) for row in rows for col in cols]
# for cell in cells:
#   print(cell)
# 리스트 이외에도 딕셔너리, 셋 같은 자료구조도 사용할 수 있다.
# 딕셔너리 컴프리헨션 
# word = 'letters'
# letter_counts = {letter: word.count for letter in set(word)} # set을 사용하여 중복된 문자를 세지 않는다. 
# print(letter_counts)
# 셋 컴프리헨션 
# a_set = {number for number in range(1,6) if number %3 == 1 }
# print(a_set)

# 튜플은 컴프리헨션이 없다 
# 제네레이터 
# 기존의 컴프리헨션에 소괄호로 열고 닫으면 된다.

# 함수 
thing = None
def is_none(thing): # def is define fuction name and parameter
    if thing is None:
        print('It\'s none')
    elif thing:
        print('It\'s true')
    else:
        print("It\'s False")
# 함수의 리턴값을 명시하지 않으면 함수는 None을 리턴한다 None 파이썬에만 있는 특별한 값으로 True 도 False 도 아닌 Null과 비슷한(?) 개념
# 파이썬은 메소드 오버로딩 없이 같은 메소드의 파라미터 구성요소를 변경 할 수 있다.
# 파이썬에서 매개변수 지정은 매우 유연하게 할 수 있다. 매개변수로 여러 자료구조를 할당할 수 있으며 매개변수의 개수를 몰라도 유연하게 대응 할 수 있는 기능을 지원한다. 
# 
def buggy(arg, result=[]):
    result.append(arg)
    print(result)
# 위치인자 일반적인 프로그래밍 언어에서 메소드의 파라미터가 여러개일 때는 그 순서에 맞게 파라미터를 입력받아야만 한다.
# 파이썬 에서는 복수의 파라미터를 요구하는 메소드가 있을 때 그 메소드의 
def menu(wine, entree, dessert): # 파라미터를 순서대로 딕셔너리를 만들어 리턴한다. 
    return{'wine': wine, 'entree': entree, 'dessert':dessert}
print(menu('sunny-side-up', 'jack Daniel', 'don-ko-tsu ramen'))
# 위와같이 메소드 파라미터를 순서와 상관없이 입력하면 딕셔너리에 키값에 맞지않는 이상한 메뉴가 리터된다. 
# 파이썬에서는 메소드의 파라미터 위치를 사용자가 까먹는 경우와 같은 위치 혼동을 피하기 위해 파라미터에 상응하는 이름을 인자에 지정할 수 있다. 이렇게
print(menu(entree ='beef', dessert = 'bagel', wine = 'bordeaux'))
# 순서도 상관없다. 파라미터에 상응하는 값을 넣어주면 된다. 그런데 이렇게 쓰려면 파라미터를 알아야 하기 때문에 메소드 사용법인 docstring을 등록하는게 좋다. 
def print_args(*args):
    print('Positional argument tuple:', args)
# 파라미터인 *(애스터리스크)args 는 파라미터의 개수를 상관없이 입력받을 수 있다. 
print_args('안녕하세요','감사해요','잘있어요', '다시만나요',1,2,3,4,5,6)
# 자료형과 파라미터의 개수와 관계없이 원하는 만큼 출력시킬 수 있다. 다만 튜플형태로 반환하기 때문에 수정, 변경이 불가능하다.
# 애스터리스크 두개**를  쓰면 딕셔너리로 사용할 수도 있다. 

# 제너레이터 
# 전체 시퀀스를 한번에 메모리에 생성하고 정렬할 필요 없이 잠재적으로 아주 큰 시퀀스를 순회할 수 있다.
# 제너레이터느 이터레이터에 대한 데이터의 소스로 자주 사용됨 
# range()와 동일한 기능을 하는 제너레이터 생성
#
def my_range(first,last,step=1):
    number = first
    while number < last:
        yield number  # 일드는 현재 숫자를 반환해주지만 return과 다르게 함수를 종료시키지 않고 계속 진행한다. 
        number += step
print(my_range(1,10)) 
ranger = my_range(1,10)
for x in ranger:
    print(x)

# 컴프리헨션을 이용해서 제너레이터를 만들거나 
# 컴프리헨션이 길어진다면 제너레이터 함수를 만들어서 사용한다.
# 데커레이터 decorator 는 하나의 함수를 취해서 또 다른 함수를 반환하는 함수다
# 기존의 함수의 기능을 추가하고 싶을대 함수를 바깥으로 하나 싸서 기존 함수를 수정하지 않고 사용할 수 있다.
# 변수의 이름레 언더스코더 가 앞뒤로 두개씩 들어가면 파이썬 내부의 시스템 사용을 위해 예약되어있다. 
def document_it(func):
    def new_function(*args, **kwargs):
        print('Running:', func.__name__)# __name__ 는 파이썬 시스템 예약어 이다. 파라미터로 들어온 함수의 이름을 출력한다.
        print('Positional aguments:', args)
        print('Keyword arguments:', kwargs)
        result = func(*args, **kwargs) # add_ints 함수 리턴값 저장 
        print('Result:', result)
        return result
    return new_function #리턴값을 리턴한다. 

def square_it(func):
    def new_function(*args, **kwargs):
        result = func(*args, **kwargs)
        return result*result
    return new_function

@square_it
@document_it # 이렇게 자동으로 데커레이터를 할당 할 수도 있는데 이렇게하면 따로 수동으로 할당할 필요가 없다 
def add_ints(a,b):
    return a+b 
cooler_add_ints = document_it(add_ints) # 데커레이터를 수동으로 할당한다. 상황에 따라 적절하게 사용한다. 
cooler_add_ints(3,5)  
# 여러개의 데커레이터를 가질 수도 있다. 
# 데커레이터가 하나 이상이면 자동을 할당할 때는 함수선언부에서 가까운 순 document_it() 먼저 실행 된다. 

word = 'bye'
def hello_printer():
    print(word)
    word = 'hello'
    print(word)
hello_printer()

# 파이썬에서의 예외처리 
short_list = [1,2,3]
position = 5
try:
    short_list[position]
except:
    print('Need a position between 0 and', len(short_list)-1,'but got position')

# 임의의 예외 발생시키기
class NotThreeMultipleError(Exception):    # Exception을 상속받아서 새로운 예외를 만듦
    def __init__(self):                    # 숫자 입력
        super().__init__('3의 배수가 아닙니다.')
 
def three_multiple():
    try:
        x = int(input('3의 배수를 입력하세요: '))
        if x % 3 != 0:                     # x가 3의 배수가 아니면
            raise NotThreeMultipleError    # NotThreeMultipleError 예외를 발생시킴
        print(x)
    except Exception as e:
        print('예외가 발생했습니다.', e)
 
three_multiple()
# 모듈과 패키징
# 모듈이란 파이썬의 정의와 문장들을 담고있는 파일이다. 
# 
# 파이썬 표준 라이브러