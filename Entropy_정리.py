# -((4 / 5) * log2(4 / 5)) + -((1 / 5) * log2(1 / 5))

def entropy(a, b):
    import math
    answer = (-a) * math.log(a, 2) + (-b) * math.log(b, 2)
    return answer


########################
def entropy2(p_list):
    import math
    return round(sum([(-p * math.log(p, 2)) for p in p_list if p != 0]), -2)
    # 그룹함수는 무조건 값을 내보낸다. (여기선 0으로)

# print(entropy2([0]))


#########################################
def class_probilities(labels):
    import collections
    b = []
    a = list(collections.Counter(labels).values())
    for i in a:
        b.append(i/len(labels))
    # print(b)
    return b

# print(class_probilities(card_yn))


inputs = [('a', 'b'), ('c','d')]
# print(inputs[0][0])


input1 = [{'ename':'scott'}, True]
# print(input1[0]['ename'])


input2 = [({'ename':'scott'}, True), ({'ename':'smith'}, False)]
# for i in input2:
#     print(i[0]['ename'])


input3 = [({'ename':'scott', 'card_yn':'y'}, True), ({'ename':'smith', 'card_yn':'n'}, False)]
# for i in input3:
#     print(i[0]['card_yn'])


input4 = [
         ( {'cust_name':'SCOTT', 'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True),
         ( {'cust_name':'SMITH', 'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True),
         ( {'cust_name':'ALLEN', 'card_yn':'N', 'review_yn':'N', 'before_buy_yn':'Y'}, False),
         ( {'cust_name':'JONES', 'card_yn':'Y', 'review_yn':'N', 'before_buy_yn':'N'}, True),
         ( {'cust_name':'WARD',  'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True) ]


# group = []
# for i in input4:
#     group.append(i[0]['card_yn'])
# print(group)


########################
def ent(ent_list):
    import math
    keys = list(ent_list[0][0].keys())
    for i in range(1, len(keys)):
        group = []
        for j in ent_list:
            group.append(j[0][keys[i]])
        print(keys[i])
        yield group
        print(' ')


def class_probilities1(labels):
    import collections
    b = []
    a = list(collections.Counter(labels).values())
    for i in a:
        b.append(i/len(labels))
    # print(b)
    yield b


def entropy3(p_list):
    import math
    return sum([(-p * math.log(p, 2)) for p in p_list if p != 0])


# for i in ent(input4):
#     for j in class_probilities1(i):
#         print(entropy3(j))


#################################################
from collections import defaultdict
import math
import collections
# group1 = {}
# group1['one'] = 'a'
# print(group1)

def noname():
    return 'a'

group2 = defaultdict(noname)
group2['one']
# print(group2)
# print(group2['one'])

group3 = defaultdict(lambda:'a')
group3['one']
# print(group3['one'])

group4 = defaultdict(list)    # 비어있는 list를 default값으로 하겠다.
group4['Y']
# print(group4)


def column_data(inputs, column):
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][column]
        # print(key)
        # print(input[1])
        groups[key].append(input[1])
    return groups

for column in ['card_yn', 'review_yn', 'before_buy_yn']:
    print(column, ': ',  column_data(input4, column).items())



