from bs4 import BeautifulSoup

with open("C:\\Users\ljh080\Desktop\\html 연습.html") as chotaeheum:
    soup = BeautifulSoup(chotaeheum, "lxml")

# print (soup.title.text)
# print(soup.find('a'))     #a 태그의 첫번째 요소만 한 줄로 가져오기
# print(soup.find_all('a')) #a 태그의 모든 요소를 한 줄로 가져오기

# for link in soup.find_all('a'):
#     print(link)                 #link 주소가 있는 요소를 가져오기
#     print(link.get('href'))     #link 주소를 가져오기

# print(soup.get_text())            #텍스트만 가져오기
# print(soup.get_text(strip=True))  #텍스트를 한줄로 가져오기

# result = soup.find(class_="number")     #class_="클래스 이름"
# print(result)                           #100000이 들어있는 요소(태그까지 포함해서)
# '''<div class="number">100000</div>'''

# print(result.get_text())           #100000 가져오기
# '''100000'''


# result = soup.find_all(class_="number")     #class_="클래스 이름"
# for link in result:
#     print(link.get_text())
# '''
# 100000
# 100000
# 1000
# 2000
# 100
# 100
# 80
# 50
# '''

# result = soup.find_all(class_="number")[2]     #class_="클래스 이름"
# print(result.get_text())
# '''1000'''

# result = soup.find_all(class_="name")[4]       #class_="클래스 이름"
# print(result.get_text())
# '''fox'''

# result = soup.find_all("ul")[2]
# print(result.li.div.text)            #fox 찾기
# '''fox'''
#
# result = soup.find(text="fox")
# print(result)                        #fox 찾기
# '''fox'''

# result = soup.find(id="primaryconsumers")
# print(result.li.div.text)            #deer 찾기
# '''deer'''

# result = soup.find_all('div', class_="number")[2]
# print(result.string)                 #1000 찾기
# '''1000'''

### 참고예제
# div_li_tags = soup.find_all(["div", "li"])
# all_css_class = soup.find_all(class_=["producerlist", "primaryconsumerslist"])