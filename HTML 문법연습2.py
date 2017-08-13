# import urllib.request           # 웹브라우저에서 html문서를 얻어오기 위해 통신하기 위한 모듈
# from bs4 import BeautifulSoup   # html문서 검색 모듈
# import re
# import os

#
# def get_save_path():
#     save_path = input("Enter the file name and file location :" )
#     save_path = save_path.replace("\\", "/")
#
#     if not os.path.isdir(os.path.split(save_path)[0]):
#         os.mkdir(os.path.split(save_path)[0])    #'폴더'가 없으면 만드는 작업  c:\\data1\\11.txt하면 data1이 생성된다.
#     return save_path



# def fetch_list_url(a):   # 레이디버그
#     list_url = "http://home.ebs.co.kr/ladybug/board/6/10059819/oneBoardList?c.page={}&hmpMnuId=106&searchKeywordValue=0&bbsId=10059819&searchKeyword=&searchCondition=&searchConditionValue=0&".format(a)
#     url = urllib.request.Request(list_url)
#     print(url)
#     res = urllib.request.urlopen(url).read().decode("utf-8")
#     print(res)
#     #위의 두가지 작업을 거치면 위의 url의 html문서를 res변수에 담을 수 있게 된다.
#
#     soup = BeautifulSoup(res, "html.parser")
#     # 위의 ebs 게시판 url 로 접속했을때 담긴 html 코드를
#     # soup 에 담겠다
#     # e_reg = re.compile("(완젼)")    #완젼이라는 텍스트를 검색하기 위해서 컴파일
#     # a = soup.find(text=e_reg)
#     # b = soup.find('p')
#     # print
#     # for link in soup.find_all('a'):
#     #     print(link.get('href'))
#     # for link in soup.find_all('p', class_="con"):   #p태그의 class "con"
#     #     print(link.get_text(strip=True))
#
#
#     a = soup.find_all('p', class_="con")
#     b = soup.find_all('span', class_="date")
#     # cnt = 0
#     # for i in a:
#     #     print(b[cnt].text, i.get_text(strip=True))
#     #     cnt += 1
#
#     for idx,link in enumerate(a,0):
#         print(b[idx].text, link.get_text(strip=True))
#
# for a in range(1, 16):
#     fetch_list_url(a)
#
#
# fetch_list_url()




#http://home.ebs.co.kr/ladybug/board/6/10059819/oneBoardList?hmpMnuId=106
#http://search.hani.co.kr/Search?command=query&keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&media=news&sort=d&period=all&datefrom=2000.01.01&dateto=2017.05.17&pageseq=0

#######################################################################
#######################################################################
#######################################################################

#
# import urllib.request
# from  bs4 import BeautifulSoup
#


# def fetch_list_url():
#     params = []
#
#     for cnt in range(1, 15):
#         list_url = "http://search.joins.com/JoongangNews?page={}&Keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&SortType=New&SearchCategoryType=JoongangNews&MatchKeyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5".format(cnt)
#                     ###여기에 스크롤링할 웹페이지의 url 을 붙여넣습니다.
#
#         url = urllib.request.Request(list_url)                          # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         res = urllib.request.urlopen(url).read().decode("utf-8")        # 영어가 아닌 한글을 담아내기 위한 문자셋인 유니코드 문자셋을
#                                                                         # 사용해서 html 문서와 html  문서내의 한글을 res 변수에 담는다. (유니코드 안쓰면 글씨 다 깨짐)
#         '''
#         참고>>  문자를 담는 set:
#                 1. 아스키코드 : 영문
#                 2. 유니코드 : 한글, 중국어
#         '''
#         soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
        # print(soup)
        # return soup

        # soup2 = soup.find_all('p')[0]
        # print(soup2)
        # return soup2

        # soup3 = soup2.find("a")     #print(soup2.a)와 같음
        # print(soup3)
        # return soup3


        # soup3 = soup2.find("a")["href"]     #이렇게 하면 href의 링크만 나옴
        # print(soup3)
        # return soup3

        # for i in range(8):
        #     soup2 = soup.find_all('p')[i]
        #     print(soup2.a["href"])
        # soup3 = soup2.find_all('a')     #이렇게 하면 href의 링크만 나옴
        # print(soup3)
        # return soup3

        # for link in soup.find_all('p'):
        #     print(link.a["href"])   #print(link.find('a')[href])도 같음

    #     for link in soup.find_all('span', class_='thumb'):   #p에는 이미지가 있는 뉴스만 있다. 이미지 없는 뉴스도 가져오기 위해서 dt사용
    #         try:
    #             # print(link.a["href"])   #print(link.find('a')[href])도 같음
    #             params.append(link.a["href"])
    #         except:                        #<dt></dt>처럼 비어있는 곳이 있기 때문에 try except를 사용해야 한다.
    #             continue
    # # print(params)
    # return params

# fetch_list_url()


        # return link.a["href"]
#

# for i in range(115):
#     fetch_list_url(i)
#     params.append(fetch_list_url(i))
#
# print(params)





################################################################
################################################################
#

# def fetch_list_url2():
#     list_url = "http://www.hani.co.kr/arti/economy/consumer/794974.html"
#                 ###여기에 스크롤링할 웹페이지의 url 을 붙여넣습니다.
#
#     url = urllib.request.Request(list_url)                          # url 요청에 따른 http 통신 헤더값을 얻어낸다
#     res = urllib.request.urlopen(url).read().decode("utf-8")        # 영어가 아닌 한글을 담아내기 위한 문자셋인 유니코드 문자셋을
#                                                                     # 사용해서 html 문서와 html  문서내의 한글을 res 변수에 담는다. (유니코드 안쓰면 글씨 다 깨짐)
#     '''
#     참고>>  문자를 담는 set:
#             1. 아스키코드 : 영문
#             2. 유니코드 : 한글, 중국어
#     '''
#     soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#
#     soup2 = soup.find_all('div', class_='text')[0].get_text(strip=True)
#
#     print(soup2)




# def fetch_list_url2():
#     params2 = fetch_list_url()
#     f = open(get_save_path(), 'w', encoding="utf-8")    #get_save_path()를 인스턴스화!!!
#
#     for i in params2:
#         list_url = "{}".format(i)
#         ###여기에 스크롤링할 웹페이지의 url 을 붙여넣습니다.
#
#         url = urllib.request.Request(list_url)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
#         res = urllib.request.urlopen(url).read().decode("utf-8")  # 영어가 아닌 한글을 담아내기 위한 문자셋인 유니코드 문자셋을
#         # 사용해서 html 문서와 html  문서내의 한글을 res 변수에 담는다. (유니코드 안쓰면 글씨 다 깨짐)
#         '''
#         참고>>  문자를 담는 set:
#                 1. 아스키코드 : 영문
#                 2. 유니코드 : 한글, 중국어
#         '''
#         soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
#
#         soup2 = soup.find_all('div', id="article_body")[0].get_text(strip=True, separator='\n')
#         print(soup2)
#         f.write(soup2 + "\n")         #soup2를 f를 사용해서 써라!!(write)
#
#     f.close()
#
# fetch_list_url2()


# c:\\data\\han'인공지능'.txt


#######################################################################################
#######################################################################################
#######################################################################################
# 네이버 이미지에 접속해서 아이언맨 이미지를 다운로드 받는 파이썬코드 작성 #################### 크롬 드라이버 필요!!


import urllib.request
from  bs4 import BeautifulSoup
from selenium import webdriver   # 웹 애플리케이션의 테스트를 자동화하기 위한 프레임 워크, 아주 좋음

from selenium.webdriver.common.keys import Keys
import time   #sleep을 위한 time 모듈

# binary = 'c:\data/chromedriver.exe'
# # 크롬 드라이버의 위치(크롬드라이버 설치 필요)
# # 팬텀js를 사용하면 백그라운드로 실행할 수 있다.
#
# browser = webdriver.Chrome(binary) # 브라우저 인스턴스화
# # browser.get("https://search.naver.com/search.naver?where=image&sm=tab_jum&ie=utf8&query=")
# browser.get("https://www.bing.com/?scope=images&FORM=Z9LH1")
# # browser.get("https://search.naver.com/search.naver?where=image&amp;sm=stb_nmr&amp;")
# # 네이버의 이미지 검색 url(그냥 naver도 가능하지만 아래 코드가 지저문해져서..
#
# # elem = browser.find_element_by_id("nx_query")  # 네이버에서 이미지 검색에 해당하는 input창의 id가 nx_query이다. 찾아서 elem으로 사용하게끔 설정
# # find_elements_by_class_name("")
# elem = browser.find_element_by_id("sb_form_q")
#
# # 검색어 입력
# elem.send_keys("인공지능")
# elem.submit()   #엔터키 역할
#
# # 반복할 횟수
# for i in range(1, 3):
#     browser.find_element_by_xpath("//body").send_keys(Keys.END)
#     # end키를 누르기 전에 바디를 활성화 해놔야한다. 마우스로 클릭하는 개념
#     time.sleep(10)
#
# time.sleep(10)
# # 타임슬립 5초(네크워크 문제로 인해서)
# html = browser.page_source             # 크롬 브라우저에서 현재 불러온 소스를 가져온다.
# soup = BeautifulSoup(html, "lxml")     # 뷰티풀 수프를 사용해서 html코드를 검색할 수 있다.
#
#
# # print(soup)
# # print(len(soup))
# ######################그림파일 저장#########################
# def fetch_list_url():
#     params = []
#     imgList = soup.find_all("img", class_="mimg")
#     for im in imgList:
#         params.append(im["src"])  #params리스트 변수에 image url을 담는다.
#     return params
#
#
# def fetch_detail_url():
#     params = fetch_list_url()
#     # print(params)
#     a = 1
#     for p in params:
#         # 다운받을 폴더경로 입력
#         urllib.request.urlretrieve(p, "c://data//bingImages/" + str(a) + ".jpg")  #다운받을 폴더 경로 입력
#
#         a = a + 1
#
#
# fetch_detail_url()
#
# browser.quit()




#-*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from datetime import datetime
import operator
class FacebookCrawler:
    # FILE_PATH = 'D:\\02.Python\\facebook_data\\'
    # CHROME_DRIVER_PATH = 'D:\\02.Python\\'
    FILE_PATH = 'c:\\data\\facebook_data\\'
    CHROME_DRIVER_PATH = 'c:\\data\\'
    def __init__(self, searchKeyword, startMonth, endMonth, scroll_down_cnt):  #시작날짜, 종료날짜
        self.searchKeyword = searchKeyword
        self.startMonth = startMonth
        self.endMonth = endMonth
        self.scroll_down_cnt = scroll_down_cnt
        self.data = {}   #게시날짜, 게시글을 수집할 딕셔너리 변수
        self.url = 'https://www.facebook.com/search/str/' + searchKeyword + '/keywords_top?filters_rp_creation_time=%7B"start_month%22%3A"' + startMonth + '"%2C"end_month"%3A"' + endMonth + '"%7D'
        self.set_chrome_driver()   #크롬드라이버 위치를 지정하는 함수
        self.play_crawling()       #크롤링을 하는 함수 실행
    # chrome driver 생성 후 chrome 창 크기 설정하는 함수.
    def set_chrome_driver(self):
        self.driver = webdriver.Chrome(FacebookCrawler.CHROME_DRIVER_PATH + 'chromedriver.exe')
        # self.driver.set_window_size(1024, 768)   #크롬창 크기 설정
    # facebook 홈페이지로 이동 후 email, password 를 입력하고 submit 보내는 함수. (로그인)
    def facebook_login(self):
        self.driver.get("https://www.facebook.com/")
        self.driver.find_element_by_id("email").clear()   # 이메일 입력창
        self.driver.find_element_by_id("email").send_keys("gh506015@gmail.com")
        self.driver.find_element_by_id("pass").clear()    # 패스워드 입력창
        self.driver.find_element_by_id("pass").send_keys("22815305ab#")
        self.driver.find_element_by_id("pass").submit()
        time.sleep(5)  #로그인 시 시간이 걸리기 때문에 5초 sleep
        self.driver.get(self.url)
    # facebook page scroll down 하는 함수
    def page_scroll_down(self):
        for i in range(1, self.scroll_down_cnt):
            self.driver.find_element_by_xpath("//body").send_keys(Keys.END)
            time.sleep(3)
    # 크롤링 된 데이터를 파일로 저장하는 함수
    def data_to_file(self):
        with open(FacebookCrawler.FILE_PATH + self.searchKeyword + ".txt", "w", encoding="utf-8") as file:
            print('데이터를 저장하는 중입니다.')
            for key, value in sorted(self.data.items(), key=operator.itemgetter(0)):   # key로 정렬하겠다.(날짜순)
                # data.items() 에 key 와 value 가 들어있고 그리고 0 번째 요소로 정령하겠다.
                file.write(str(datetime.fromtimestamp(key)) + ' : ' + value + '\n')    # 문자형으로 변환
            file.close()
            print('데이터 저장이 완료되었습니다.')
    # 크롤링 수행하는 메인 함수
    def play_crawling(self):
        try:
            self.facebook_login()
            time.sleep(5)
            self.page_scroll_down()
            html = self.driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            #  . 이 클래스 # 이 id  붙이면 and 떨어뜨리면 or 조건
            for tag in soup.select('.fbUserContent._5pcr'):    # 클래스명 페이스북은 클래스명이 '.'으로 시작한다.
                                                               # '.fbUserContent._5pcr' 붙어있는 것은 class & class라는 뜻
                                                               # '.fbUserContent  ._5pcr' 떨어져있는 것은 class or class
                usertime = tag.find('abbr', class_='_5ptz')
                content = tag.find('div', class_='_5pbx userContent').find('p')   # 게시글
                if usertime is not None and content is not None:
                    self.data[int(usertime['data-utime'])] = content.get_text(strip=True)
                    # data 딕셔너리         key                          value
            self.data_to_file()   # data_to_file() 함수를 실행해서 data딕셔너리의 내용을 os의 파일로 생성한다.
            self.driver.quit()
        except:
            print('정상 종료 되었습니다.')
crawler = FacebookCrawler('멍들', '2012-12', '2017-05', 7)
crawler.play_crawling()

