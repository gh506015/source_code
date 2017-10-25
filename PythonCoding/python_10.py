# CHAPTER 10 시스템
# 10.1 파일
# 파이썬은 많은 다른 언어처럼 유닉스의 파일 연산 패턴을 지니고 있다. chown(), chmod() 함수 등은 똑같은 이름을 사용한다. 그리고 몇가지 새로운 함수가 존재한다.


# 10.1.1 생성하기: open()
# oops.txt파일을 생성해보자
fout = open('oops.txt', 'wt')   # 지정경로에 파일 생성
print('Oops, I created a file.', file=fout)
fout.close()

# 이제 몇가지 테스트를 수행해보자
# 10.1.2 존재여부 확인하기: exists()
# 파일 혹은 디렉토리가 실제로 존재하는지 확인하기 위해 exists()함수를 사용한다. 상대경로와 절대경로를 지정할 수 있다.
import os
os.path.exists('oops.txt')     # True
os.path.exists('./oops.txt')   # True
os.path.exists('waffle')       # False
os.path.exists('.')            # True
os.path.exists('..')           # True


# 10.1.3 타입확인하기: isfile()
# 이 절에 등장하는 세 함수(isfile, isdir, isabs)는 이름이 파일인지, 디렉토라인지, 또는 절대경로인지 확인한다.
# 먼저 isfile()함수를 사용하여 평범한 파일인지 간단한 질문을 던져본다.

name = 'oops.txt'
os.path.isfile(name)  # 이게 파일이냐?
os.path.isdir(name)  # 이게 디렉토리냐?
os.path.isdir('.')  # 하나의 점(.)은 현재 디렉토리를 나타내고, 두개의 점(..)은 부모(상위) 디렉토리를 나타낸다. 이들은 항상존재하기 때문에 True를 반환한다.

# os 모듈은 절대경로와 상대경로를 처리하는 많은 함수를 제공한다. isabs()함수는 인자가 절대경로인지 확인한다.
# 실제로 존재하는 파일이름을 인자에 넣지 않아도 된다.
os.path.isabs('name')   # False
os.path.isabs(name)     # False
os.path.isabs('/big/fake/name')   # True
os.path.isabs('big/fake/name')   # False


# 10.1.4 복사하기: copy()
# copy()함수는 shutil이라는 다른 모듈에 들어있다. 다음 예제는 oops.txt를 ohno.txt로 복사한다.
import shutil
shutil.copy('oops.txt', 'ohno.txt')
# shutil.move()함수는 파일을 복사한 후 원본파일을 삭제한다.(이동)


# 10.1.5 이름 바꾸기: rename()
# rename()은 말 그대로 파일 이름을 변경한다. 다음예제는 ohno.txt를 ohwell.txt로 이름을 바꾼다.
import os
os.rename('ohno.txt', 'ohwell.txt')


# 10.1.6 연결하기: link(), symlink()
# 유닉스에서 파일은 한 곳에 있지만, 링크(link)라 불리는 여러 이름을 가질 수 있다. 저수준의 하드링크에서 주어진 파일을 모두 찾는것은 쉬운 일이 아니다
# 심벌릭링크(symbolic link)는 원본파일을 새 이름으로 연결하여 원본파일과 새 이름의 파일을 한 번에 찾을 수 있도록 해준다.
# link()함수는 하드링크를 생성하고, symlink()함수는 심벌릭 링크를 생성한다. islink()함수는 파일이 심벌릭 링크인지 확인한다.
# oops.txt파일의 하드링크인 새 yikes.txt파일을 만들어보자
os.link('oops.txt', 'yikes.txt')
os.path.isfile('yikes.txt')   # True

# oops.txt파일의 심벌릭 링크인 새 jeepers.txt파일을 만들어보자
os.path.islink('yikes.txt')   # False, 심벌릭링크가 아니다.(그냥 링크)
os.symlink('oops.txt', 'jeepers.txt')
os.path.islink('jeepers.txt')


# 10.1.7 퍼미션 바꾸기: chmod()
# 유닉스 시스템에서 chmod()는 파일의 퍼미션(permission, 권한)을 변경한다. 사용자에 대한 읽기, 쓰기, 실행 퍼미션이 있다.
# 그리고, 사용자가 속한 그룹과 나머지에 대한 퍼미션이 각각 존재한다.
# 이 명령은 사용자, 그룹, 나머지 퍼미션을 묶어서 압축된 8진수의 값을 취한다. oops.txt를 이 파일의 소유자(파일을 생성한 사용자)만 읽을 수 있도록 만들어보자.
os.chmod('oops.txt', 0o400)

# 이러한 수수께끼같은 8진수 값을 사용하기보다는 (약간) 잘 알려지지 않은 아리송한 심벌을 사용하고 싶다면 stat모듈을 임포트하여 다음과 같이 쓸 수 있다.
import stat
os.chmod('oops.txt', stat.S_IRUSR)
### 설명이 약간 부족하니 더 찾아서 공부하시오!


# 10.1.8 오너십 바꾸기: chown()
# 이 함수는 유닉스/리눅스/맥에서 사용된다. 숫자로 된 사용자 아이디(uid)와 그룹 아이디(gid)를 지정하여 파일의 소유자와 그룹에 대한 오너십을 바꿀 수 있다.
uid = 5
gid = 22
os.chown('oops', uid, gid)   # AttributeError 뜨는데 파이썬 버전때문인듯함


# 10.1.9 절대 경로 얻기: abspath()
# 이 함수는 상대 경로를 절대 경로로 만들어준다. 만약 현재 디렉토리가 /usr/gaberlunzie고, oops.txt파일이 거기에 있다면, 다음과 같이 입력할 수 있다.
# 그러니까 쉽게 말하면 상대경로를 입력하면 절대경로를 출력해준다~ 이말임
os.path.abspath('oops.txt')


# 10.1.10 심벌릭링크 경로 얻기: realpath()
# 이전 절에서 oops.txt 파일의 심벌릭 링크인 jeepers.txt 파일을 만들었다. 다음과 같이 realpath()함수를 사용하여 jeepers.txt파일로부터
# 원본파일인 oops.txt 파일의 이름을 얻을 수 있다.
os.path.realpath('jeepers.txt')   # 'C:\\python\\source_code\\oops.txt'


# 삭제하기: remove()
# remove()함수를 사용하여 oops.txt파일과 작별인사를 나누자
os.remove('oops.txt')
os.path.exists('oops.txt')


# 10.2 디렉토리
# 대부분의 운영체제에서 파일은 디렉토리의 계층구조 안에 존재한다.(최근에는 폴더라고 부른다.)
# 이러한 모든 파일과 디렉터리의 컨테이너는 파일 시스템이다.(volume이라고도 한다.)
# 표준 os 모듈은 이러한 운영체제의 특성을 처리하고, 조작할 수 있는 함수를 제공한다.

# 10.2.1 생성하기: mkdir()
# 시를 저장할 poems 디렉토리를 생성한다.
os.mkdir('poems')
os.path.exists('poems')


# 10.2.2 삭제하기: rmdir()
os.rmdir('poems')
os.path.exists('poems')


# 10.2.3 콘텐츠 나열하기
# 다시 poems 디렉토리를 생성한다.
os.mkdir('poems')

# 그리고 이 디렉토리의 콘텐츠를 나열한다.(아직 아무것도 없음)
os.listdir('poems')

# 이제 하위 디렉토리를 생성한다.
os.mkdir('poems/mcintype')
os.listdir('poems')

# 하위 디렉토리에 파일을 생성한다.
fout = open('poems/mcintype/the_good_man', 'wt')
fout.write('''Cheerful and happy was his mood, 
He to the poor was kind and good,
And he oft' times did find them food.''')
fout.close()

# 드디어 파일이 생겼다. 디렉토리의 콘텐츠를 나열해보자
os.listdir('poems/mcintype')


# 10.2.4 현재 디렉토리 바꾸기: chdir()
# 이 함수를 이용하면 현재 디렉토리에서 다른 디렉토리로 이동할 수 있다. 즉, 현재 디렉토리를 바꿀 수 있다. 현재 디렉토리를 떠나서 poems디렉토리로 이동해보자
import os
os.chdir('poems')
os.listdir('.')


# 10.2.5 일치하는 파일 나열하기: glob()
# glob()함수는 복잡한 정규표현식이 아닌, 유닉스 쉘 규칙을 사용하여 일치하는 파일이나 디렉토리의 이름을 검색한다. 규칙은 다음과 같다.
# 1. 모든 것에 일치 : *(re모듈에서의 .*와 같다.)
# 2. 한 문자에 일치 : ?
# a,b, 혹은 c 문자에 일치 : [abc]
# a,b, 혹은 c를 제외한 문자에 일치 : [!abc]
# m으로 시작하는 모든 파일이나 디렉토리를 찾는다.
import glob
glob.glob('m*')

# 두 글자로 된 파일이나 디렉토리를 찾는다.
glob.glob('??')

# m으로 시작하고 e로 끝나는 여덟글자의 단어를 찾는다.
glob.glob('m??????e')

# k,l, 혹은 m으로 시작하고 e로 끝나는 단어를 찾는다.
glob.glob('[klm]*e')


# 10.4 달력과 시간
# 날짜는 다양한 형식으로 표현할 수 있다.
# 1. July 29 1984
# 2. 29 Jul 1984
# 3. 29/7/1984
# 4. 7/29/1984
# 윤년은 부딪히는 또 다른 문제이다. 매 100년마다 오는 해는 윤년이 아니고 매 400년마다 오는 해는 윤년이다.
# 윤년은 4년마다 오는데 100년째(25번째)는 윤년이 아니다가 400년째 되는 해는 윤년이 된다.
import calendar
calendar.isleap(1900)
calendar.isleap(1996)
calendar.isleap(1999)   # 윤년검사기

# 파이썬 표준 라이브러리는 datetime, time, calendar, dateutil 등 시간과 날짜에 관한 여러가지 모듈이 있다. 일부 중복되는 기능이 있어서 혼란스럽다.


# 10.4.1 datetime 모듈
# 이는 여러 메서드를 가진 4개의 주요 객체르르 정의한다.
# date: 년, 월, 일
# time: 시, 분, 초, 마이크로초
# datetime: 날짜와 시간
# timedelta: 날짜 와/또는 시간 간격
# 년, 월, 일을 지정하여 date 객체를 만들 수 있다. 이 값은 속성으로 접근할 수 있다.
from datetime import date
halloween = date(2015, 10, 31)
halloween      # datetime.date(2015, 10, 31)
halloween.day  # 31
halloween.month  # 10
halloween.year  # 2015
halloween.isoformat()   # 날짜 출력하는 메서드



































































































