from tkinter import *
import random
import time
# import arcade


#############################################################################################여기는 사람
# class Man_me:
#     def __init__(self, canvas):
#         self.canvas = canvas
#         self.man = canvas.create_rectangle(0, 0, 10, 20, fill='magenta')
#         self.canvas.move(self.man, 295, 480)
#         self.x = 0
#         self.y = 0
#         # self.canvas_width = self.canvas.winfo_width()      #canvas 가로
#         # self.canvas_height = self.canvas.winfo_height()    #canvas 세로
#         self.canvas.bind_all('<KeyPress-Left>', self.turn_left)
#         self.canvas.bind_all('<KeyPress-Right>', self.turn_right)   #canvas가 감지
#         self.canvas.bind_all('<KeyPress-Down>', self.stop)
#         # print(self.canvas_width)
#
#     def draw(self):
#         man_pos = self.canvas.coords(self.man)       #self.man의 좌상우하의 좌표, 위치
#
#         if man_pos[0] <= 0 and self.x < 0:                      #self.man이 오른쪽으로 나가지 않도록
#             self.x = 5
#         elif man_pos[2] >= 600 and self.x > 0:
#             self.x = -5
#
#         self.canvas.move(self.man, self.x, self.y)
#
#
#     def turn_left(self, evt):
#         self.x = -5
#
#
#     def turn_right(self, evt):
#         self.x = 5
#
#     def stop(self, evt):
#         self.x = 0
#############################################################################################여기는 사람



class Man_random:

    def __init__(self, canvas):
        self.canvas = canvas
        self.man = canvas.create_rectangle(0, 0, 10, 20, fill='magenta')
        self.canvas.move(self.man, 295, 480)         # self.man 그리기
        self.x = 0
        self.y = 0
        self.man_pos = self.canvas.coords(self.man)  # self.man의 좌상우하의 좌표, 위치 선언


    def draw(self):
        self.man_pos = self.canvas.coords(self.man)
        self.random()   # 랜덤함수는 여기서 돌려준다.

        # self.man이 화면 밖으로 나가지 않도록 하는 코드
        if self.man_pos[0] <= 0 and self.x < 0:
            self.x = 5
        elif self.man_pos[2] >= 600 and self.x > 0:
            self.x = -5

        # random()에서 받은 속력값을 대입해서 self.man 그리기
        self.canvas.move(self.man, self.x, self.y)


    def random(self):
        # 랜덤으로 self.man이 움직일 속력 반환
        random_move = (-5, 0, 5)
        self.x = random.choice(random_move)



class Poop:

    def __init__(self, canvas, man):   # append 되면서 실행
        self.get_point = 0   # 득점상황
        self.poop_x = 0      # self.poop의 x좌표
        self.man_x = 0       # self.man의 x좌표
        self.man_speed = 0   # self.man의 속력

        self.canvas = canvas
        self.man = man
        self.poop = canvas.create_oval(0, 0, 10, 10, fill='#E86A0C')
        self.x = 0
        self.y = 7   # poop이 떨어지는 속도 조절
        self.random_pos = None
        self.x_pos = 0
        self.random()
        self.canvas.move(self.poop, self.x_pos, 0)      # self.poop 그리기
        self.poop_pos = self.canvas.coords(self.poop)   # self.poop의 좌상우하의 좌표, 위치 선언


    def random(self):
        # 랜덤으로 self.poop가 움직일 좌표 반환
        self.random_pos = [i for i in range(590)]
        self.x_pos = random.choice(self.random_pos)


    def draw(self):
        self.canvas.move(self.poop, self.x, self.y)      #똥 그리기, self.poop_pos도 같이 update함


    def check_out(self):
        self.poop_pos = self.canvas.coords(self.poop)
        if self.poop_pos[3] >= 520:
            return True
        return False


    def hit_man_get_point(self):   # 여기서 점수들을 판정해주고 판정당시의 정보들을 모아준다.
        self.poop_pos = self.canvas.coords(self.poop)

        # 득점 상황 선언, self.poop이 63개 이상 있을 때 처음 3번째 self.poop이 판정받고 return한다.(함수에서 나온다.)
        if self.man.man_pos[1] <= self.poop_pos[3] <= self.man.man_pos[3]-11:

            if self.man.man_pos[0] <= self.poop_pos[0] <= self.man.man_pos[2] \
                    or self.man.man_pos[0] <= self.poop_pos[2] <= self.man.man_pos[0]:
                self.get_point = -1
                self.man_x = int(self.man.man_pos[0])
                self.man_speed = self.man.x
                self.poop_x = int(self.poop_pos[0])
                print(self.man_x, self.man_speed, self.poop_x, self.get_point)
                return (self.man_x, self.man_speed, self.poop_x, self.get_point)

            else:
                self.get_point = 1
                self.man_x = int(self.man.man_pos[0])
                self.man_speed = self.man.x
                self.poop_x = int(self.poop_pos[0])
                print(self.man_x, self.man_speed, self.poop_x, self.get_point)
                return (self.man_x, self.man_speed, self.poop_x, self.get_point)


    def __del__(self):
        return 'del'






tk = Tk()
tk.title("Dodge Your Poop Faster")   #게임 창의 제목 출력
tk.resizable(0, 0)                   #tk.resizable(가로크기조절, 세로크기조절)
tk.wm_attributes("-topmost", 1)      #생성된 게임창을 다른창의 제일 위에 오도록 정렬
tk.update()  # 여기서 한번 다시 적어준다.

canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
#bd=0, highlightthickness=0 은 베젤의 크기를 의미한다.
canvas.configure(background='#E8D487')
canvas.pack()  #앞의 코드에서 전달된 폭과 높이는 매개변수에 따라 크기를 맞추라고 캔버스에에 말해준다.



man = Man_random(canvas)
poop = []
cnt = 0


while 1:
    tk.update()
    tk.update_idletasks()
    poop.append(Poop(canvas, man))   # 객체 생성, Poop.POOP_X
    man.draw()                       # Man_random.MAN_X, Man_random.MAN_speed

    for i in range(len(poop)):  # poop에 들어가있는 객체의 마지막 순서에 해당하는 객체의 메소드가 실행된다!
        try:
            poop[i].draw()      # 객체 실행
            poop[i].hit_man_get_point()
        except IndexError:
            continue

        if poop[i].check_out():
            del poop[i]


    time.sleep(0.05)