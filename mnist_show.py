# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# (훈련데이터) , (훈련데이터 라벨), (테스트 데이터), (테스트 데이터 라벨)
# flatten = True라는 것은 입력 이미지를 평탄하게 1차원 배열로 변환해라.
# normalize : 입력 이미지의 픽셀의 값을 0~1(True) 사이로 할지 아니면 원래값인 0 ~ 255(False)로 할지를 결정하는 함수
# 0 ~ 255 범위의 각 픽셀의 값을 0.0~1.0 사이의 범위로 변환을 하는데 이렇게 특정 범위로 변환처리를 하는것을 "정규화"라고 하고
# 신경망의 입력데이터에 특정 변환을 가하는것을 "전처리"라 한다.

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
print(img.shape)  # (28, 28)

img_show(img)
