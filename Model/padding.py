# 이미지 주변에 padding을 넣고 300*300으로 변환, 저장하기.
import cv2
import numpy as np
from imutils import paths

filepath = "C:\\Users\\haram\\PycharmProjects\\OpenBankProject\\1"
imagePaths = list(paths.list_images(filepath))

size = 500

for i, imagePath in enumerate(imagePaths):
    save_dicname = imagePath.split('\\')
    save_filename = "/".join(save_dicname[:-2]) + '/2/' + save_dicname[-1]
    print(save_filename)
    # print(path)
    img = cv2.imread(imagePath)
    # print(img.shape[1], img.shape[0])
    print(imagePath)

    # 가로와 세로 중 큰 값을 size로 맞추고, 빈 공간에 padding
    if(img.shape[1] > size or img.shape[0] > size):
        percent = 1
        if(img.shape[1] > img.shape[0]) :
            percent = size/img.shape[1]
        else:
            percent = size/img.shape[0]

        img = cv2.resize(img, dsize=(0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_LINEAR) # 양 선형 보간법

    # 이미지 범위 지정
    y, x, h, w = (0, 0, img.shape[0], img.shape[1])

    # 그림 주변에 검은색으로 칠하기
    w_x = (size-(w-x))/2  # w_x = (size - 그림)을 뺀 나머지 영역 크기 [ 그림나머지/2 [그림] 그림나머지/2 ]
    h_y = (size-(h-y))/2

    if(w_x < 0):         # 크기가 -면 0으로 지정.
        w_x = 0
    elif(h_y < 0):
        h_y = 0

    M = np.float32([[1, 0, w_x], [0, 1, h_y]])
    img_re = cv2.warpAffine(img, M, (size, size))
    # print(img_re)
    # print(img_re.shape)
    # cv2.imshow("img_re", img_re)

    # 이미지 저장하기
    cv2.imwrite(save_filename, img_re)