# -*- coding: UTF-8 -*-
import cv2 as cv  # cv2 영상처리 cv라이브러리
import argparse  # 2.7버전 optparse 대체 하기위해 추가됨 api 수정이 필요한 기능 지원 인자 구분?
import numpy as np  # numpy 별명 np사용 다차원 배열 사용을 위해
import time  # time() 함수, strftime() 함수 , lovaltime() 함수 사용 시간관련 함수
import os
import socket

from imutils import paths

from utils import choose_run_mode, load_pretrain_model # utils.py파일에 3개 함수 가져오기
from Pose.pose_visualizer import TfPoseVisualizer  # pose파일 밑에 pose에있는 pose_visualizer 중에서 TfPoseVisualizer클래스 가져옴
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model

HOST = '36.38.61.187'
PORT = 8888

# model 4
smoke = load_model('Model/4.model')

# ArgumentParser에 원하는 description을 입력하여 parser객체 생성
parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')  # openpose에 의한 작업 인식?

parser.add_argument('--video', help='Path to video file.',
                    default=os.path.basename("C:/Users/haram/PycharmProjects/OpenBankProject/"
                                             "tabaco1.mp4"))

# parse_args() method로 명령창에서 주어진 인자를 파싱한다.
args = parser.parse_args()  # args 이름으로 파싱 성공시 args.parameter 형태로 주어진 인자 값을 받아서 사용가능

# 관련 모델 가져오기
# tensorflow 추상화 라이브러리
estimator = load_pretrain_model('VGG_origin')  # 훈련 모델 로드(VGG_origin) 분류??

# 인자 초기화
realtime_fps = '0.0000'
start_time = time.time()
fps_interval = 1
fps_count = 0
run_timer = 0
frame_count = 0

# 동영상 파일 읽고 쓰기(웹캠 입력만 테스트)
cap = choose_run_mode(args)  # cap 객체에 choose_run_mode 파싱

while cv.waitKey(1) < 0: #키가 입력될때까지 반복
    data = []
    has_frame, show = cap.read()  # has_frame 과 show에 비디오를 한프레임씩 읽음 성공시 True, 실패시 False
    if has_frame:
        fps_count += 1  # fps 카운트
        frame_count += 1  # frame 카운트

        # pose estimation 추정치 배열
        humans = estimator.inference(show)  # 비디오에서 사람 객체 추정
        # get pose info 사람의 동작을 추정 frame,joints, bboxes, xcenter 반환
        pose = TfPoseVisualizer.draw_pose_rgb(show, humans)  # return frame, joints, bboxes, xcenter

        image = img_to_array(show)  # image (450, 300, 3)
        image = cv.resize(image, dsize=(224, 224),
                          interpolation=cv.INTER_NEAREST)  # 이웃 보간법 사용

        image = preprocess_input(image)
        data.append(image)
        data = np.array(data, dtype="float32")

        predIdxs = smoke.predict(data)
        predIdxs = np.argmax(predIdxs, axis=1)

        for i in range(30):
            print()

        for i in range(len(humans)):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((HOST, PORT))

            cv.putText(show, str(i), (pose[2][i][0], pose[2][i][1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            if predIdxs == 1:
                msg = "1"
                s.send(msg.encode(encoding='utf_8', errors='strict'))
                data = s.recv(1024)
                print('인공지능이 ' + str(i) + '번째 객체가 흡연 중인 것을 탐지했습니다. \n관리자에게 \'' + data.decode() + '\'를 전송합니다.')
                s.close()
            # elif predIdxs == 0:
            #     msg = "0"  # exec(open("test4.py 0"))
            #     s.send(msg.encode(encoding='utf_8', errors='strict'))
            #     data = s.recv(1024)
            #     print('result: ' + data.decode())
            #     s.close()

        height, width = show.shape[:2] # 한 프레임씩 읽은 비디오의 높이, 넓이 크기

        # 실시간 fps값 보이기
        if (time.time() - start_time) > fps_interval:
            # 이과정의 프레임 수 계산 INTERVAL이 1초라면 FPS
            realtime_fps = fps_count / (time.time() - start_time)
            fps_count = 0  # 프레임수 0
            start_time = time.time()
        fps_label = 'FPS:{0:.2f}'.format(realtime_fps)  # 출력 할 fps
        cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)  # 화면에 fps 출력

        # 감지된 사람 수 보이기
        num_label = "Human: {0}".format(len(humans))  # 비디오에서 딴 사람객체 카운트해서 보여줌
        cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)  # 화면에 사람 수 출력

        # 현재 실행 시간 및 총 프레임 수 보이기
        if frame_count == 1:
            run_timer = time.time()
        run_time = time.time() - run_timer
        time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
        cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv.imshow('Action Recognition based on OpenPose', show)  # 창 이름

cap.release()  # cap 비디오 객체 해제