# -*- coding: UTF-8 -*-
import cv2 as cv  # cv2 영상처리 cv라이브러리
from imutils import paths

from utils import choose_run_mode, load_pretrain_model # utils.py파일에 3개 함수 가져오기
from Pose.pose_visualizer import TfPoseVisualizer  # pose파일 밑에 pose에있는 pose_visualizer 중에서 TfPoseVisualizer클래스 가져옴

# padding 된 폴더 경로
filepath = "C:\\Users\\haram\\PycharmProjects\\OpenBankProject\\1024data\\2"
# filepath = "C:\\Users\\haram\\PycharmProjects\\OpenBankProject\\1024data\\test"
imagePaths = list(paths.list_images(filepath))
# "\\" --> "/"
for i in range(len(imagePaths)):
    imagePaths[i] = imagePaths[i].replace("\\", "/")

for i, imagePath in enumerate(imagePaths):
    # print('imagePath', imagePath)  # OpenBankProject/1024data/2/bb0695.jpg
    save_dicname = imagePath.split("/")
    # print('save_dicname', save_dicname)  # 'OpenBankProject', '1024data', '2', 'bb0695.jpg']
    save_filename = "/".join(save_dicname[:-2]) + '/3/' + save_dicname[-1]
    # print('save_filename', save_filename)  # OpenBankProject/1024data/3/bb0695.jpg
    estimator = load_pretrain_model('VGG_origin')  # 훈련 모델 로드(VGG_origin) 분류??
    # print('estimator', estimator.graph_path)
    show = cv.imread(imagePath)
    humans = estimator.inference(show)
    pose = TfPoseVisualizer.draw_pose_rgb(show, humans)
    cv.imwrite(save_filename, show)