# 이 Python 3 환경에는 많은 유용한 분석 라이브러리가 설치되어 있습니다.
# 이것은 카글/도커 이미지로 정의된다: https://github.com/kaggle/docker-python
# 예를 들어, 여기에 로드해야 할 몇 가지 유용한 패키지가 있습니다.

import os
# for dirname, _, filenames in os.walk('./1024data'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# "Save & Run All"을 사용하여 버전을 만들 때 출력으로 보존되는 현재 디렉터리(/kaggle/working/)에 최대 5GB까지 쓸 수 있습니다.
# /kaggle/temp/에 임시 파일을 쓸 수도 있지만 현재 세션 외부에 저장되지는 않습니다.

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.models import load_model

# 입력 데이터 파일은 읽기 전용 ".../input/" 디렉토리에서 사용할 수 있습니다.
# 예를 들어 실행 또는 Shift+Enter를 누르면 입력 디렉토리 아래에 모든 파일이 나열됩니다.
'''
smoking	1996
nonsmok	1279
len(trainX) 2603
len(testX)	651
===========
3254
len(imagePaths) 3254
'''
'''


# model 1                                               # model 2
              precision    recall  f1-score   support                 precision    recall  f1-score   support
 not_smoking       0.79      0.82      0.81       255    not_smoking       0.82      0.83      0.83       255
     smoking       0.88      0.86      0.87       396        smoking       0.89      0.89      0.89       396

    accuracy                           0.85       651       accuracy                           0.86       651
   macro avg       0.84      0.84      0.84       651      macro avg       0.86      0.86      0.86       651
weighted avg       0.85      0.85      0.85       651   weighted avg       0.87      0.86      0.86       651

# model 3                                               # model 4
              precision    recall  f1-score   support                 precision    recall  f1-score   support
 not_smoking       0.83      0.84      0.84       439    not_smoking       0.82      0.85      0.83       227
     smoking       0.88      0.87      0.87       571        smoking       0.87      0.85      0.86       281

    accuracy                           0.86      1010       accuracy                           0.85       508
   macro avg       0.85      0.85      0.85      1010      macro avg       0.85      0.85      0.85       508
weighted avg       0.86      0.86      0.86      1010   weighted avg       0.85      0.85      0.85       508


'''

def run():
    dataset_path = './1024data/after'
    model_store_dir = '2.model'

    BS = 32

    imagePaths = list(paths.list_images(dataset_path))

    data = []
    labels = []

    for imagePath in imagePaths:
        # label = non_smoking or smoking
        label = imagePath.split(os.path.sep)[-2]
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(label)

    data = np.array(data, dtype="float32")
    print(data.shape)  # (3254, 224, 224, 3)
    print(type(data))

    model = load_model('./case3/1.model')  #, custom_objects={"InstanceNormalization": InstanceNormalization}

    predIdxs = model.predict(data, batch_size=BS)
    predIdxs = np.argmax(predIdxs, axis=1)  # 행으로 > 3254, 224, 3

    # model.summary()

    for i in range(len(data)):
        if i % 100 == 0:
            print("labels: " + str(labels[i]) + " predict: " + str(predIdxs[i]))


    # print(classification_report(testY.argmax(axis=1), predIdxs,target_names=lb.classes_))
    #
    # N = EPOCHS
    #
    # plt.style.use("ggplot")
    # plt.figure()
    # plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    # plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    # plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    # plt.title("Training Loss and Accuracy")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss/Accuracy")
    # plt.legend(loc="lower left")
    # plt.savefig('ploy.jpg')

if __name__ == '__main__':
    run()