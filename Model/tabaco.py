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


# 입력 데이터 파일은 읽기 전용 ".../input/" 디렉토리에서 사용할 수 있습니다.
# 예를 들어 실행 또는 Shift+Enter를 누르면 입력 디렉토리 아래에 모든 파일이 나열됩니다.

import tensorflow as tf

# hello = tf.constant('hello, Tensorflow!')
# sess = tf.Session()
# print(sess.run(hello))

from tensorflow.python.client import device_lib

# import tensorflow as tf
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#     raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))

# def get_available_devices():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos]
#
# print(get_available_devices())

# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

def run():
    dataset_path = './1024data/'
    model_store_dir = '2.model'

    INIT_LR = 1e-4
    EPOCHS = 100
    BS = 32
    imagePaths = list(paths.list_images(dataset_path))
    print(imagePaths)
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

    labels = np.array(labels)
    # labels = non_smoking or smoking

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    # labels2= [1. 0.] or [0. 1.]

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
    # print('trinaX', trainX.shape)     # (2603, 224, 224, 3)
    # print(len(trainX))                # 2603
    # print('textX', testX.shape)       # textX (651, 224, 224, 3)
    # print(len(testX))                 # 651

    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)

    for layer in baseModel.layers:
        layer.trainable = False
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    print('textX', testX.shape)       # textX (651, 224, 224, 3)
    print(len(testX))                 # 651

    H = model.fit(aug.flow(trainX, trainY, batch_size=BS),
                  steps_per_epoch=len(trainX) // BS,
                  validation_data=(testX, testY),
                  validation_steps=len(testX) // BS,
                  epochs=EPOCHS)

    predIdxs = model.predict(testX, batch_size=BS)
    predIdxs = np.argmax(predIdxs, axis=1)

    print(classification_report(testY.argmax(axis=1), predIdxs,target_names=lb.classes_))

    # 모델 저장
    model.save(model_store_dir, save_format="h5")
    N = EPOCHS

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('ploy.jpg')

if __name__ == '__main__':
    run()