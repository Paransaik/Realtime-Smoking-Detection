## Object pose detection

A skeleton-based real-time online action recognition project, classifying and recognizing base on framewise joints, which can be used for safety monitoring..  
(The code comments are partly descibed in chinese)

---

## Introduction

_The **pipline** of this work is:_

- Realtime pose estimation by [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose);
- Online human tracking for multi-people scenario by [DeepSort algorithm](https://github.com/nwojke/deep_sortv);
- Action recognition with DNN for each person based on single framewise joints detected from Openpose.

---

## Dependencies

- python >= 3.5
- Opencv >= 3.4.1
- sklearn
- tensorflow & keras
- numpy & scipy
- pathlib

---

## Add

- python=3.8
- tensorflow == 2.5.0
- keras==2.4.3
- py-opencv==4.0.1
- scikit-learn==0.24.2

---

## Usage

- Download the openpose VGG tf-model with command line `./download.sh`(/Pose/graph_models/VGG_origin) or fork [here](https://pan.baidu.com/s/1XT8pHtNP1FQs3BPHgD5f-A#list/path=%2Fsharelink1864347102-902260820936546%2Fopenpose%2Fopenpose%20graph%20model%20coco&parentPath=%2Fsharelink1864347102-902260820936546), and place it under the corresponding folder;
- `python main.py`, it will **start the webcam**.
  (you can choose to test video with command `python main.py --video=test.mp4`, however I just tested the webcam mode)
- By the way, you can choose different openpose pretrained model in script.  
  **VGG_origin**: training with the VGG net, as same as the CMU providing caffemodel, more accurate but slower, **mobilenet_thin**: training with the Mobilenet, much smaller than the origin VGG, faster but less accurate.  
  **However, Please attention that the Action Dataset in this repo is collected along with the** **_VGG model_** **running**.

---

## Training with own dataset

- prepare data(actions) by running `main.py`, remember to **_uncomment the code of data collecting_**, the origin data will be saved as a `.txt`.
- transforming the `.txt` to `.csv`, you can use EXCEL to do this.
- do the training with the `traing.py` in `Action/training/`, remember to **_change the action_enum and output-layer of model_**.

---

![4  결과보고서003](https://user-images.githubusercontent.com/30463982/176209421-30a3d1ab-7e73-4411-9c27-6fbd06740e59.png)
![4  결과보고서004](https://user-images.githubusercontent.com/30463982/176209428-942c99ac-ad72-4183-8293-afaeef81dddc.png)
![4  결과보고서005](https://user-images.githubusercontent.com/30463982/176209430-27c3facb-ab9b-4307-b3f6-44efdad0e7bb.png)
![4  결과보고서006](https://user-images.githubusercontent.com/30463982/176209433-7afa4922-92c4-48ef-a86e-d873ac074114.png)
![4  결과보고서007](https://user-images.githubusercontent.com/30463982/176209436-fa47581d-6e47-482b-b0ff-088c7fc116ce.png)
![4  결과보고서008](https://user-images.githubusercontent.com/30463982/176209439-a0a1fdb9-64a4-46f0-a704-68a05f7b8f0a.png)
![4  결과보고서009](https://user-images.githubusercontent.com/30463982/176209443-4e7867d5-f6d7-44d6-9955-22c4f4c66e34.png)
![4  결과보고서010](https://user-images.githubusercontent.com/30463982/176209446-d8846538-e302-4b8a-a4c9-3a431383fecb.png)
![4  결과보고서011](https://user-images.githubusercontent.com/30463982/176209447-6d19e4ac-b2aa-42f4-969b-56091bb18b35.png)
![4  결과보고서012](https://user-images.githubusercontent.com/30463982/176209452-f5d62f66-680a-4590-ae55-4de300826fd8.png)
![4  결과보고서013](https://user-images.githubusercontent.com/30463982/176209332-cabf2d69-1087-45ae-a575-f205afbb83c2.png)
![4  결과보고서014](https://user-images.githubusercontent.com/30463982/176209346-83370ba6-d4c3-4ab2-a07a-356bbdec79d7.png)
![4  결과보고서015](https://user-images.githubusercontent.com/30463982/176209349-da11d9da-4938-490e-baca-c94e691794bd.png)
![4  결과보고서016](https://user-images.githubusercontent.com/30463982/176209356-0ece58d8-856c-408b-926c-44b565714c3d.png)
![4  결과보고서017](https://user-images.githubusercontent.com/30463982/176209358-6bd12f1c-0fa6-4b41-b11b-f62823bcd8a0.png)
![4  결과보고서018](https://user-images.githubusercontent.com/30463982/176209364-a4d6a5a8-b2c7-403f-9b09-84544a2e64aa.png)
![4  결과보고서019](https://user-images.githubusercontent.com/30463982/176209366-1ce5281c-d444-4ce5-a925-1b41cdb3cbbf.png)
![4  결과보고서020](https://user-images.githubusercontent.com/30463982/176209368-35c7fc9a-8d2d-413a-80c1-f49b1eecae18.png)
![4  결과보고서021](https://user-images.githubusercontent.com/30463982/176209373-d9ffa16f-a9fd-4db4-9dfa-327dac6ab62c.png)
![4  결과보고서022](https://user-images.githubusercontent.com/30463982/176209375-51c6c66a-d11c-41e2-ae8e-91bb221baf82.png)
![4  결과보고서023](https://user-images.githubusercontent.com/30463982/176209380-b90c7056-833a-41a1-9d63-ee4f6f0377ef.png)
![4  결과보고서024](https://user-images.githubusercontent.com/30463982/176209383-bd3dcb24-7cce-4f62-8855-0788a4047307.png)
![4  결과보고서025](https://user-images.githubusercontent.com/30463982/176209387-1daac121-552f-468f-babd-07bc767f5178.png)
![4  결과보고서026](https://user-images.githubusercontent.com/30463982/176209390-30703d70-0887-45cf-904c-76b0b031c4d7.png)
![4  결과보고서027](https://user-images.githubusercontent.com/30463982/176209391-ba1a548e-4b44-4435-87d2-33674ce453c0.png)
![4  결과보고서028](https://user-images.githubusercontent.com/30463982/176209395-2fa5c9c0-164e-4bc0-ad7b-e978acfe8e5c.png)
![4  결과보고서029](https://user-images.githubusercontent.com/30463982/176209399-123205c0-8002-41a4-8591-8e6b44c726fd.png)
![4  결과보고서030](https://user-images.githubusercontent.com/30463982/176209403-dcdb07f1-4ac8-4b4c-99f3-2f8c9ba180f6.png)
![4  결과보고서031](https://user-images.githubusercontent.com/30463982/176209409-742888a5-ea96-4e8f-b8f4-5836c039ba95.png)
