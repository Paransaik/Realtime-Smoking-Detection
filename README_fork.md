# Fork from nvnnghia/Online-Realtime-Action-Recognition-based-on-OpenPose

## 환경

- Windows 10 Pro v2004
- Python 3.7
- Pipenv >= 2018.11.26
- tensorflow == 1.15
  - 사용하다. pipenv install package\tensorflow-1.15.3-cp37-cp37m-win_amd64.whl
  -타임아웃 문제 때문에
- scikit-learn == 0.20
  - sklearn.utils.linear_assignment_ .linear_assignment is removed from scikit-learn 0.23
- fix nametuple syntax error

# # # 테스트

- 將 test.mp4 放到與 main.py 同一層的目錄
- pipenv shell
  - pipenv shell
  - python main.py --video=test.mp4
- pipenv run
  - pipenv run python main.py --video=test.mp4
