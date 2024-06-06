# ASMR Test Server
Code for testing ASMR service by using Rest API

### 로컬 환경에서 실행 시 
- python3 설치 
- flask/app.py의 필요한 패키지 설치

### 가상환경에서 실행 시
/flask로 cd후 
```bash
$ pip install pipenv
```
flask 폴더에서 실행
```bash
$ pipenv shell
```
flask 폴더 내에 Pipfile 보고 필요한 패키지 자동 설치
```bash
$ pipenv install
```

### 실행 환경
- vscode의 live server extension 

universal_classfier 모델 다운로드 필요( 용량 너무 커서 깃헙에서 다운 불가)
아래 링크에서 다운로드 후 flask 폴더 내에 위치시키기
https://drive.google.com/file/d/1BCWJ0TPpQFnI2q2u0xSn2-tg_x_K2dRi/view?usp=drive_link
