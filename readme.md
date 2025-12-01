# 내가 왕이될 상인가 (관상 프로그램)

> 얼굴 이미지 → 정량 지표 추출 → 선(Line) 시각화 → GPT/Gemini 관상 해석 
> 백엔드(Flask)와 프론트엔드(HTML/CSS/JS)로 구성된 프로젝트입니다.

<img width="1220" height="851" alt="image" src="https://github.com/user-attachments/assets/df2da8be-f1fb-448f-9bed-74d86cc666c3" />

## 소개
이 프로젝트는 얼굴 이미지 한 장으로 정량 지표(metrics) 를 추출하고,
이를 기반으로 정밀 선(Line) 시각화와 AI 관상 해석을 제공하는 프로그램입니다.

눈·코·입·턱·광대 등 주요 부위의 거리·각도·비율·대칭성을 실제로 측정해
LLM(GPT/Gemini)의 해석이 더 신뢰성 있게 이루어지도록 합니다.

특징

- 기존 관상 앱 대비 높은 정확도·일관성

- 선(Line) 기반 시각화로 측정 근거가 명확함

- 정량 지표를 활용해 LLM 호출 비용 절감

## 1. 사전 준비

* Python 3.7 이상

* 가상환경(venv) 권장

* (Windows) dlib 빌드를 위해 CMake + Visual C++ Build Tools 필수

## 2. 설치 및 환경 구성

[CMake 공식 사이트](https://cmake.org/download/)에서 설치

```bash
# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필수 라이브러리 설치
pip install --upgrade pip
pip install face_recognition pillow
```

> dlib 빌드가 필요하므로 CMake 및 Visual C++ Build Tools(Windows) 설치 필수

## 3. 지표

다음 지표를 계산합니다:

* **거리(Distances)**: 눈 사이 거리, 코 길이, 입 너비 등
* **비율(Ratios)**: 얼굴 폭/높이 대비 비율
* **각도(Angles)**: 눈 축, 턱 선 기울기 등
* **면적(Areas)**: 눈·입술 영역 면적
* **대칭성(Symmetry)**: 좌우 대칭 오차
* **고급 지표(Advanced)**: 골든비율 적합도, T존 지수, 얼굴 구역 비율

상세 코드는 `app.py`에 구현되어 있습니다.


## 4. 실행 예시

```bash
python app.py --image test.jpg --output result.json
```

* `app.py` 는 랜드마크 추출, 메트릭 계산, JSON 출력
* 이후 GPT API 호출 스크립트와 연동

## 5. 주의사항 및 팁

* **정면 사진 권장**: 비스듬한 각도는 왜곡 발생
* **프라이버시**: 민감 데이터이므로 HTTPS/암호화 적용
* **면책 조항**: 오락용 분석임을 명시
* **대체 솔루션**: 설치 불가 시 Mediapipe 또는 AWS Rekognition 사용

