# Stock Prediction Model

## 프로젝트 개요

본 프로젝트는 **졸업작품**으로 개발된 주식 예측 모델입니다. TAFAS(Test-time Adaptation Framework for Time Series), Time-LLM, SAN(Slice-based Adaptive Normalization) 등의 최신 시계열 예측 기법을 활용하여 주식 가격을 예측하는 딥러닝 모델을 구현했습니다.

## 주요 특징

- **비정상성 대응**: 시계열 데이터의 비정상성(non-stationarity) 문제를 해결하기 위한 테스트타임 적응 프레임워크 적용
- **다중 모델 앙상블**: TAFAS, Time-LLM, SAN 모델을 결합한 앙상블 방식
- **장기 예측**: 장기간 주식 가격 변동 예측에 최적화
- **실시간 적응**: 예측 시점의 시장 요점을 제공하여 실시간 적응에 사용


## 사용법

### 모델 실행
```bash
bash scripts/{model}/{dataset}_{pred_len}/run.sh
```

예시:
```bash
bash scripts/FreTS/ETTh1_720/run.sh
```

### 모델 훈련
체크포인트가 없는 경우, `run.sh` 파일에서 `TRAIN.ENABLE`을 `True`로 변경하여 훈련 가능합니다.

## 라이센스 및 인용

## 라이센스
본 프로젝트는 다음 오픈소스 프로젝트들을 활용했습니다:

- **TAFAS**: MIT License  
  - 출처: [kimanki/TAFAS](https://github.com/kimanki/TAFAS)
  - 논문: Kim, HyunGi, et al. "Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation." AAAI 2025

- **SAN**: Apache License 2.0  
  - 출처: [icantnamemyself/SAN](https://github.com/icantnamemyself/SAN)
  - 논문: "Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective" (NeurIPS 2023)

- **Time-LLM**: Apache License 2.0  
  - 출처: [KimMeen/Time-LLM](https://github.com/KimMeen/Time-LLM)
  - 논문: "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models"