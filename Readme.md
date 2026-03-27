# 💳 신용카드 이상거래 탐지 시스템

> XGBoost 기반 실시간 신용카드 사기 탐지 시스템 · Flask API + 실시간 대시보드

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-189AB4?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle_Fraud_Detection-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![Version](https://img.shields.io/badge/Version-v1.0-brightgreen?style=flat-square)

---

## 📋 프로젝트 개요

Kaggle **Credit Card Fraud Detection** 데이터셋을 활용하여 XGBoost 기반 머신러닝 모델을 학습하고, 실시간으로 신용카드 거래 데이터를 입력받아 이상거래 여부를 판별한 뒤 대시보드에 시각화하는 내부 분석 시스템입니다.

### 핵심 문제: 극단적인 클래스 불균형

전체 **284,807건**의 거래 중 사기 거래는 단 **492건(0.17%)** 에 불과합니다.  
단순 정확도(Accuracy) 기반 평가는 의미가 없으며, **Recall과 PR-AUC 중심의 평가 전략**이 필수적입니다.

### 기술 스택

| 구분 | 기술 |
|------|------|
| ML 모델 | XGBoost |
| 백엔드 | Flask, Flask-CORS |
| 프론트엔드 | HTML / JavaScript (SSE) |
| 데이터 처리 | pandas, numpy, scikit-learn |
| 모델 직렬화 | joblib |
| 데이터셋 | Kaggle — mlg-ulb/creditcardfraud |

---

## 🎯 성공 기준

| 지표 | 목표값 | 설명 |
|------|--------|------|
| ROC-AUC | ≥ 0.95 | 전체 분류 성능 |
| PR-AUC | ≥ 0.70 | 불균형 데이터 기준 핵심 지표 |
| Recall | 최대화 | FN(미탐) 최소화 — 사기를 놓치지 않음 |
| FN 비용 | FP 비용보다 높음 | 사기 미탐지의 금전 피해 > 정상 거래 차단 불편 |

---

## 🗂 데이터셋 명세

| 항목 | 내용 |
|------|------|
| 출처 | Kaggle — mlg-ulb/creditcardfraud |
| 총 거래 건수 | 284,807건 |
| 사기 거래 | 492건 (0.172%) |
| 정상 거래 | 284,315건 (99.828%) |
| 피처 수 | 30개 (V1~V28, Time, Amount) |
| 레이블 컬럼 | Class (0: 정상, 1: 사기) |

### 피처 구조

| 컬럼명 | 설명 |
|--------|------|
| V1 ~ V28 | PCA 변환 익명화 피처 — 원본 의미 비공개 |
| Time | 첫 번째 거래로부터 경과 시간(초) — StandardScaler 정규화 적용 |
| Amount | 거래 금액(달러) — StandardScaler 정규화 적용, 역변환으로 원본 금액 복원 |
| Class | 레이블: 0(정상) / 1(사기) |

### 전처리 전략

| 전처리 항목 | 방법 | 비고 |
|-------------|------|------|
| Amount 스케일링 | 별도 `amount_scaler`로 StandardScaler 적용 | 역변환으로 실제 금액 복원 가능 |
| Time 스케일링 | 별도 `time_scaler`로 StandardScaler 적용 | Amount와 scaler 혼용 오류 방지 |
| 데이터 분할 | `train_test_split(test_size=0.2, stratify=y)` | 사기 비율 0.17% 유지 |
| 테스트 데이터 저장 | `test_data.csv`에 Amount 원본 컬럼 포함 | 대시보드 실제 금액 표시용 |

---

## 🧠 모델 선정

### 4가지 모델 비교 검토

| 모델 | ROC-AUC | 사기 탐지 적합성 |
|------|---------|----------------|
| Logistic Regression | ~0.97 | 낮음 — 비선형 패턴 학습 불가 |
| Decision Tree | ~0.85 | 낮음 — 과적합 경향 강함 |
| Random Forest | ~0.95 | 높음 |
| **XGBoost** ✅ | **~0.98** | **매우 높음** |

### 최종 선택: XGBoost

**이유 1 — 불균형 데이터 내장 처리**  
`scale_pos_weight = 정상 수 / 사기 수 (≈578)` 파라미터 하나로 SMOTE 없이 클래스 불균형을 내장 처리합니다.

**이유 2 — 희귀 패턴 집중 학습**  
이전 트리가 틀린 샘플(놓친 사기 거래)에 다음 트리가 집중하는 Boosting 구조로, 0.17%의 극히 드문 사기 패턴을 반복적으로 보정합니다.

**이유 3 — PR-AUC 최적화**  
`eval_metric="aucpr"`으로 PR-AUC 기준 직접 최적화가 가능합니다. 불균형 데이터에서는 ROC-AUC보다 PR-AUC가 더 신뢰도 높은 지표입니다.

### 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `n_estimators` | 300 | early_stopping으로 실제 사용 트리 수 결정 |
| `max_depth` | 6 | 트리 최대 깊이 |
| `learning_rate` | 0.05 | 각 트리의 기여 가중치 |
| `subsample` | 0.8 | 각 트리 학습 시 사용할 샘플 비율 |
| `colsample_bytree` | 0.8 | 각 트리 학습 시 사용할 피처 비율 |
| `scale_pos_weight` | ≈ 578 | 정상 수 / 사기 수 — 불균형 보정 핵심 파라미터 |
| `eval_metric` | `aucpr` | 불균형 데이터 최적 평가 지표 |
| `early_stopping_rounds` | 20 | 과적합 방지 |

---

## 🏗 시스템 아키텍처

```
[학습] creditcard.csv → train_model.py → xgb_fraud_model.pkl + test_data.csv
[추론] test_data.csv → app.py(Flask) → SSE 스트리밍 → dashboard.html
[통신] 프론트엔드 ← API 폴링(0.5s) + SSE 실시간 이벤트 ← Flask 백엔드
```

### 파일 구성

| 파일 | 역할 |
|------|------|
| `train_model.py` | 모델 학습 스크립트 — 데이터 로드 → 전처리 → XGBoost 학습 → 파일 저장 |
| `app.py` | Flask 백엔드 API — 예측 엔드포인트, SSE 스트리밍, 상태 관리 |
| `dashboard.html` | 프론트엔드 대시보드 — 실시간 차트, 거래 로그 테이블, 지표 카드 |
| `xgb_fraud_model.pkl` | 학습된 XGBoost 모델 (joblib 직렬화) |
| `amount_scaler.pkl` | Amount 전용 StandardScaler |
| `time_scaler.pkl` | Time 전용 StandardScaler |
| `test_data.csv` | 테스트 데이터 (Amount 원본 컬럼 포함) |
| `model_metrics.json` | ROC-AUC, PR-AUC, 혼동행렬 등 성능 지표 |

### Flask API 엔드포인트

| 메서드 | 엔드포인트 | 기능 |
|--------|------------|------|
| GET | `/api/metrics` | 모델 성능 지표 반환 (ROC-AUC, PR-AUC, 혼동행렬) |
| GET | `/api/status` | 현재 스트리밍 상태 — 처리 건수, 시간대별/금액대별 집계 |
| POST | `/api/stream/start` | 스트리밍 시작 (threshold 파라미터 수신) |
| POST | `/api/stream/stop` | 스트리밍 중지 |
| POST | `/api/stream/reset` | 상태 초기화 |
| POST | `/api/predict` | 단일 거래 예측 — JSON 거래 데이터 → 사기 확률 반환 |
| POST | `/api/threshold` | 판정 임계값 실시간 변경 (0.01 ~ 0.99) |
| GET | `/api/events` | SSE 실시간 이벤트 스트림 |

---

## 📊 대시보드 기능

### 지표 카드

| 카드 | 표시 내용 |
|------|----------|
| 처리된 거래 | 현재까지 처리된 총 거래 건수 |
| 이상거래 탐지 | 탐지된 이상거래 건수 및 전체 대비 비율(%) |
| 정상 거래 | 정상으로 판정된 거래 건수 및 비율 |
| 이상거래 비율 | 실시간 사기율 (기준치 0.17%와 비교) |
| 모델 ROC-AUC | 학습된 모델의 ROC-AUC / PR-AUC |

### 차트

- **시간대별 이상거래 추이 (Line Chart)** — 24시간대별 이상거래 누적 건수
- **이상거래 금액대 분포 (Bar Chart)** — $0-50 / $50-100 / $100-200 / $200-500 / $500-1000 / $1000+ 구간별 건수

> 💡 도넛 차트를 금액대 Bar Chart로 교체한 이유: 정상/이상 비율은 지표 카드에서 이미 수치로 표시되어 중복이었습니다. 금액대 분포로 교체하여 "어느 금액대에서 사기가 주로 발생하는가"라는 더 실질적인 인사이트를 제공합니다.

### Threshold 슬라이더

기본값 `0.5`. 슬라이더로 `0.1 ~ 0.9` 범위에서 실시간 조정 가능합니다.

| 설정 | 효과 |
|------|------|
| 임계값 낮춤 (예: 0.3) | 더 많은 거래를 이상으로 분류 → Recall ↑, Precision ↓ |
| 임계값 높임 (예: 0.7) | 확실한 경우만 이상으로 분류 → Precision ↑, Recall ↓ |
| 운영 권장 | 사기 탐지 목적상 임계값을 낮게 유지하여 Recall 우선 |

---

## 🚀 실행 가이드

### 1. 패키지 설치

```bash
pip install xgboost scikit-learn pandas numpy joblib flask flask-cors
```

### 2. 데이터 준비

[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)에서 `creditcard.csv`를 다운로드하여 `backend/` 폴더에 위치시킵니다.

### 3. 모델 학습

```bash
python train_model.py
# 약 1~2분 소요
# → xgb_fraud_model.pkl, amount_scaler.pkl, time_scaler.pkl, test_data.csv 생성
```

### 4. Flask 서버 실행

```bash
python app.py
# → http://localhost:5000
```

### 5. 대시보드 열기

`dashboard.html`을 브라우저로 열고 **스트리밍 시작** 버튼을 클릭합니다.

### 생성 파일 목록

| 파일 | 설명 |
|------|------|
| `xgb_fraud_model.pkl` | 학습된 XGBoost 모델 |
| `amount_scaler.pkl` | Amount 전용 StandardScaler |
| `time_scaler.pkl` | Time 전용 StandardScaler |
| `feature_cols.json` | 모델 입력 피처 목록 |
| `model_metrics.json` | ROC-AUC, PR-AUC, 혼동행렬 등 성능 지표 |
| `test_data.csv` | 테스트 데이터 (Amount 원본 컬럼 포함) |

---

## 🐛 개발 중 주요 이슈 및 해결

<details>
<summary><strong>FileNotFoundError: xgb_fraud_model.pkl</strong></summary>

**원인**: `app.py` 실행 전 `train_model.py` 미실행, 또는 다른 폴더에서 실행하여 파일 경로 불일치

**해결**:
```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
```
절대경로 기반 처리 및 필수 파일 존재 여부 사전 점검 로직 추가

**실행 순서**: `train_model.py` → `app.py`
</details>

<details>
<summary><strong>Amount $undefined 표시</strong></summary>

**원인**: scaler 하나로 Amount와 Time을 순차적으로 fit하여 마지막 fit(Time)이 `scaler.pkl`에 저장됨. Amount 역변환 시 잘못된 scaler 참조

**해결**: `amount_scaler` / `time_scaler` 분리 저장 및 `test_data.csv`에 원본 Amount 컬럼 직접 포함하여 역변환 불필요하도록 구조 단순화
</details>

<details>
<summary><strong>도넛 차트 중복 정보</strong></summary>

**원인**: 정상/이상 비율이 지표 카드와 중복

**해결**: 금액대 분포 Bar Chart로 교체하여 더 유의미한 인사이트 제공
</details>

---

## 🔭 향후 고도화 방향

- [ ] SMOTE 또는 언더샘플링 추가 적용을 통한 성능 개선
- [ ] 실제 카드사 PCA 변환 파이프라인 연동 시 실거래 데이터 적용
- [ ] 모델 재학습 주기 설정 및 드리프트 감지 기능
- [ ] 관리자 페이지 — 임계값 이력 관리, 탐지 이력 DB 저장
- [ ] Docker 기반 배포 패키징 및 `.env` 환경설정 가이드 제공
- [ ] 사용자 인증 및 권한 관리 기능
- [ ] 데이터셋을 직접 입력하여 해당 데이터 판별

---

## 📄 라이선스

본 프로젝트는 학습 및 포트폴리오 목적으로 제작되었습니다.  
데이터셋: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Andrea Dal Pozzolo et al.)