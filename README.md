# CALPHAD–SP-240–SEM–Active Learning 기반 부식합금 설계

## 1. 핵심 개념 및 프레임워크

본 연구의 프레임워크는 부식실험을 전산으로 대체하는 것이 아닙니다. **SP-240 기반 전기화학 부식실험을 제한된 시편 예산 내에서 가장 효율적으로 수행하기 위한 의사결정 시스템(Closed-Loop Active Learning)**을 구축하는 데 목적이 있습니다.

- **Ground Truth**: 부식 성능의 정답값은 항상 SP-240 전기화학 측정에서 획득합니다.
  - $i_{\mathrm{corr}}$: 동전위 분극곡선(PDP) 및 Tafel 외삽법 기반 부식전류밀도
  - $E_{\mathrm{pit}}$: 순환 분극곡선(Cyclic polarization) 기반 피팅 개시전위
  - $R_{\mathrm{ct}}$: 전기화학 임피던스 분광법(EIS) 기반 전하전달저항
  - $E_{\mathrm{corr}}$: OCP 및 혼합전위 해석 보조지표
- **CALPHAD (Thermo-Calc)**: 부식률을 직접 계산하지 않으며, 상안정성·원소분배·유해상 석출·응고 편석 범위를 계산하여 **물리적으로 실현 가능한 조성 공간(Feasible domain)을 정의**하는 사전 필터 역할을 수행합니다.
- **GPR & Active Learning**: 누적된 SP-240 실측 데이터를 기반으로 성능 예측값($\mu$)과 불확실도($\sigma$)를 계산하여, 다음 사이클에서 제조 및 시험할 최적 후보를 제안합니다.

$$\boxed{ \text{Composition Space} \xrightarrow{\text{CALPHAD Filter}} \text{Feasible Candidates} \xrightarrow{\text{Fabrication \& SP-240}} \text{Corrosion Truth} \xrightarrow{\text{GPR \& Acquisition}} \text{Next Candidates} }$$

---

## 2. 전체 연구 워크플로우

```text
POSEIDON 500 기준 조성
        │
        ▼
Ce / RE 및 Cr 첨가 범위 설정
        │
        ▼
Thermo-Calc CALPHAD 계산
- 평형 상분율 (Equilibrium phase fraction)
- Matrix 내 Cr 유효농도
- 유해상 형성 가능성
- Scheil 응고경로 및 Freezing range
        │
        ▼
CALPHAD Feasible 조성 풀 확정 (C_CALPHAD = 1)
        │
        ▼
초기 후보군 (8–12개) 용해 및 시편 제조
        │
        ▼
SEM / EDS 조직 정량
- Inclusion 개수밀도 (N_incl), 크기 (d_incl), 조성
- 부식 피트 개시 사이트 (Pit initiation site)
        │
        ▼
SP-240 전기화학 부식 측정
- OCP, EIS (R_ct)
- PDP (i_corr), Cyclic Polarization (E_pit, E_rep)
        │
        ▼
Gaussian Process Regression (GPR) 학습
- 입력: 조성 x = [X_Ce, X_Cr] (초기 모델)
- 출력: log10(i_corr), E_pit, log10(R_ct)
        │
        ▼
Active Learning 후보 선별 (Cycle별 3개)
├── 후보 A (Exploitation 1): 예상 i_corr 최소화
├── 후보 B (Exploitation 2): 예상 E_pit 최대화
└── 후보 C (Exploration): 불확실도 최대 + CALPHAD 상경계
        │
        ▼
차기 시편 제작, 재측정 및 폐루프 반복