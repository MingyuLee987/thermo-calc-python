# CALPHAD–Scheil–Active Learning 기반 Ce–Cr 합금 설계

## 1. Project Overview

본 프로젝트는 **Thermo-Calc CALPHAD 계산, Scheil–Gulliver 비평형 응고 계산, SEM/EDS 미세조직 분석, SP-240 전기화학 부식시험 및 Gaussian Process Regression(GPR) 기반 Active Learning을 결합한 순차적 합금 설계 프레임워크**이다.

연구의 핵심 목적은 Ce–Cr 조성 공간의 모든 조성을 직접 제조·평가하는 것이 아니라,

**계산적으로 가능한 조성 공간을 먼저 좁힌 뒤 → 실제 제조·부식시험을 수행하고 → 실험 결과를 이용하여 다음 조성을 선택하는 것**

이다.

CALPHAD 계산은 부식률을 직접 예측하는 모델로 사용하지 않는다. 대신 열역학적으로 실현 가능한 조성을 정의하는 **physics-based feasibility filter**로 사용한다.

최종 부식 성능의 Ground Truth는 실제 SP-240 전기화학 측정을 통해 획득한다.

---

# 2. Research Concept

전체 연구 전략은 다음과 같다.

```text
Initial Ce–Cr Composition Space
            │
            ▼
┌─────────────────────────────┐
│  CALPHAD Equilibrium        │
│  Screening @ 900 °C         │
│                             │
│  • Matrix phase stability   │
│  • Phase fraction           │
│  • Cr retention in matrix   │
│  • Harmful phase formation  │
└─────────────┬───────────────┘
              │
              ▼
     CALPHAD Feasible Domain
              │
              ▼
┌─────────────────────────────┐
│  Scheil–Gulliver            │
│  Solidification Simulation  │
│                             │
│  • Liquidus temperature     │
│  • Terminal solidification  │
│  • Freezing range           │
│  • T90–T99 interval         │
└─────────────┬───────────────┘
              │
              ▼
     Experimental Candidates
              │
              ▼
       Melting / Processing
              │
              ▼
           SEM / EDS
              │
              ▼
     SP-240 Corrosion Tests
              │
              ▼
     Experimental Ground Truth
              │
              ▼
┌─────────────────────────────┐
│ Gaussian Process Regression │
│       Active Learning       │
│                             │
│  Prediction μ(x)            │
│  Uncertainty σ(x)           │
└─────────────┬───────────────┘
              │
              ▼
       Next Experiments
              │
              └──────► Closed Loop
```

즉,

**Thermodynamic feasibility → Solidification behavior → Fabrication → Corrosion experiment → Active Learning**

순으로 조성 공간을 탐색한다.

---

# 3. Alloy Design Space

기본 합금계는 Fe 기반 합금이며 Ce와 Cr을 주요 설계변수로 사용한다.

Thermo-Calc 계산에는 다음 원소가 포함된다.

```text
Fe – C – Si – Mn – Cu – Ni – P – S – Cr – Ce
```

사용 Thermo-Calc database:

```text
TCFE13
```

기본 조성은 다음과 같다.

| Element | Composition (wt%) |
| ------- | ----------------: |
| C       |             0.050 |
| Si      |             0.750 |
| Mn      |             0.600 |
| Cu      |             0.350 |
| Ni      |             0.200 |
| P       |             0.015 |
| S       |             0.005 |
| Fe      |           Balance |

Ce와 Cr을 설계변수로 설정한다.

```text
Ce : 0.000 – 0.099 wt%
step = 0.001 wt%

Cr : 0.800 – 1.399 wt%
step = 0.001 wt%
```

따라서 초기 계산 조성 공간은 약

```text
100 × 600 = 60,000 compositions
```

으로 구성된다.

모든 조성을 직접 제조하는 대신 CALPHAD를 이용하여 열역학적으로 의미 있는 영역을 먼저 추출한다.

---

# 4. Step 1 — CALPHAD Equilibrium Screening

Main script:

```text
01_tc_calphad_screen1.py
```

CALPHAD screening의 목적은

> **“주어진 Ce–Cr 조성이 실제 열처리 조건에서 목표 matrix를 안정적으로 유지할 수 있는가?”**

를 판단하는 것이다.

평형 계산 조건은 실제 열처리 조건을 반영하여 설정한다.

```text
Solution treatment temperature : 900 °C
Pressure                       : 1 × 10^5 Pa
Target matrix                  : FCC_A1
```

900 °C 용체화 조건에서 주요 matrix를 Austenite(`FCC_A1`)로 설정한다.

---

## 4.1 Matrix Phase Stability

각 Ce–Cr 조성에 대해 평형 계산을 수행하고 FCC_A1 phase fraction을 계산한다.

목표 조건:

```text
FCC_A1 phase fraction ≥ 0.95
```

즉,

$$
f_{\mathrm{FCC\_A1}}\ge0.95
$$

인 조성을 우선적으로 선택한다.

이는 용체화 열처리 조건에서 최소 95% 이상의 Austenite matrix를 확보하기 위한 조건이다.

---

## 4.2 Cr Retention in Matrix

Nominal Cr 농도만으로 합금의 실제 matrix 조성을 판단하기 어렵다.

첨가된 Cr 일부는 carbide 또는 secondary phase 형성에 사용될 수 있기 때문이다.

따라서 CALPHAD 계산을 이용하여 FCC_A1 내부의 실제 Cr 농도

$$
C_{Cr}^{FCC}
$$

를 별도로 계산한다.

Screening criterion:

$$
C_{Cr}^{FCC}\ge0.80\;wt\%
$$

이를 통해 단순한 nominal composition이 아니라 **열처리 후 실제 matrix에 유지되는 effective Cr concentration**을 기준으로 후보를 평가한다.

---

## 4.3 Harmful Phase Screening

다음 상들을 harmful secondary phases로 정의한다.

```text
SIGMA
MU_PHASE
CHI
LAVES_C14
LAVES_C15
```

전체 harmful phase fraction에 대해

$$
f_{\mathrm{harmful}}\le0.01
$$

조건을 적용한다.

즉, 유해상 분율이 1%를 초과하는 조성은 CALPHAD feasible domain에서 제외한다.

---

## 4.4 Tracked Phases

CALPHAD 계산에서는 다음 상들의 안정성도 추적한다.

```text
FCC_A1
BCC_A2
M7C3_D101
MS_B1
CE2S3
CE3S4_D73
CE2C3_D5C
```

특히 Ce 첨가에 따라 Ce-containing sulfide/carbide의 열역학적 안정성이 어떻게 변화하는지를 확인할 수 있다.

향후 SEM/EDS에서 관찰되는 Ce-containing inclusion과 CALPHAD 결과를 비교하여

```text
Calculated Ce-containing phase
             ↓
      SEM/EDS inclusion
             ↓
    Pit initiation behavior
             ↓
      Corrosion response
```

의 상관관계를 분석하는 것을 목표로 한다.

---

# 5. CALPHAD Feasibility Criterion

최종 CALPHAD feasibility는 다음 세 조건을 동시에 만족하는 경우로 정의한다.

$$
\boxed{
f_{\mathrm{FCC}}\ge0.95
}
$$

$$
\boxed{
f_{\mathrm{harmful}}\le0.01
}
$$

$$
\boxed{
C_{Cr}^{FCC}\ge0.80\,wt\%
}
$$

따라서,

```text
CALPHAD feasible
       =
Matrix stability
       AND
Low harmful-phase fraction
       AND
Sufficient Cr retained in matrix
```

가 된다.

CALPHAD는 여기서 최적 부식 조성을 결정하는 것이 아니라,

**전체 Ce–Cr 공간에서 물리적으로 실현 가능한 후보 공간을 정의하는 constraint**

역할을 한다.

---

# 6. Step 2 — CALPHAD Result Visualization

Scripts:

```text
01b_plot_calphad_map.py
01b_plot_calphad_corrosion.py
```

CALPHAD 결과를 Ce–Cr 2차원 composition map으로 시각화한다.

주요 분석 대상은 다음과 같다.

```text
Ce concentration
        │
        ├── Matrix fraction
        ├── Cr concentration in matrix
        ├── Harmful phase fraction
        └── CALPHAD feasibility
```

이를 통해 전체 조성공간에서 feasible / infeasible 영역 및 thermodynamic boundary를 확인한다.

---

# 7. Step 3 — Scheil–Gulliver Solidification Simulation

Main script:

```text
02_tc_scheil_selected.py
```

평형 CALPHAD 계산은 특정 온도에서의 최종 안정상 정보를 제공하지만, 실제 합금 제조 과정에서는 비평형 응고가 발생한다.

따라서 CALPHAD screening을 통과한 후보에 대해 추가적으로 **Scheil–Gulliver solidification simulation**을 수행한다.

Scheil 계산의 핵심 질문은 다음과 같다.

> **“열역학적으로 가능한 이 조성이 실제 응고 과정에서도 제조하기 적합한가?”**

---

# 8. Scheil Solidification Descriptors

Scheil simulation으로부터 고상분율–온도 관계

$$
f_s(T)
$$

를 계산하고 주요 응고 descriptor를 추출한다.

---

## 8.1 Liquidus Temperature

응고가 시작되는 온도를

$$
T_L
$$

로 정의한다.

---

## 8.2 Terminal Solidification Temperature

Scheil 계산에서 최종 응고가 진행되는 온도를

$$
T_{end}
$$

로 정의한다.

---

## 8.3 Freezing Range

전체 응고 온도범위는

$$
\Delta T_{freeze}=T_L-T_{end}
$$

로 계산한다.

넓은 freezing range는 실제 제조 과정에서 장시간 solid–liquid coexistence가 발생할 가능성을 의미하므로 조성별 solidification behavior를 비교하는 descriptor로 활용한다.

---

## 8.4 Terminal Solidification Interval

Scheil curve에서

$$
f_s=0.90
$$

및

$$
f_s=0.99
$$

에 해당하는 온도를 각각 \(T_{90}\), \(T_{99}\)로 정의한다.

Terminal solidification descriptor는

$$
\Delta T_{90-99}=T_{90}-T_{99}
$$

로 계산한다.

이 값은 응고 후반부 90–99% solid fraction 영역의 온도구간을 나타내며, 조성에 따른 terminal solidification behavior와 제조 민감도를 비교하는 보조 지표로 사용한다.

> **Note**
>
> 현재 코드의 Scheil 단계는 주로 solidification-temperature 기반 descriptor를 계산한다. Ce/Cr의 최종 액상 농축량 또는 segregation coefficient를 직접적인 microsegregation descriptor로 계산하는 기능은 향후 확장 대상으로 남겨두고 있다.

---

# 9. Scheil Result Visualization

Script:

```text
02b_plot_scheil_descriptors.py
```

Scheil 결과를 이용하여 다음 descriptor들을 비교한다.

```text
1. Liquidus / terminal solidification temperature
2. Total freezing range
3. T90 – T99 solidification interval
4. Composition-dependent solidification behavior
```

CALPHAD와 Scheil 계산을 결합함으로써

```text
Equilibrium stability
        +
Non-equilibrium solidification
        ↓
More realistic candidate screening
```

이 가능하다.

---

# 10. Step 4 — Alloy Fabrication and SEM/EDS

CALPHAD 및 Scheil 계산 결과를 기반으로 실제 제조·평가 가치가 높은 조성을 선택한다.

실제 제조된 시편에 대해서는 SEM/EDS를 이용하여 미세조직을 분석한다.

주요 분석 항목:

```text
• Inclusion number density
• Inclusion size
• Inclusion composition
• Ce-containing inclusion
• Secondary phases
• Pit initiation site
```

특히 Ce 첨가에 의해 형성되는 inclusion과 부식 pit initiation의 상관관계를 분석한다.

---

# 11. Step 5 — SP-240 Electrochemical Corrosion Test

Experimental data template:

```text
03_sp240_template.csv
```

부식 성능의 Ground Truth는 SP-240 기반 전기화학 측정에서 얻는다.

주요 측정값은 다음과 같다.

### Open Circuit Potential

$$
E_{corr}
$$

### Potentiodynamic Polarization

Tafel extrapolation을 이용하여

$$
i_{corr}
$$

를 측정한다.

### Cyclic Polarization

Pitting corrosion resistance 평가를 위해

$$
E_{pit}
$$

및 repassivation 관련 전위를 측정한다.

### Electrochemical Impedance Spectroscopy

Equivalent circuit fitting으로

$$
R_{ct}
$$

를 계산한다.

시험 환경:

```text
Electrolyte          : 3.5 wt% NaCl
Temperature          : 25 °C
Reference electrode  : Ag/AgCl
```

---

# 12. Step 6 — Gaussian Process Regression

Main script:

```text
04_active_learning.py
```

실험에서 얻어진 데이터를 이용하여 Gaussian Process Regression 모델을 학습한다.

초기 모델의 입력은

$$
\mathbf{x}=
[X_{Ce},X_{Cr}]
$$

이며 주요 출력은

$$
y_1=\log_{10}(i_{corr})
$$

$$
y_2=E_{pit}
$$

등의 부식 성능 지표이다.

GPR은 각 미측정 조성에서

$$
\mu(\mathbf{x})
$$

와

$$
\sigma(\mathbf{x})
$$

를 동시에 제공한다.

여기서

* \(\mu\): 예상 부식 성능
* \(\sigma\): 모델의 예측 불확실도

를 의미한다.

따라서 단순히 가장 좋은 것으로 예상되는 조성만 선택하는 것이 아니라, 아직 충분한 정보가 없는 조성도 실험 대상으로 선택할 수 있다.

---

# 13. Physics-Constrained Active Learning

본 프로젝트의 중요한 특징은 ML이 전체 Ce–Cr 공간을 자유롭게 탐색하지 않는다는 것이다.

후보 공간을

$$
\mathcal{C}_{CALPHAD}
=
\{\mathbf{x}\mid CALPHAD\ feasible\}
$$

로 제한한다.

따라서 다음 실험 조성은 반드시

$$
\mathbf{x}_{next}\in\mathcal{C}_{CALPHAD}
$$

조건을 만족한다.

즉,

```text
Machine Learning
       │
       ▼
"어디를 실험할 것인가?"

CALPHAD
       │
       ▼
"어디까지 실험할 수 있는가?"
```

라는 역할 분담을 갖는다.

이를 통해 열역학적으로 의미가 낮은 조성에 실험 비용을 사용하는 것을 줄인다.

---

# 14. Active Learning Candidate Selection

각 Active Learning cycle에서는 크게 exploitation과 exploration을 고려한다.

### Candidate A — Corrosion-rate exploitation

예상 \(i_{corr}\)가 낮은 후보:

$$
\arg\min_{\mathbf{x}}\mu_{\log i_{corr}}(\mathbf{x})
$$

### Candidate B — Pitting-resistance exploitation

예상 \(E_{pit}\)가 높은 후보:

$$
\arg\max_{\mathbf{x}}\mu_{E_{pit}}(\mathbf{x})
$$

### Candidate C — Exploration

GPR uncertainty가 높거나 CALPHAD thermodynamic boundary 근처에 위치한 후보를 선택한다.

이를 통해

```text
Known good region
      +
Unknown region
      +
Thermodynamically interesting region
```

을 균형 있게 탐색한다.

---

# 15. Closed-Loop Sequential Alloy Design

전체 연구는 다음 cycle을 반복한다.

```text
Cycle n
   │
   ▼
CALPHAD feasible candidates
   │
   ▼
GPR prediction
   │
   ├── μ(x): expected performance
   │
   └── σ(x): uncertainty
   │
   ▼
Select next compositions
   │
   ▼
Fabrication
   │
   ▼
SEM / EDS
   │
   ▼
SP-240
   │
   ▼
New experimental data
   │
   ▼
Update GPR
   │
   ▼
Cycle n+1
```

따라서 본 연구는 단순한 Thermo-Calc 계산이나 ML 기반 부식 예측이 아니라,

$$
\boxed{
\text{Physics}
+
\text{Experiment}
+
\text{Machine Learning}
}
$$

을 결합한 **Closed-Loop Sequential Alloy Design Framework**를 목표로 한다.

---

# 16. Repository Structure

```text
thermo-calc-python/
│
├── config.py
│   └── Alloy composition, Thermo-Calc database,
│       temperature and feasibility criteria
│
├── 01_tc_calphad_screen1.py
│   └── Equilibrium CALPHAD screening
│
├── 01b_plot_calphad_map.py
│   └── CALPHAD composition-map visualization
│
├── 01b_plot_calphad_corrosion.py
│   └── CALPHAD/corrosion-related visualization
│
├── 02_tc_scheil_selected.py
│   └── Scheil–Gulliver solidification simulation
│
├── 02b_plot_scheil_descriptors.py
│   └── Scheil descriptor visualization
│
├── 03_sp240_template.csv
│   └── Experimental corrosion-data template
│
├── 04_active_learning.py
│   └── GPR + Active Learning candidate selection
│
├── debug_tcpython_api.py
│   └── TC-Python API debugging utilities
│
├── main.py
├── run.py
│
├── data/
│   └── Experimental / intermediate data
│
└── output/
    └── CALPHAD, Scheil and Active Learning results
```

---

# 17. Recommended Execution Order

기본적인 분석 순서는 다음과 같다.

```bash
python 01_tc_calphad_screen1.py
python 01b_plot_calphad_map.py
python 01b_plot_calphad_corrosion.py

python 02_tc_scheil_selected.py
python 02b_plot_scheil_descriptors.py

python 04_active_learning.py
```

각 단계는 독립적인 분석이 아니라 이전 단계의 결과를 다음 단계의 입력 또는 constraint로 사용하는 순차적 workflow로 구성된다.

---

# 18. Interpretation of Each Module

| Module          | Scientific Question         |
| --------------- | --------------------------- |
| CALPHAD         | 이 조성이 열역학적으로 가능한가?          |
| Matrix Cr       | 첨가한 Cr이 실제 matrix에 얼마나 남는가? |
| Phase analysis  | 원하지 않는 2차상이 형성되는가?          |
| Scheil          | 실제 응고 과정은 어떻게 진행되는가?        |
| SEM/EDS         | 계산에서 예상한 조직이 실제로 형성되는가?     |
| SP-240          | 실제 부식 성능은 어떠한가?             |
| GPR             | 미측정 조성의 성능과 불확실도는 어떠한가?     |
| Active Learning | 다음 실험은 어느 조성에서 수행해야 하는가?    |

---

# 19. Current Scope and Future Development

현재 프레임워크는 CALPHAD equilibrium calculation과 Scheil solidification descriptor를 이용하여 물리적 후보 공간을 정의하고, 실제 부식 데이터를 이용한 Active Learning으로 다음 실험 조성을 선정하는 것을 목표로 한다.

향후 다음 기능으로 확장할 수 있다.

**Scheil microsegregation descriptors**

```text
Ce enrichment in terminal liquid
Cr enrichment in terminal liquid
Maximum/minimum solid composition
Segregation index
Composition at fs = 0.90 / 0.95 / 0.99
```

예를 들어,

$$
S_i=
\frac{C_i^{terminal}}{C_i^{nominal}}
$$

과 같은 descriptor를 도입하면 Ce와 Cr의 비평형 미세편석을 보다 직접적으로 정량화할 수 있다.

또한 향후 GPR 입력변수를

$$
[Ce,Cr]
$$

에서

$$
[
Ce,\,
Cr,\,
f_{FCC},\,
Cr_{matrix},\,
f_{Ce-phase},\,
\Delta T_{freeze},\,
\Delta T_{90-99},\,
N_{incl},\,
d_{incl}
]
$$

등으로 확장하면 단순 조성 기반 모델에서 **physics-informed surrogate model**로 발전시킬 수 있다.

---

# 20. Final Objective

본 프로젝트의 최종 목표는 **가장 많은 조성을 계산하거나 실험하는 것**이 아니다.

목표는 제한된 실험 예산 내에서

> **“어떤 조성을 다음으로 제조하고 측정해야 가장 많은 정보를 얻으면서 동시에 우수한 부식 성능의 합금을 발견할 수 있는가?”**

를 계산적으로 결정하는 것이다.

최종적으로,

$$
\boxed{
\text{CALPHAD}
\rightarrow
\text{Scheil}
\rightarrow
\text{Fabrication}
\rightarrow
\text{SEM/EDS}
\rightarrow
\text{SP-240}
\rightarrow
\text{GPR}
\rightarrow
\text{Active Learning}
}
$$

으로 연결되는 **physics-constrained closed-loop alloy design framework** 구축을 목표로 한다.

---

## Important Note

CALPHAD 및 Scheil 결과는 실제 부식 성능의 Ground Truth가 아니다.

CALPHAD는 **thermodynamic feasibility**, Scheil은 **non-equilibrium solidification behavior**, SEM/EDS는 **real microstructure**, SP-240은 **experimental corrosion performance**를 담당한다.

따라서 계산 결과만으로 특정 조성이 최고의 내식성을 갖는다고 판단하지 않으며, 최종 성능 평가는 반드시 실제 전기화학 실험 결과를 기반으로 한다.
