# MedCalc-Bench Verified test: Qwen3-0.6B-Base vs Qwen3-4B

## 1. The 2 × 2 grid (greedy accuracy %, pass@1 / pass@8 % on 275 problems, learnable band /275)

| policy | C: per-calculator, values | B: per-calculator, note | D: generic, values | A: generic, note |
|---|---|---|---|---|
| Qwen3-0.6B-Base | **26.2** · 15.5 / 52.7 · band 145 · loops 4.2 | **20.6** · 10.2 / 38.9 · band 107 · loops 5.5 | **4.9** · 3.1 / 17.8 · band 49 · loops 9.8 | **4.9** · 2.8 / 17.8 · band 49 · loops 15.7 |
| Qwen3-4B | — | **55.5** · 51.4 / 85.1 · band 212 · loops 6.0 | — | — |

## 2. Gaps (accuracy points)

| policy | knowledge B−A | knowledge C−D | extraction D−A | extraction C−B |
|---|---|---|---|---|
| Qwen3-0.6B-Base | +15.7 | +21.3 | +0.0 | +5.5 |
| Qwen3-4B | — | — | — | — |

## 3–6. Condition C (per-calculator, values)

**Outcome taxonomy (counts of 1,100)**

| outcome | Qwen3-0.6B-Base | Qwen3-4B |
|---|---|---|
| correct | 288 | — |
| wrong value (>50% off) | 286 | — |
| wrong value (<=50% off) | 281 | — |
| wrong value (<=10% off) | 76 | — |
| no answer: truncated (no loop) | 64 | — |
| wrong value (date / weeks) | 55 | — |
| no answer: degenerate loop | 46 | — |
| bad format in box | 4 | — |

**Accuracy by category (%)**

| category | n | Qwen3-0.6B-Base | Qwen3-4B |
|---|---|---|---|
| lab test | 380 | 27.4 | — |
| risk | 240 | 13.8 | — |
| physical | 240 | 34.2 | — |
| severity | 80 | 31.2 | — |
| diagnosis | 60 | 36.7 | — |
| date | 60 | 1.7 | — |
| dosage | 40 | 52.5 | — |

## 3–6. Condition B (per-calculator, note)

**Outcome taxonomy (counts of 1,100)**

| outcome | Qwen3-0.6B-Base | Qwen3-4B |
|---|---|---|
| wrong value (>50% off) | 328 | 132 |
| wrong value (<=50% off) | 301 | 155 |
| correct | 227 | 610 |
| wrong value (<=10% off) | 70 | 33 |
| no answer: degenerate loop | 60 | 66 |
| no answer: truncated (no loop) | 54 | 55 |
| wrong value (date / weeks) | 54 | 47 |
| bad format in box | 6 | 2 |

**Accuracy by category (%)**

| category | n | Qwen3-0.6B-Base | Qwen3-4B |
|---|---|---|---|
| lab test | 380 | 24.7 | 65.8 |
| risk | 240 | 9.6 | 24.6 |
| physical | 240 | 31.7 | 81.2 |
| severity | 80 | 10.0 | 45.0 |
| diagnosis | 60 | 8.3 | 48.3 |
| date | 60 | 0.0 | 15.0 |
| dosage | 40 | 52.5 | 80.0 |

**Qwen3-4B vs Qwen3-0.6B-Base: 423 problems wrong → right, 40 right → wrong (net +383)**

largest gains (problems): Ideal Body Weight +18; QTc Bazett Calculator +18; Delta Ratio +18; Albumin Corrected Delta Gap +17; Maintenance Fluids Calculations +16

largest losses (problems): Mean Arterial Pressure (MAP) -2; Wells' Criteria for DVT -2; HOMA-IR (Homeostatic Model Assessment for Insulin Resistance) -1

Newly solved, `Mean Arterial Pressure (MAP)` (gold 86.66667):

- base answered `96.66667` (wrong value (<=50% off)):

```
The mean average pressure is computed by the formula 1/3 * (systolic blood pressure) + 2/3 * (diastolic blood pressure). Plugging in the values, we get 1/3 * 120.0 mm Hg + 2/3 * 70.0 mm Hg = 96.66667 mm Hg.
Hence, the patient's mean arterial pressure is 96.66667 mm Hg. The answer is \boxed{96.66667}.
```

- Qwen3-4B answered `86.67`:

```
The mean average pressure is computed by the formula 1/3 * (systolic blood pressure) + 2/3 * (diastolic blood pressure). Plugging in the values, we get 1/3 * 120.0 mm Hg + 2/3 * 70.0 mm Hg = 86.66667 mm Hg.
Hence, the patient's mean arterial pressure is 86.66667 mm Hg. The answer is \boxed{86.67}.
```
