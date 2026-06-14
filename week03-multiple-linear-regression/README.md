# Week 3: Multiple Linear Regression & Vectorization

> Andrew Ng — Machine Learning Specialization (Course 1)
> 입력 특징이 여러 개인 선형 회귀, 그리고 이를 빠르게 계산하는 벡터화

---

## 1. 다중 선형 회귀 (Multiple Linear Regression)

입력 특징(변수)이 여러 개인 선형 회귀 모델.

$$f_{\vec{w},b}(\vec{x}) = w_1x_1 + w_2x_2 + w_3x_3 + \cdots + w_nx_n + b$$

- 가중치 $w$를 하나의 벡터 $\vec{w} = [w_1, w_2, \cdots, w_n]$ 로 표현할 수 있다.
- 입력 $x$ 도 하나의 벡터 $\vec{x} = [x_1, x_2, \cdots, x_n]$ 로 표현할 수 있다.
- 결국 이 공식은 두 벡터의 내적(dot product) 으로 표현된다:

$$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$

### 표기법 비교

| | Previous notation | Vector notation |
|---|:---|:---|
| Parameters | $w_1, \cdots, w_n$, $b$ | $\vec{w} = [w_1 \cdots w_n]$, $b$ |
| Model | $f_{\vec{w},b}(\vec{x}) = w_1x_1 + \cdots + w_nx_n + b$ | $f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$ |
| Cost function | $J(w_1, \cdots, w_n, b)$ | $J(\vec{w}, b)$ |
| Gradient descent | $w_j = w_j - \alpha \dfrac{\partial}{\partial w_j} J(\vec{w}, b)$ | $\vec{w} = \vec{w} - \alpha \dfrac{\partial}{\partial \vec{w}} J(\vec{w}, b)$ |

---

## 2. 벡터화 (Vectorization)

### ① 벡터화 없이 — 반복문 사용

```python
f = 0
for j in range(0, n):
    f = f + w[j] * x[j]
f = f + b
```

### ② 벡터화 없이 — 각 원소 곱을 직접 나열

```python
f = w[0] * x[0] + w[1] * x[1] + ...
```

→ 파라미터 개수 $n$ 이 커질수록 비효율적이고, 데이터셋이 커질수록 연산 속도도 느려진다.

### ③ 벡터화 — `np.dot`

```python
f = np.dot(w, x) + b
```

- 반복문 없이 데이터를 통째로 묶어 계산한다.
- 컴퓨터의 병렬 연산을 활용 → 한 줄로 표현되고 속도도 빠르다.

---

## 3. `compute_gradient()`

입력된 데이터(전체든 일부 배치든)를 바탕으로:

1. 각 데이터마다 오차를 줄이기 위해 가중치가 이동해야 할 방향(기울기)을 구하고
2. 그 기울기들을 모두 더한 뒤 데이터 개수로 나눠 평균을 내서
3. 모델이 최종적으로 수정해야 할 "각 가중치들의 평균 기울기" 를 계산해 반환하는 함수.

---

## 4. NumPy 기초

### `.shape`

배열이 몇 행, 몇 열로 이루어져 있는지 보여준다.

- `(4,)` 처럼 콤마 뒤가 비어 있는 형태
  → "이 배열은 1차원이고, 그 차원에 데이터가 4개 들어있다"는 NumPy만의 표기법.

### 벡터의 사칙연산 조건

$+, -, \times, \div$ 를 할 때는 반드시 둘 중 하나를 만족해야 한다:

- 서로의 shape가 똑같거나
- 한쪽 벡터의 원소 개수가 1개이거나

### `reshape(-1)`

`-1` 을 인자로 넣으면 "이 자리에 들어갈 숫자는 전체 개수에 맞춰 자동으로 계산해줘"라는 의미.

```python
a = np.arange(6)   # [0 1 2 3 4 5]
a.reshape(-1, 2)   # → shape (3, 2)
a.reshape(2, -1)   # → shape (2, 3)
```

---

## 번외: 정규 방정식 (Normal Equation)

행렬 계산 공식을 이용해 최적의 가중치 $w$ 와 편향 $b$ 를
반복 없이(without iterations) 단번에 계산해서 찾아내는 방식.

- 오직 선형 회귀에서만 적용할 수 있다.
