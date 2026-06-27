# Week 5: Logistic Regression, Regularization & Intro to Neural Networks

> Andrew Ng — Machine Learning Specialization (Course 1 ~ Course 2)
> 분류를 위한 로지스틱 회귀와 과적합을 막는 정규화, 그리고 신경망의 도입부

---

## 1. 분류 문제 (Classification Problems)

- identifying email as Spam or Not Spam
- determining a tumor is malignant or benign

linear regression은 classification(분류)에 효과적이지않다.

- 극단적인 데이터가 추가된 경우에 분류를 위해 사용되는 결정 경계(Decision Boundary)가 크게 흔들리는 문제
- 분류 문제에서는 0~1 사이의 확률값을 기대하지만 선형회귀는 0미만, 1초과의 값도 출력하는 문제 발생

출력값이 둘중 하나인 분류 문제를 binary classification이라고 한다.

---

## 2. 로지스틱 회귀 (Logistic Regression)

### 시그모이드 함수 (sigmoid function)

0~1 사이의 값을 출력한다.

$$g(z) = \frac{1}{1 + e^{-z}}$$

### 구조: 선형식(선형회귀) + 시그모이드

- 선형 단계: `z = w·x + b` → 중간 점수(score) 계산, 데이터에 맞춰 w, b가 변하면서 직선 변형
- 시그모이드 단계: `g(z) = 1 / (1 + e^(−z))` → z를 0~1 확률로 변환

---

## 3. 결정 경계 (Decision Boundary)

- 예측이 갈리는 경계선, `g(z) = 0.5` (시그모이드 함수는 z = 0에서 0.5가 된다)

→ z = w·x + b = 0 인 지점이 결정 경계

- 만약 데이터에 제곱 항(x^2, x^3) 같은 다항식이 포함되면, 직선이 아니라 원이나 포물선 같은 복잡한 곡선 형태의 결정 경계도 만들 수 있다.

---

## 4. 로지스틱 회귀의 비용 함수 (Cost Function)

손실(데이터 1개에 대해) vs 비용(전체 데이터에 대해): 손실은 데이터 한개에서의 점수를 의미하고 비용함수는 그 손실들을 전부 더해서 평균 낸 점수.

$$J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} L\left(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}\right)$$

선형회귀에서 쓰이던 제곱오차 비용 함수를 로지스틱 회귀(시그모이드)에 쓰게 되면 비용함수가 convex하지 않은 울퉁불퉁하게 되기 때문에 경사하강법 X.

→ logistic regression을 위한 전용 손실함수(로그손실 함수)를 만듬.

$$
L\left(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}\right) =
\begin{cases}
-\log\left(f_{\vec{w},b}(\vec{x}^{(i)})\right) & \text{if } y^{(i)} = 1 \\
-\log\left(1 - f_{\vec{w},b}(\vec{x}^{(i)})\right) & \text{if } y^{(i)} = 0
\end{cases}
$$

- y = 1일때, L = -log(f(x)) → 1에 가깝게 예측하도록 유도
- y = 0일때, L = -log(1-f(x)) → 0에 가깝게 예측하도록 유도

손실함수를 하나의 식으로 합치면:

$$L\left(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}\right) = -y^{(i)}\log\left(f_{\vec{w},b}(\vec{x}^{(i)})\right) - (1-y^{(i)})\log\left(1 - f_{\vec{w},b}(\vec{x}^{(i)})\right)$$

이 손실함수를 쓰게되면 비용함수가 convex(볼록)하게 그려지기 때문에 경사하강법이 잘 작동할 수 있다.

최종 목표: 비용함수 J(w,b)를 가장 작게 만드는 w, b를 찾는 것.

---

## 5. 로지스틱 회귀의 경사하강법 (Gradient Descent)

$$
\begin{aligned}
&\text{repeat} \lbrace \\
&\quad w_j = w_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} \left(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}\right) x_j^{(i)} \right] \\
&\quad b = b - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} \left(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}\right) \right] \\
&\rbrace \quad \text{simultaneous updates}
\end{aligned}
$$

선형회귀와 로지스틱회귀의 경사하강법 업데이트 공식은 똑같아 보일 수 있지만, 실제로 f(x)의 정의가 다르다.

- 선형회귀: f(x) = `w·x + b`
- 로지스틱회귀: f(x) = `g(w·x + b)`

---

## 6. Scikit-Learn으로 로지스틱 회귀

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()

lr_model.fit(X, y)            # 모델에게 훈련 데이터 주고 학습
y_pred = lr_model.predict(X)  # 모델에게 훈련 데이터를 주고 예측
lr_model.score(X, y)          # 모델이 예측한 결과와 실제 정답을 비교 채점
```

---

## 7. 과적합 vs 과소적합 (Overfitting vs Underfitting)

- Underfitting (= high bias): 모델이 훈련 데이터셋에 적합하지 않은 것, 모델이 너무 단순해서 데이터의 패턴을 못 담음
- Generalization(일반화): 학습 데이터셋에도 포함되어 있지 않은 완전히 새로운 데이터에 대해서도 정확한 예측할 하는 것
- Overfitting (= high variance): 모델이 훈련 데이터셋에 너무 완벽하게 적합한 것, 모델이 너무 복잡해서 훈련데이터의 우연/잡음 까지도 외워버린다 → 데이터 조금만 바뀌어도 결과 달라짐

---

## 8. 과적합 해결 (Addressing Overfitting)

1. Collect more data
2. Not using so many of the polynomial features = feature selection → 특성 중에서 가장 유용한 특성들만 골라 사용하는 것
3. Regularization (정규화): 특정 특징들을 완전히 삭제해버리는 대신 그 특징들이 모델에 미치는 영향력을 감소시키는 방법, 가중치 수치들을 작게 만든다

- 작은 파라미터는 더 단순한 모델을 의미한다. 각 특성이 출력에 주는 영향이 약해지므로 모델 전체가 완만해지고 단순해진다.
- 가중치가 크면 개별 데이터 하나하나에 모두 예민하게 반응하기에 쓸데없는 노이즈까지 학습해버린다.

---

## 9. 정규화를 적용한 비용 함수 / 경사하강법

### 정규화된 선형 회귀 (regularized linear regression)

입력 특성이 매우 많은 경우에, 어떤 파라미터가 중요할지 미리 알 수 없기 때문에 전부 다 조금씩 영향력을 감소시킨다. → $\frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$ 라는 새 항을 더해서:

$$J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}\right)^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

λ(람다): 정규화 파라미터. 람다의 값은 모델을 데이터에 맞추려는 학습 목표(훈련 오차 최소화)와 과적합을 방지하려는 제어 목표(가중치 크기 최소화) 사이의 균형을 조절하고 결정한다.

→ 람다 값이 클수록 과적합 방지(규제)에 무게를 두어 모델이 단순해지고, 0에 가까울수록 데이터 자체를 완벽하게 맞추는 데 집중하게 된다.

정규화된 비용함수를 이용한 선형 회귀의 경사하강법:

$$
\begin{aligned}
&\text{repeat} \lbrace \\
&\quad w_j = w_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} \left(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}\right) x_j^{(i)} + \frac{\lambda}{m} w_j \right] \\
&\quad b = b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}\right) \\
&\rbrace
\end{aligned}
$$

### 정규화된 로지스틱 회귀 (regularized logistic regression)

$$J(\vec{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log\left(f_{\vec{w},b}(\vec{x}^{(i)})\right) + (1-y^{(i)}) \log\left(1 - f_{\vec{w},b}(\vec{x}^{(i)})\right) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

경사하강법의 업데이트 공식 형태는 정규화된 선형 회귀와 동일하다 (f의 정의만 다름).

---

## 10. 신경망 도입 (Intro to Neural Networks)

동기: trying to build software to mimic the brain.

### 특징

- 선형회귀와 로지스틱 회귀와 같은 알고리즘들로는 데이터의 양을 늘린다고해서 성능이 향상되지 않았지만, 신경망은 데이터가 많아질수록 신경망의 규모도 대응되서 커지면 성능이 향상된다.
- 신경망의 은닉층은 선형/로지스틱 회귀에서와 달리 특성을 지정하지 않아도 학습을 통해 데이터에서 예측에 유용한 특성을 스스로 만들어낸다 (= 자동 특성 학습).

### 구성 요소

- neuron: 특정 숫자 값을 1개 또는 여러개 입력 받고(input), 그에 대응되는 결과 값을 output
- layer: grouping of neurons which takes as input the same or similar features (동일하거나 비슷한 값들을 입력으로 받는 뉴런들의 집합)
- activations: 들어온 신호(입력값)들의 총합을 보고 다음 단계로의 동작을 결정하는 스위치

Neural Network는 크게 입력층(Input Layer), 은닉층(Hidden Layer), 출력층(Output Layer)로 구성.

모든 layer에 있는 각각의 뉴런들은 모든 입력 변수를 전부 받아들인다 (모든 특성에 접근).

입력층으로 특징값들을 가진 벡터가 주어지고 → 은닉층의 입력으로 들어가면서 은닉층에서 또 다시 특정 벡터로 output → 출력층이 해당 output을 입력으로 받아 하나의 숫자로 출력함으로써 최종 prediction 만듬.

### 순전파 (Forward Propagation)

각 뉴런의 연산 결과(입력값에 가중치와 편향을 연산)를 입력층에서 출력층 방향으로 데이터 전달하는 과정 (left to right).

---

## 11. TensorFlow

### Sequential 모델

층이 여러개 있는 인공 신경망 만드는 편리한 수단 (Dense는 layer의 또 다른 이름).

```python
model = Sequential([
    Dense(units=25, activation='relu'),    # 1번째 층 (은닉층 1)
    Dense(units=15, activation='relu'),    # 2번째 층 (은닉층 2)
    Dense(units=1,  activation='sigmoid')  # 3번째 층 (로지스틱 레이어 / 출력층)
])
```

Dense function: 이전 layer의 activation값을 입력으로 받아서 다음 layer의 input값인 새로운 activation값을 출력한다.

### 주요 함수

- `model.summary()`: 모델의 layer와 파라미터의 개수를 보여준다
- `model.get_weights()`: 내부에 저장하고 있는 가중치와 편향값을 확인해주는 함수
- `model.predict(x_new)`: 학습 데이터가 아닌 새로운 데이터에 대해 inference를 수행
- `model.compile()`: 손실함수를 정의하고 모델을 최적화할 방법 지정
- `model.fit(x, y)`: 경사하강법 실행해 주어진 데이터에 맞는 가중치 학습

### 데이터 형태 관련

- 모델에 데이터를 입력할 때 차원확장 과정을 거쳐야한다. 단 1개의 데이터를 넣더라도, 맨 앞의 차원을 데이터의 개수를 나타내야하기에 `[데이터의 개수, 특징 수]` 형태로 변환.
- 텐서플로우에서 가중치 W의 형태는 `[입력 특징의 개수, 출력 뉴런의 개수]`로 표현된다.
