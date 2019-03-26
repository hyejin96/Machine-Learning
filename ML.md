머신러닝
# __머신 러닝의 개념과 용어__

## Supervised / Unsupervised learning
### Supervised learning
- 데이터를 가지고 학습하는 것
- 이미지를 주었을 때 이것이 무엇인지 찾아내는 것
ex) 고양이 그림을 주고 이게 고양이인지 맞추는 것
ex)
- Image Labeling
- Email Spam Filter
- Predicting exam score

### Unsupervised learning
- 구글 뉴스는 자동적으로 비슷한 뉴스를 그룹핑한다.
- 혹은 단어들 가운데 비슷한 단어들을 그룹핑한다.
- 학습이 아닌 데이터를 분석해서 학습하는 것

~~~ 
> Training Data Set : 반드시 필요한 것
> Supervised learning 통해 학습한 데이터를 표형태로 나타낸 것
> 그것을 통해 x에 대한 값을 예상할 수 있다.
~~~

## AlphaGo(알파고)
- 기존에 바둑판에서 사람들이 한 수를 다 학습한다.
- Supervised Learning이라 할 수도 있고, 
- 이세돌이 놓은 수에 대한 수비 값을 내 놓는데 그것을 Training Data Set에서 가져온다.

## Supervised learning의 종류
- regression : 0 ~ 100 숫자로 예측하는 경우(시험성적)
- Binary Classification(분류) : pass / non-pass에 대해 둘 중 하나로 예측하는 것
  (label이 두 개)
- multi-label Classification : A, B, C, D, F등 여러개의 Label에서 하나를 선택한다.

## TensorFlow : 
- 구글에서 만든 오픈소스 라이브러리
- data flow graph이다.
- python이라는 프로그램을 가지고 사용할 수 있다.
- 코드 수정 없이 CPU/GPU 모드로 동작
- 아이디어 테스트에서 서비스 단계까지 이용 가능
- 계산 구조와 목표 함수만 정의하면 자동으로 미분 계산을 처리

## data flow graph란?
- 그래프란 노드와 엣지로 이루어져있는데
- 노드가 하나의 operation이다.
- 엣지는 하나의 데이터를 말한다.(= tensor, 텐서라고도 불림)

## 실습
```python
Hello TensorFlow!
$ python
>>> import tensorflow as tf # tensorflow 임포트 
>>> hello = tf.constant('Hello, TensorFlow!')   
>>> sess = tf.Session() # session을 만들어야 실행할 수 있다.
>>> print sess.run(hello)
'b' Hello, TensorFlow! # 'b'는 Binary literals라 하여 3에서만 나타나는데 신경쓰지 않아도 된다.
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> print sess.run(a+b)
42
>>>
```

## Computational Graph
<img src="https://camo.githubusercontent.com/d3abc24f14f3d7dca08dcba9815e00ee68003a7b/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f67657474696e675f737461727465645f61646465722e706e67" width="200">

``` python
# 1. 그래프를 만들었다.(Build Graph)
node1 = tf.constant(3.0, tf.float32) # tf.float32 : Data type
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1, node2)

# 결과값이 나오지 않는다.
print("node1:", node1, "node2:", node2) # node1: Tensor("Const_1:0", shape=(), dtype=float32) node2: Tensor("Const_2:0", shape=(), dtype=float32)
print("node3: ", node3) # node3:  Tensor("Add:0", shape=(), dtype=float32)

# 2. session을 만들고, sess.run을 통해 op를 했다.
# 3. 결과값 리턴
sess = tf.Session() # 섹션 생성
print("sess.run(node1, node2): ", sess.run([node1, node2])) # sess.run(node1, node2):  [3.0, 4.0]
print("sess.run(node3): ", sess.run(node3)) # sess.run(node3):  7.0
```
## TensorFlow Mechanics
 1. 그래프를 만들고 (bulid graph)
 2. sess.run을 통해 그래프를 생성시키고(op),
 3. 결과값 리턴

## Placeholder
- 데이터를 변경하기 위한 수단
- placeholder로 노드 만들기
- adder_node 노드 만들기
- tensorFlow에서 adder_node에서 값을 알려줘 하는 것을 feed_dict로 값을 넘겨준다.
```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5})) # 7.5
print(sess.run(adder_node, feed_dict={a: [1,3], b: [2, 4]})) # [ 3.  7.]
```

## Tensor : [1, 2, 3]과 같은 배열

### Ranks
| Rank | Math entity | Python Example |
|:---:|:---:|:---:|
| `0` | Scalar(magnititude only) | `s = 483` |
| `1` | Vector(magnitude and direction) | `v = [1, 2, 3]` |
| `2` | Matrix(table of number) | `m = [[1, ,2, 3], [4, 5, 6]]` |
| `n` | n-Tensor | ... |

### Shapes
| Rank | Shape | Dimension number | Example |
|:---:|:---:|:---:|:---:|
| `0` | [] | 0-D |	A 0-D tensor. A scalar. |
| `1` | [D0] |1-D |	A 1-D tensor with shape [5]. |
| `2` | [D0, D1] | 2-D |	A 2-D tensor with shape [3, 4]. |
| `n` | [D0, D1, ... Dn-1] |	n-D |	A tensor with shape [D0, D1, ... Dn-1]. |

### Types
- tf.float32를 많이 사용한다.
- tf.int32를 많이 사용한다.

## Linear Regression(선형 회귀)의 Hypothesis와 cost

1. Regression: 0~100의 수치를 학습데이터를 통해 학습해서 결과값을 예측해주는 것
2. Linear Hypothesis : Linear Regression를 학습한다는 것은 가설을 세운다는 것이다. 
 - 어떤 데이터가 있다면 그 데이터에 대하여 선으로 표현하여 어떤 선이 결과값에 더 맞을까를 생각하는 것
 - H(x) = Wⅹ + b
   W(weight) : 기울기
   b(bias) : 절편
 - Linear Regression에서 사용하는 1차원 방정식을 가리키는 용어
 - W와 b는 계속 바뀌며, 최종 결과로 나온 가설을 모델이라 부르며, "학습되었다"라고 한다.
 <img src = "https://t1.daumcdn.net/cfile/tistory/2669EA3E5790FD3317" width = "150">

3. Cost function(= Loss function, 비용함수) : 
 - 직선으로부터 각각의 데이터까지의 거리를 측정하는 것을 cost라 하며, 가장 작은 값을 찾으면 목표 달성이다.
 - 루프를 돌 때마다W, b를 비용이 적게 발생하는 방향으로 수정하게 된다.
 - H(x) - y --> +값 또는 -값이 나올 수도 있기때문에 좋지 않은 공식이다.
 - (H(x) - y)²으로 계산하여 측정한다.
 - 하지만 값은 여러개가 존재할 수 있기 때문에 sum()함수를 사용하여 구한다.
 - 따라서 아래와 같이 정의 할 수 있다.
 <img src = "https://t1.daumcdn.net/cfile/tistory/992487335A0BDCC42D" width = "200">
 - 데이터까지의 합계를 m으로 나눈 결과는 해당 가설에 대한 비용이다.
 - 가장 작은 값을 가지는 W,b를 가지는 값을 찾는 것이 학습이 된다.
 - Goal : Minimize cost
  __minimize cost(W,b)__




