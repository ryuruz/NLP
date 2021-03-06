# 한국어 임베딩
### 임베딩(embedding)

+ 자연어를 숫자의 나열인 벡터로 바꾼 결과 혹은 그 일련의 과정 전체를 가리키는 용어
+ 말뭉치(corpus)의 의미, 문법 정보가 응축되어 있음
+ 벡터 -> 사칙연산 가능, 단어/문서 관련도(relevance) 계산 가능
+ 임베딩을 통해 __전이학습__ 가능
  + 전이 학습(Transfer Learning) : 특정 문제를 풀기 위해 학습한 모델을 다른 문제를 푸는 데에 재사용하는 기법
  + 대규모 말뭉치를 pretrain한 임베딩을 문서 분류 모델의 입력값으로 쓰고, 해당 임베딩을 포함한 모델 전체를 문서 분류 과제를 잘 할 수 있도록 fine tunning하는 방식

# XLNet
+ 2019년 구글 연구팀이 발표한 기법

+ 이 연구팀에서 임베딩의 최근 흐름을 크게 두 가지로 구분
  + AR(AutoRegressive) model : 데이터를 순차적으로 처리하는 기법의 총칭
    + ELMo, GPT 등 : 이전 문맥을 바탕으로 다음 단어를 예측하는 과정에서 학습하는 모델
  + 문맥을 양방향(bidirectional)으로 볼 수 없는 한계가 있음
  + AE(AutoEncoding) model : 입력값을 복원하는 기법을 두루 일컬음
    + BERT : 문장 일부에 노이즈(마스킹)를 주어 문장을 원래대로 복원하는 과정에서 학습하는 모델
      + 마스킹 처리된 단어가 어떤 단어일지 맞추는 것에 포커스를 둔다는 점에서 DAE라고 표현하기도 함
    + 양방향 모델이기는 하나, 마스킹 처리한 토큰들을 서로 독립으로 가정하기 때문에 각 토큰들의 의존 관계를 따질 수 없다는 한계가 있음
    
+ 두 모델의 한계를 극복하기 위해 Permutation Language model을 제안
  + 토큰을 랜덤으로 셔플한 뒤 그 뒤바뀐 순서를 기반으로 언어 모델을 학습하는 기법
  + 특정 토큰을 예측할 때 문장 전체 문맥을 살필 수 있게 됨
    + 해당 토큰을 제외한 문장의 부분집합 전부를 학습할 수 있다는 것
  + 단어 간 의존관계 포착에 유리
  
  ex) [1,2,3,4]의 토큰 4 개짜리 문장을 랜덤으로 뒤섞은 결과가 [3,2,4,1]일 때, 셔플된 시퀀스의 첫 번째 단어인 3번 토큰을 맞춰야 하는 상황일 때는 2,4,1 토큰은 3번 토큰이 등장한 후 나온 단어들이므로 입력으로 줄 수 없고 정답인 3번 토큰의 정보도 줄 수 없으므로 이런 상황에서는 이전 segment의 메모리 정보를 사용하게 된다.
  
  ex) 같은 문장을 셔플하여 [2,4,3,1]의 결과가 나왔고, 이 상황에서도 역시 3번 토큰을 맞춰야하는 상황이라면 메모리, 2번, 4번 토큰이 입력된다.
  
  + 퍼뮤테이션 언어 모델의 실제 구현은 토큰을 뒤섞는 것이 아니라 attention mask를 이용하여 실현됨
  + 랜덤 셔플 결과가 [3,2,4,1]과 [3,2,1,4]이고 세번째 토큰을 예측해야 할 경우, [3,2]의 같은 입력을 이용하여 다른 출력을 내야하는 모순이 존재
    + two-stream self attention 기법을 이용하여 해결
      + query stream과 content stream 두 가지를 혼합한 self attention 기법
        + query stream : 토큰과 위치 정보를 활용한 selg attention 기법
        + content stream : 기존 트랜스포머 네트워크와 거의 유사한 기법

+ Transformer-XL : XLNet 이전에 발표된 모델로, Yang et al. 연구팀에서는 이 모델의 segment recurrence와 relative position embedding 기법을 차용함
  + Segment recurrence : 기존 트랜스포머 네트워크에서 고정된 길이의 문맥 정보만 활용할 수 있다는 단점을 보완하여, 좀 더 긴 context를 활용하기 위해 제안된 기법 
    + 1. 우선 문서를 작은 segment 단위로 자른 후, 첫 번째 segment를 기존 트랜스포머 네트워크처럼 충분히 학습시킨 후 저장(cache)
    + 2. 두 번째 segment를 학습시키며, 이번 경우에는 첫 번째 segment 정보를 활용하여 계산함
      + 현재 segment를 학습할 때 고려하는 직전 segment 계산 결과를 memory라고 부름
  + 단어 쌍 사이의 거리 정보인 상대 위치를 활용함  
    + 상대 위치는 음수값이 존재하지 않음
    
### 모델 구현 
+ 생략


# Bi-LSTM Hegemony

+ Chistopher Manning 교수 2017/10/20 강연 내용
+ task가 무엇이든지 간에 attention이 적용된 BiLSTM 모델이 자연어처리 분야에서 최고 성능을 낸다
___
+ Vanilla RNN은 grdient vanishing/exploding 문제에 취약한 구조를 가지고 있음
+ LSTM (Long Short Term Memory)는 cell state를 도입하여 그래디언트 문제를 해결하고자 함
  + 직전 시점 정보와 현 시점 정보를 더해줌으로써 그래디언트가 효과적으로 흐를 수 있게 한다.

+ Vanilla Seq2Seq는 소스 문장을 벡터화하는 encoder와 인코딩된 벡터를 타겟 문장으로 변환하는 decoder로 구성됨
+ 각 encoder와 decoder는 LSTM 셀을 사용함
+ 문장 길이가 길고 층이 깊으면, encoder가 압축해야할 정보가 너무 많아져 정보 손실이 일어나고, decoder는 encoder가 압축한 정보를 초반 예측에만 사용하는 경향을 보임
+ 이 때문에 bottle-neck 문제가 발생한다고 칭하며 attention mechanism이 도입됨

+ 앞에서 뒤, 뒤에서 앞을 모두 고려하는 양방향의 네트워크를 사용하면 성능 개선에 도움이 될 수 있음

### LSTM 셀을 사용하고, encoder에 양방향 네트워크, decoder에 attention mechanism을 적용한 BiLSTM with attention 모델
+ 기계번역에서 특히 좋은 성능을 냄
  + end-to-end 학습 : output에 대한 손실을 최소화하는 과정에서 모든 파라미터들이 동시에 학습됨
  + 분산 표상 (Distributed representation) : 단어와 구(phrase) 간 유사성을 입력 벡터에 내재화해 성능을 개선함
  + 개선된 문맥 탐색 (exploitation) : LSTM과 attention으로 문장의 길이가 길어져도 성능 저하를 막을 수 있음
  + 다범주 분류에 좋은 성능을 내는 딥러닝 기법을 사용하여 문장 생성 능력이 개선됨

### BiLSTM을 적용한 연구 
+ Chen et al.(2016) : 문맥에서 질의에 대한 응답을 찾는 모델 구축
+ Eric&Manning (2017) : attention score가 높은 encoder 입력 단어를, decoder 입력에 복사해 넣으면서 decoder의 입력을 학습시키는 기법 개발


# 딥러닝 기반 자연어처리 기법의 최근 연구동향
Young, T., Hazarika, D., Poria, S., & Cambria, E. (2017). Recent Trends in Deep Learning Based Natural Language Processing. arXiv preprint arXiv:1708.02709. 논문 내용
