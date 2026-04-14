# Research Direction for a Paper

이 문서는 현재 `Gawee` 프로젝트를 바탕으로, 실제 AI 컴파일러 개발 과정에서 자연스럽게 나오는 논문 문제 정의를 정리한 것이다.

핵심 목표는 다음과 같다.

1. "모델이 M1에서 돌아간다 / 안 돌아간다" 수준의 약한 문제를 피한다.
2. 현재 코드베이스가 이미 가진 강점과 빈 부분을 분리한다.
3. 기존 연구와 자연스럽게 이어지는 문제 정의를 만든다.
4. 실제로 M1급 하드웨어에서 재현 가능한 실험 설계를 만든다.

---

## 1. 지금 프로젝트가 이미 가진 것

현재 프로젝트는 아래 구조를 이미 갖고 있다.

- PyTorch FX 기반 frontend
- 자체 graph IR (`Gawee IR`)
- graph-level rewrite pass
- FLOPs / memory-access 추정 기반 cost analysis
- JSON export
- MLIR dialect 및 Linalg/SCF/LLVM lowering 골격

즉, 이 프로젝트는 이미 "컴파일러의 모양"은 갖추고 있다. 따라서 논문 문제는 "컴파일러를 만들 수 있는가?"가 아니라, 다음과 같은 더 좁고 더 현실적인 질문으로 가야 한다.

- 어떤 rewrite가 실제로 유효한가?
- 어떤 정적 지표가 실제 runtime을 설명하는가?
- shape 변화나 frontend noise가 optimization decision을 얼마나 흔드는가?

---

## 2. 약한 문제 정의와 강한 문제 정의

### 2.1. 약한 문제 정의

다음과 같은 문제 정의는 첫 논문 주제로 약하다.

- "ResNet-18은 edge device에서 무겁다"
- "메모리가 부족해서 M1에서 inference가 어렵다"
- "연산 수를 줄이면 빨라질 것이다"

이런 문제 정의가 약한 이유는 명확하다.

- ResNet-18은 M1에서 충분히 실행 가능하다.
- 단순 node count 감소는 컴파일러 연구의 핵심 기여가 아니다.
- FLOPs 감소와 실제 latency 감소는 일반적으로 동일하지 않다.

즉, "실행 가능 여부"는 문제라기보다 전제에 가깝다.

### 2.2. 강한 문제 정의

현재 프로젝트에서 더 자연스러운 문제는 아래와 같다.

> 합법적인 graph rewrite가 존재하더라도, 그것이 target hardware에서 실제 latency improvement를 보장하지 않는다. 따라서 compiler는 rewrite의 legality뿐 아니라 hardware-specific profitability를 판단해야 한다.

또는 더 구체적으로:

> 현재 Gawee는 graph rewrite 전후의 FLOPs와 memory access를 추정할 수 있지만, 이 정적 지표만으로 Apple M1에서의 실제 runtime improvement를 설명하거나 예측할 수 없다.

이 문제 정의는 실제 AI compiler 연구와 잘 맞는다. 현대 DL compiler는 단순히 "변환 가능한 rewrite"를 많이 가지는 것이 아니라, "무엇을 적용해야 실제로 이득인지"를 판단해야 하기 때문이다.

---

## 3. 왜 FLOPs / memory-access 감소가 runtime 개선을 보장하지 않는가

이 부분은 문제 정의의 핵심 배경이다.

### 3.1. FLOPs는 간접 지표다

FLOPs는 계산량을 설명할 수는 있지만, 실행 시간 자체를 직접 설명하지는 못한다. 실제 latency는 다음 요소의 영향을 받는다.

- memory access cost
- cache locality
- operator implementation quality
- vectorization / threading behavior
- tensor layout
- kernel launch or dispatch overhead
- target hardware 특성

즉, 두 graph가 비슷한 FLOPs를 가지더라도 latency는 크게 다를 수 있고, 반대로 FLOPs가 줄어도 latency는 거의 줄지 않을 수 있다.

### 3.2. elementwise / bookkeeping / memory-bound op의 비중

DL graph에서 성능은 큰 convolution만으로 결정되지 않는다. 다음 연산들도 실제 latency에 영향을 준다.

- elementwise add / relu
- reshape / flatten / cat
- layout-sensitive op
- graph를 분할시키는 작은 op
- backend library가 이미 최적화하고 있는 op 경계

frontend rewrite가 graph를 더 "예쁘게" 만들어도, backend가 이미 비슷한 최적화를 하고 있거나 메모리 바운드 구간이 지배적이면 end-to-end 이득은 작아질 수 있다.

### 3.3. shape 의존성

rewrite profitability는 shape에 따라 달라진다.

- batch size가 바뀌면 memory traffic과 arithmetic intensity가 바뀐다.
- spatial dimension이 바뀌면 fusion 이득이 달라진다.
- 작은 tensor에서는 compute보다 dispatch overhead가 더 크게 보일 수 있다.

따라서 single example input으로 계산한 정적 cost는 실제 deployment 조건을 충분히 대변하지 못할 수 있다.

---

## 4. 기존 연구가 이미 제기한 문제

아래 논문들은 "FLOPs와 실제 latency가 다르다", "target hardware-aware cost model이 필요하다", "graph rewrite는 cost-based selection이 필요하다"는 문제를 직접적으로 제기한다.

### 4.1. ShuffleNet V2 (ECCV 2018)

Ningning Ma et al., "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"

이 논문은 architecture design이 보통 FLOPs 같은 indirect metric에 의존하지만, 실제 speed는 memory access cost와 platform characteristics에도 크게 좌우된다고 지적한다. 이 논문은 본 프로젝트의 문제 정의에 직접 연결된다. 즉, `FLOPs 감소 == latency 감소`라는 가정은 이미 깨져 있다.

링크:

- ECCV open access: https://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html
- ECVA page: https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.php

이 논문이 주는 메시지:

- 직접 측정 지표를 봐야 한다.
- memory access와 hardware 특성이 중요하다.
- indirect proxy만으로는 최적화를 정당화할 수 없다.

### 4.2. MnasNet (CVPR 2019)

Mingxing Tan et al., "MnasNet: Platform-Aware Neural Architecture Search for Mobile"

이 논문은 mobile device latency를 FLOPs 같은 proxy가 아니라 target device에서 직접 최적화 대상으로 둔다. 이는 "device-specific profitability"라는 관점을 강하게 뒷받침한다.

링크:

- Google Research: https://research.google/pubs/mnasnet-platform-aware-neural-architecture-search-for-mobile/

이 논문이 주는 메시지:

- 실제 deployment hardware를 optimization loop에 넣어야 한다.
- mobile/edge latency는 hardware-aware하게 다뤄야 한다.

### 4.3. FBNet (CVPR 2019)

Bichen Wu et al., "FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search"

이 논문 역시 FLOP count가 actual latency를 항상 반영하지 않는다고 본다. latency-aware design space exploration 자체가 독립적인 연구 주제가 될 수 있음을 보여준다.

링크:

- CVF open access: https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_FBNet_Hardware-Aware_Efficient_ConvNet_Design_via_Differentiable_Neural_Architecture_Search_CVPR_2019_paper.html

이 논문이 주는 메시지:

- 더 적은 FLOPs의 모델이 더 빠르다는 보장은 없다.
- target hardware 측정 또는 calibrated model이 필요하다.

### 4.4. TVM (OSDI 2018)

Tianqi Chen et al., "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning"

TVM은 graph-level optimization뿐 아니라 low-level program optimization을 위해 learning-based cost model을 도입한다. 이는 DL compiler가 단순 규칙 기반이 아니라 cost-model 기반 decision을 해야 함을 보여준다.

링크:

- USENIX: https://www.usenix.org/conference/osdi18/presentation/chen

이 논문이 주는 메시지:

- DL compiler는 hardware-aware optimization 문제를 가진다.
- cost model은 선택적 최적화의 핵심 구성 요소다.

### 4.5. Ansor (OSDI 2020)

Lianmin Zheng et al., "Ansor: Generating High-Performance Tensor Programs for Deep Learning"

Ansor는 low-level tensor program optimization에서 learned cost model을 사용한다. 이 논문은 "좋은 성능의 코드를 얻기 위해서는 검색과 cost model이 필요하다"는 점을 분명히 한다. 비록 operator schedule 수준의 연구이지만, graph rewrite profitability 문제에도 직접적인 힌트를 준다.

링크:

- USENIX: https://www.usenix.org/conference/osdi20/presentation/zheng

이 논문이 주는 메시지:

- 정적 규칙만으로는 충분하지 않다.
- target hardware에 맞춘 성능 추정이 필요하다.

### 4.6. TASO (SOSP 2019)

Zhihao Jia et al., "TASO: Optimizing Deep Learning Computation with Automated Generation of Graph Substitutions"

TASO는 자동 생성된 graph substitution을 cost-based search로 탐색한다. 이 논문은 graph rewrite가 많을수록 좋은 것이 아니라, search와 cost evaluation이 함께 있어야 한다는 점을 보여준다.

링크:

- Project page: https://catalyst.cs.cmu.edu/projects/taso.html

이 논문이 주는 메시지:

- graph substitution은 search space를 만든다.
- 어떤 rewrite를 선택할지 cost-based decision이 필요하다.

### 4.7. TENSAT (MLSys 2021)

Yichen Yang et al., "Equality Saturation for Tensor Graph Superoptimization"

이 논문은 tensor graph optimization을 equality saturation으로 확장한다. 여기서도 본질은 동일하다. 많은 equivalent graph 중 무엇을 선택할지 결정하는 cost function이 필요하다.

링크:

- Project page: https://www.mwillsey.com/papers/tensat

이 논문이 주는 메시지:

- legality는 시작점일 뿐이다.
- 최종 선택은 cost function의 품질에 좌우된다.

### 4.8. The Deep Learning Compiler: A Comprehensive Survey

Mingzhen Li et al., "The Deep Learning Compiler: A Comprehensive Survey"

이 survey는 DL compiler를 frontend / IR / backend / optimization 측면에서 체계적으로 정리한다. Gawee의 현재 위치를 설명할 때 좋은 배경 논문이다.

링크:

- arXiv entry via dblp: https://dblp.org/rec/journals/corr/abs-2002-03794

이 논문이 주는 메시지:

- DL compiler 연구는 multi-level IR, graph optimization, backend code generation을 모두 포함한다.
- Gawee는 이미 survey가 말하는 전형적인 구조를 따르고 있다.

---

## 5. 현재 Gawee에서 자연스럽게 나오는 핵심 문제

현재 코드베이스를 기준으로 보면, 자연스러운 논문 문제는 아래 둘이다.

## 5.1. Problem A: Profitability-Aware Graph Rewrite on Apple M1

### 문제 배경

Gawee는 이미 다음을 할 수 있다.

- graph를 IR로 변환한다
- rewrite pass를 적용한다
- FLOPs / bytes read / bytes write를 계산한다
- MLIR로 lowering한다

하지만 아직 다음은 없다.

- pass 전후 correctness를 정량적으로 검증하는 체계
- pass 전후 runtime을 반복 측정하는 harness
- rewrite profitability를 설명하는 feature/model
- shape 변화에 따른 pass 효과 분석

즉, 현재 Gawee는 `legal rewrite`는 할 수 있지만 `profitable rewrite`를 판별하지는 못한다.

### 문제 정의

> On Apple M1-class CPUs, graph rewrites that reduce node count, FLOPs, or estimated memory traffic do not necessarily yield proportional end-to-end latency improvement. The compiler therefore needs a profitability criterion, not just legality-preserving rewrite rules.

### 왜 이 문제가 가치가 큰가

- 실제 compiler는 rewrite rule collection보다 rewrite selection이 더 어렵다.
- production compiler에서는 모든 합법적 rewrite를 무조건 적용하지 않는다.
- hardware-aware profitability는 compiler engineering의 핵심 문제다.

### Gawee에서의 구체적 연구 질문

1. 현재 cost model의 어떤 feature가 실제 M1 latency와 가장 잘 상관하는가?
2. node count, FLOPs, bytes 중 어느 지표가 가장 약한가?
3. 어떤 graph rewrite가 M1에서 consistently profitable한가?
4. shape가 달라질 때 profitability decision은 얼마나 흔들리는가?

### 필요한 개발 방향

- benchmark harness 추가
- equivalence test 추가
- repeated latency measurement 추가
- pass ablation 실험 추가
- feature extraction 추가
- profitability predictor 또는 heuristic 추가

---

## 5.2. Problem B: Shape-Aware Cost Model Calibration

### 문제 배경

현재 frontend는 concrete example input에 크게 의존해 shape를 얻는다. 이 방식은 compiler prototype 단계에서는 합리적이지만, 논문 관점에서는 다음 문제가 있다.

- representative input 하나만으로 cost를 계산한다
- shape variation에 대한 robustness가 없다
- partial-known / unknown shape에 대한 모델이 없다

결과적으로, 현재 cost model은 "한 입력에 대한 정적 계산기"에 가깝고, "deployment 조건을 반영하는 predictor"는 아니다.

### 문제 정의

> The profitability of graph rewrites is shape-dependent, yet prototype compilers often estimate costs from a single concrete input. This can make pass-selection decisions brittle under realistic input-size variation.

### 왜 이 문제가 가치가 큰가

- 실제 inference workload는 batch와 spatial size가 변할 수 있다.
- shape variation은 memory traffic과 cache behavior를 바꾼다.
- profitability prediction이 shape에 민감하다면, compile-time decision은 쉽게 틀릴 수 있다.

### Gawee에서의 구체적 연구 질문

1. single-shape calibration은 얼마나 취약한가?
2. shape range를 사용하면 더 robust한 profitability decision이 가능한가?
3. shape uncertainty를 cost model 안에 confidence/range 형태로 넣을 수 있는가?

### 필요한 개발 방향

- shape sweep benchmark 추가
- per-shape cost / latency 데이터셋 생성
- shape-sensitive feature 설계
- 단일 값 cost가 아니라 range-aware cost 설계

---

## 6. activation lifetime / peak memory는 어디에 위치하는가

이 주제는 의미가 있지만, 현재 프로젝트에서 첫 논문의 메인 문제로 두기에는 상대적으로 약할 수 있다.

그 이유는 다음과 같다.

- "ResNet-18이 M1에서 메모리 때문에 실행 불가"는 성립하지 않는다.
- memory planning만으로 문제 정의를 세우면 임팩트가 작아질 수 있다.
- 현재 코드베이스의 더 큰 공백은 memory planner보다 benchmark/profitability/cost validation 쪽이다.

하지만 이 주제는 완전히 버릴 필요가 없다. 오히려 다음처럼 보조 축으로 넣으면 좋다.

- memory traffic feature
- tensor lifetime overlap 통계
- peak live bytes 추정
- memory-bound 여부를 판단하는 feature

즉, lifetime analysis는 메인 contribution이 아니라 profitability model의 feature engineering 일부로 넣는 편이 더 자연스럽다.

---

## 7. 현재 프로젝트에 가장 자연스러운 논문 서사

가장 자연스러운 서사는 아래 흐름이다.

1. Gawee는 graph rewrite와 IR lowering이 가능한 AI compiler prototype이다.
2. 현재 optimization 효과는 node count/FLOPs/bytes 감소로만 설명된다.
3. 하지만 기존 연구는 FLOPs와 실제 latency의 불일치를 이미 지적해 왔다.
4. 따라서 Gawee에서도 rewrite의 효과를 M1에서 실제 runtime 기준으로 다시 정의해야 한다.
5. 이를 위해 profitability-aware benchmark/cost model/shape-aware evaluation을 도입한다.
6. 최종적으로 "어떤 rewrite가 언제 실제로 이득인지"를 설명하는 compiler methodology를 제안한다.

이 서사의 장점은 다음과 같다.

- 현재 구현과 바로 이어진다.
- 시스템 구축과 논문 문제 정의가 분리되지 않는다.
- M1 단일 장비에서도 충분히 실험 가능하다.
- graph compiler와 MLIR lowering 프로젝트라는 현재 정체성을 유지한다.

---

## 8. 추천 논문 제목 방향

### 방향 1: profitability 중심

- Toward Profitability-Aware Graph Rewriting for Edge AI Compilers on Apple M1
- When Do Graph Rewrites Actually Help? A Profitability Study on an MLIR-Based AI Compiler

### 방향 2: cost model 중심

- Bridging Static Graph Cost and Real CPU Latency in an MLIR-Based AI Compiler
- Shape-Aware Cost Modeling for Graph Rewrite Selection on Apple M1

### 방향 3: 두 문제를 합친 방향

- Profitability-Aware and Shape-Aware Graph Optimization for an Edge AI Compiler

---

## 9. 논문 작성 전 우선 구현해야 할 것

논문 문제 정의를 강하게 만들기 위해, 다음 순서로 개발하는 것이 적절하다.

1. pass correctness / equivalence test 작성
2. benchmark harness 작성
3. pass ablation 실험 자동화
4. shape sweep 실험 자동화
5. cost-latency correlation 분석
6. profitability heuristic 또는 predictor 설계

이 중 1-4가 없으면 논문 주장 자체가 약해진다.

---

## 10. 핵심 결론

현재 Gawee에서 가장 자연스러운 논문 문제는 "더 많은 rewrite를 구현하는 것"이 아니다.

더 중요한 문제는 다음이다.

- 어떤 rewrite가 실제로 이득인지 판단할 수 있는가?
- 정적 cost 지표와 실제 M1 runtime 사이의 간극을 어떻게 줄일 것인가?
- shape 변화가 그 판단을 얼마나 흔드는가?

즉, 이 프로젝트는 앞으로 `graph optimization의 수집`보다 `graph optimization의 선택 기준` 쪽으로 발전해야 한다. 그것이 현재 코드베이스와 기존 연구 모두에 가장 자연스럽게 연결되는 방향이다.

---

## References

1. Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun. "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design." ECCV 2018.
   Link: https://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html

2. Mingxing Tan et al. "MnasNet: Platform-Aware Neural Architecture Search for Mobile." CVPR 2019.
   Link: https://research.google/pubs/mnasnet-platform-aware-neural-architecture-search-for-mobile/

3. Bichen Wu et al. "FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search." CVPR 2019.
   Link: https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_FBNet_Hardware-Aware_Efficient_ConvNet_Design_via_Differentiable_Neural_Architecture_Search_CVPR_2019_paper.html

4. Tianqi Chen et al. "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." OSDI 2018.
   Link: https://www.usenix.org/conference/osdi18/presentation/chen

5. Lianmin Zheng et al. "Ansor: Generating High-Performance Tensor Programs for Deep Learning." OSDI 2020.
   Link: https://www.usenix.org/conference/osdi20/presentation/zheng

6. Zhihao Jia et al. "TASO: Optimizing Deep Learning Computation with Automated Generation of Graph Substitutions." SOSP 2019.
   Link: https://catalyst.cs.cmu.edu/projects/taso.html

7. Yichen Yang et al. "Equality Saturation for Tensor Graph Superoptimization." MLSys 2021.
   Link: https://www.mwillsey.com/papers/tensat

8. Mingzhen Li et al. "The Deep Learning Compiler: A Comprehensive Survey." 2020.
   Link: https://dblp.org/rec/journals/corr/abs-2002-03794
