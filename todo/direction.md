# TODO: Career Direction for Gawee

이 문서는 `Gawee`를 어떤 포트폴리오 축으로 밀어야
`AI 추론 최적화 엔지니어`로 가장 경쟁력 있게 보일지를 정리한 것이다.

핵심 전제는 단순하다.

- 모든 JD를 완벽히 커버하는 것은 비효율적이다.
- 대신 여러 JD가 공통으로 강하게 요구하는 축을 중심으로 포트폴리오를 설계해야 한다.
- 현재 우선순위는 `1) ONNX rewrite full-stack 경험 > 2) compiler/op/lowering 역량 > 3) quantization 경험`이다.

---

## 1. 최종 포지셔닝

내가 만들고 싶은 포지셔닝은 아래다.

> PyTorch/ONNX 모델을 실제 디바이스 배포 관점에서 분석하고,
> unsupported op를 graph rewrite와 decomposition으로 해결하며,
> correctness / latency / memory를 직접 검증할 수 있는
> `AI inference optimization engineer`

여기서 중요한 것은 "조금씩 다 안다"가 아니라 아래를 끝까지 닫는 경험이다.

- 모델 선정
- Torch -> ONNX 변환
- unsupported op audit
- ONNX rewrite / graph surgery
- correctness validation
- runtime latency 비교
- 실패 원인 분석
- 재현 가능한 script / report 정리

즉 주력 서사는 `full-stack inference optimization loop를 실제로 실행해 본 사람`이어야 한다.

---

## 2. 우선순위

### Priority 1. ONNX Rewrite And Full-Stack Execution Ownership

가장 강하게 밀어야 할 축은 아래다.

- `torch -> onnx` 변환 이슈를 실제로 다뤘다.
- dynamic/static shape 문제를 다뤘다.
- unsupported op를 찾아 primitive subset이나 backend-friendly form으로 내렸다.
- rewrite 전후 correctness를 숫자로 비교했다.
- rewrite 전후 runtime latency를 실제로 비교했다.
- 특정 디바이스 / runtime 제약을 기준으로 실패 원인을 설명할 수 있다.

이 축이 가장 중요한 이유는 JD 다수가 여기서 직접 겹치기 때문이다.

직접 근거:

- [../jd/nota.md](../jd/nota.md)
  - `Torch ONNX 변환`, `dynamic/static 정적화`, `unsupported ops 수정/대체`,
    `Front/Middle-End Graph Rewriting`, `변환 전후 품질 비교 및 latency/메모리 최적화`를 명시한다.
- [../jd/mobilint1.md](../jd/mobilint1.md)
  - `모델 파싱`, `IR 변환`, `그래프 레벨 최적화`, `Custom Op 정의 및 Decomposition`을 전면에 둔다.
- [../jd/mob2.md](../jd/mob2.md)
  - `딥러닝 네트워크 모델을 NPU 언어로 변환`, `모델 최적화 엔진 개발`을 요구한다.
- [../jd/deepx.md](../jd/deepx.md)
  - `intermediate graph-level representation`, `DNN model conversion`, `optimization`, `resource constraint`를 요구한다.
- [../jd/boss.md](../jd/boss.md)
  - `모델 포팅 및 최적화`, `커스텀 연산 구현`, `컴파일러 및 런타임 최적화`를 함께 본다.
- [../jd/zetic.md](../jd/zetic.md)
  - `온디바이스 벤치마킹`, `정확도 체크`, `회귀 테스트`, `컴파일러/런타임 제약 해결`을 명시한다.

결론적으로, `ONNX rewrite + 실제 검증`은 가장 넓은 공통분모다.

### Priority 2. Compiler-Level Operation Definition And Lowering

두 번째 축은 compiler-level 역량이다.

- 연산의 semantic contract를 이해한다.
- custom op를 primitive op sequence로 decomposition할 수 있다.
- legality 조건과 invariant를 설명할 수 있다.
- IR / graph / lowering / backend-friendly representation 사이의 관계를 이해한다.
- 가능하면 특정 op를 직접 정의하고, pass와 verifier까지 연결할 수 있다.

이 축이 중요한 이유는 주력 축을 단순 tooling 경험이 아니라
`compiler-aware optimization 능력`으로 끌어올려 주기 때문이다.

직접 근거:

- [../jd/mobilint1.md](../jd/mobilint1.md)
  - `Custom Op 정의 및 분해`, `그래프 최적화`, `shape/layout/stride 이해`를 우대한다.
- [../jd/deepx.md](../jd/deepx.md)
  - `compiler optimization`, `tiling`, `memory optimization`, `operation fusion`을 요구한다.
- [../jd/boss.md](../jd/boss.md)
  - `커스텀 오퍼레이터 및 커널 개발`, `TVM/MLIR/frontend`, `컴파일러 개발 및 최적화`를 요구한다.
- [../jd/opt.md](../jd/opt.md)
  - `추론 엔진`, `자원 관리`, `C++ 구현`, `모델 변환 최적화`를 함께 본다.
- [../jd/typecast.md](../jd/typecast.md)
  - `TensorRT / ONNX`, `컴파일러 및 IR 이해`, `TorchDynamo`, `Triton`, `CUTLASS`를 우대한다.

즉 2번 축은 "나는 모델 변환만 하는 사람이 아니라,
필요하면 연산 의미와 lowering까지 내려가 문제를 푼다"를 보여주는 역할이다.

### Priority 3. Quantization As A Supporting Axis

세 번째 축은 quantization이다.

- PTQ / dynamic quantization / QAT 차이를 이해한다.
- latency / memory / drift tradeoff를 수치로 설명할 수 있다.
- calibration 정책과 failure case를 재현 가능하게 정리할 수 있다.
- quantization이 실제 backend / runtime 제약과 어떻게 연결되는지 설명할 수 있다.

quantization을 3순위로 두는 이유는,
여러 JD에서 중요하게 언급되지만 대체로 `주력 단독 축`보다는
`전체 추론 최적화 능력을 보강하는 축`으로 작동하기 때문이다.

직접 근거:

- [../jd/openedge.md](../jd/openedge.md)
  - `PTQ/QAT/Pruning`, `NPU HW를 고려한 신경망 최적화`를 핵심 업무로 둔다.
- [../jd/mob3.md](../jd/mob3.md)
  - `Quantization, Pruning` 등 경량화를 직접 역할로 둔다.
- [../jd/nota.md](../jd/nota.md)
  - `quantization-friendly 구조`, `메모리/latency 최적화`를 요구한다.
- [../jd/zetic.md](../jd/zetic.md)
  - `양자화`, `혼합 정밀도`, `정확도/배터리/메모리 tradeoff`를 강조한다.
- [../jd/vurun.md](../jd/vurun.md)
  - `TensorRT`, `Quantization`, `Pruning` 기반의 엣지 최적화를 요구한다.

즉 quantization은 반드시 있어야 하지만,
현재 포트폴리오의 정체성을 규정하는 1번 축을 대체하진 않는다.

---

## 3. 이 방향이 맞는 이유

JD 전체를 다 읽고 나면 공통분모는 생각보다 선명하다.

### A. "모델을 가져와 실제로 돌려본 경험"이 중요하다

많은 JD는 연구 아이디어 자체보다 다음을 본다.

- 특정 모델이 왜 안 도는가
- 어떤 op가 문제인가
- 어떻게 rewrite / decomposition 할 것인가
- 정확도는 유지되는가
- latency / memory는 실제로 좋아졌는가

이 요구는 특히
[../jd/nota.md](../jd/nota.md),
[../jd/zetic.md](../jd/zetic.md),
[../jd/vurun.md](../jd/vurun.md),
[../jd/typecast.md](../jd/typecast.md),
[../jd/opt.md](../jd/opt.md)에서 강하다.

즉 "현업 경험처럼 보이는 포트폴리오"는
논문풍 추상화보다 `실행-검증-분석`이 닫혀 있어야 한다.

### B. 단순 model optimization보다 graph/compiler 관점이 더 강한 차별점이다

`ONNX Runtime 좀 써봤다` 수준은 차별화가 약하다.
반면 아래가 있으면 강해진다.

- unsupported op audit 자동화
- decomposition pass 구현
- verifier / invariant 정의
- dynamic shape export 문제 분석
- primitive subset 기준 lowering

이건 특히
[../jd/mobilint1.md](../jd/mobilint1.md),
[../jd/deepx.md](../jd/deepx.md),
[../jd/boss.md](../jd/boss.md),
[../jd/mob2.md](../jd/mob2.md)와 잘 맞는다.

### C. quantization은 필요하지만 단독 정체성으로는 범용성이 덜하다

quantization 중심 JD도 존재하지만, 전체적으로 보면 더 넓은 공통분모는 아니다.

- [../jd/openedge.md](../jd/openedge.md)
- [../jd/mob3.md](../jd/mob3.md)

반면 다수 JD는 quantization을 `있으면 좋은 핵심 보강축`으로 다룬다.
따라서 지금 단계에서는 quantization을 주력으로 삼기보다
`rewrite 이후의 추가 최적화 축`으로 두는 편이 효율적이다.

### D. system/kernel-only 방향으로 너무 가면 JD 커버리지가 오히려 줄어든다

물론 아래 JD들은 kernel/system 성향이 강하다.

- [../jd/moreh/npu.md](../jd/moreh/npu.md)
- [../jd/moreh/gpu.md](../jd/moreh/gpu.md)
- [../jd/moreh/sys.md](../jd/moreh/sys.md)
- [../jd/mangoboost.md](../jd/mangoboost.md)

하지만 이 포지션들은 커널, 통신, 클러스터, 시스템 소프트웨어 비중이 높다.
현재 `Gawee`로 가장 강하게 만들 수 있는 서사는 여기가 아니라
`model conversion + graph rewrite + runtime validation`이다.

즉 시스템/커널은 장기 확장 축이지,
현재 포트폴리오의 중심으로 두기엔 효율이 떨어진다.

---

## 4. 내가 피해야 할 포지셔닝

아래 방향은 피하는 것이 좋다.

### A. "모든 걸 다 하는 범용 AI 엔지니어"

이건 메시지가 약해진다.
JD는 결국 특정 문제를 실제로 해결할 수 있는 사람을 원한다.

### B. "compiler toy project를 만든 사람"

IR만 있고 실제 runtime validation이 없으면
현업형 inference optimization 서사로 보기 어렵다.

### C. "quantization만 하는 사람"

quantization 단독 서사는 일부 JD에는 맞아도
전체 커버리지는 떨어진다.

### D. "kernel/system specialist"

이는 Moreh, MangoBoost류에는 맞을 수 있지만
현재 목표인 `wide but coherent coverage`에는 맞지 않는다.

---

## 5. Gawee에서 바로 만들어야 할 증거

방향성을 말로만 두지 않으려면,
`Gawee`에서 아래 산출물이 실제로 나와야 한다.

### A. ONNX rewrite evidence

- [ ] 공개 모델 5개 + vision 2개 baseline 확보
- [ ] dynamic shape 모델 2개 export 성공
- [ ] unsupported op before/after 표 생성
- [ ] rewrite 전후 correctness report 생성
- [ ] rewrite 전후 latency report 생성
- [ ] 실패 사례와 우회 전략 문서화

### B. compiler-level evidence

- [ ] primitive op subset 명시
- [ ] `LayerNorm`, `GELU`, `mask`, `Gather` decomposition 구현
- [ ] legality 조건과 skip reason 문서화
- [ ] verifier / invariant harness 추가
- [ ] middle-end correctness + speedup harness 추가

### C. quantization evidence

- [ ] vision 2개 + NLP 1개 이상 PTQ 결과 확보
- [ ] FP32 vs quantized latency / size / drift 표 생성
- [ ] calibration policy와 failure case 문서화

---

## 6. 한 줄 결론

현재 가장 경쟁력 있는 방향은 아래다.

> `ONNX rewrite를 중심으로 full-stack inference optimization 경험을 만들고,
> 그 아래를 compiler-level lowering 능력으로 받치며,
> quantization을 보조 축으로 붙이는 것`

이 방향은 JD 전체를 기준으로 볼 때
가장 넓은 공통분모를 잡으면서도,
단순 범용 포지셔닝보다 훨씬 선명하다.
