# Phase 6: MLIREmitter (JSON → Gawee MLIR)

## 개요

MLIREmitter는 JSON 그래프를 읽어 Gawee dialect MLIR을 생성하는 **프론트엔드**다.
Python(PyTorch)에서 추출한 신경망 구조 JSON이 입력이고, `gawee.conv`, `gawee.relu` 등의 MLIR IR이 출력이다.

```
graph.json  →  MLIREmitter  →  gawee.mlir
```

---

## 1. 전체 구조

```cpp
class MLIREmitter {
  MLIRContext *ctx;                                       // MLIR 컨텍스트
  std::unique_ptr<OpBuilder> builder;                     // op 생성 도구
  std::unordered_map<std::string, Value> valueMap;        // 이름 → SSA 값 매핑
  std::vector<std::pair<std::string, RankedTensorType>> weightArgs;  // weight 인자 목록
};
```

### 핵심 멤버의 역할

| 멤버 | 역할 |
|------|------|
| `builder` | MLIR op을 생성하는 도구 (삽입 위치 관리) |
| `valueMap` | JSON의 텐서 이름 → MLIR Value 매핑 (`"conv1"` → `%0`) |
| `weightArgs` | weight/bias 텐서 목록 (함수 인자로 추가) |

---

## 2. emit() 함수의 흐름

### 2-Pass 구조

```cpp
OwningOpRef<ModuleOp> MLIREmitter::emit(const llvm::json::Object &graph) {
  // === Pass 1: weight 수집 ===
  for (const auto &nodeVal : *nodes) {
    if (opType == "Conv" || opType == "MatMul") {
      // weight, bias 정보를 weightArgs에 추가
      weightArgs.push_back({weightName, weightType});
    }
  }

  // === 함수 시그니처 구성 ===
  // inputTypes = [graph inputs] + [weight/bias args]
  auto func = builder->create<func::FuncOp>(loc, "forward", funcType);

  // === 이름 → block argument 매핑 ===
  valueMap["input"] = entryBlock->getArgument(0);
  valueMap["conv1_weight"] = entryBlock->getArgument(1);

  // === Pass 2: 노드별 emit ===
  for (const auto &nodeVal : *nodes) {
    emitNode(*node, *values);  // 각 노드를 Gawee op으로 변환
  }

  // === return 생성 ===
  builder->create<func::ReturnOp>(loc, returnValues);
}
```

### 왜 2-Pass인가?

Weight와 bias는 **함수 인자**로 전달된다 (상수가 아님).
함수 시그니처를 만들려면 모든 weight의 shape을 먼저 알아야 한다.
그래서 첫 번째 pass에서 weight 정보를 수집하고, 두 번째 pass에서 op을 생성한다.

---

## 3. valueMap: SSA 값 추적

신경망 그래프에서 한 op의 출력이 다른 op의 입력이 된다.
`valueMap`이 이 연결을 관리한다:

```cpp
// Conv emit 후:
valueMap["conv1_output"] = convOp.getResult();

// 다음 Relu emit 시:
Value input = lookupValue("conv1_output");  // 이전 conv의 결과를 가져옴
auto reluOp = builder->create<ReluOp>(loc, resultType, input);
valueMap["relu1_output"] = reluOp.getResult();
```

### lookupValue 함수

```cpp
Value MLIREmitter::lookupValue(llvm::StringRef name) {
  auto it = valueMap.find(name.str());
  if (it != valueMap.end()) return it->second;
  return nullptr;  // 못 찾으면 null
}
```

---

## 4. emitNode: Op 디스패치

```cpp
bool MLIREmitter::emitNode(const llvm::json::Object &node,
                           const llvm::json::Object &values) {
  auto opType = node.getString("op_type");

  if (*opType == "Conv")      return emitConv(node, values);
  else if (*opType == "Relu") return emitRelu(node, values);
  else if (*opType == "Add")  return emitAdd(node, values);
  // ...
}
```

JSON의 `op_type` 필드에 따라 적절한 emit 함수를 호출한다.

---

## 5. emitConv 상세

```cpp
bool MLIREmitter::emitConv(const llvm::json::Object &node,
                           const llvm::json::Object &values) {
  // (1) input: JSON의 inputs 배열에서 이름을 읽고, valueMap에서 Value를 찾음
  auto inputName = (*inputs)[0].getAsString();
  Value input = lookupValue(*inputName);

  // (2) output type: JSON의 values에서 shape을 읽어 RankedTensorType으로 변환
  auto resultType = parseShape(outputInfo->getArray("shape"));

  // (3) weight/bias: 함수 인자로 추가된 것을 valueMap에서 찾음
  Value weight = lookupValue(nodeName->str() + "_weight");
  Value bias = lookupValue(nodeName->str() + "_bias");

  // (4) attributes: JSON에서 stride, padding, dilation 읽기
  auto strides = getArrayAttr("stride");

  // (5) Gawee op 생성
  auto convOp = builder->create<ConvOp>(
      loc, resultType, input, weight, bias,
      builder->getDenseI64ArrayAttr(strides),
      builder->getDenseI64ArrayAttr(padding),
      builder->getDenseI64ArrayAttr(dilation));

  // (6) 출력을 valueMap에 등록
  valueMap[outputName->str()] = convOp.getResult();
}
```

---

## 6. llvm::json API

JSON 파싱에 LLVM의 `llvm/Support/JSON.h`를 사용한다:

```cpp
// Object에서 필드 읽기
const auto *inputs = node.getArray("inputs");     // JSON 배열
auto opType = node.getString("op_type");          // JSON 문자열
auto intVal = attrs->getInteger("start_dim");     // JSON 정수
auto boolVal = attrs->getBoolean("ceil_mode");    // JSON 불리언

// 값이 없으면 nullptr / std::nullopt 반환
if (!inputs || inputs->empty()) { /* 에러 */ }

// 배열 순회
for (const auto &v : *arr) {
  if (auto i = v.getAsInteger()) result.push_back(*i);
}
```

### 주의: JSON 키 이름은 snake_case

JSON에서는 `kernel_size`, `ceil_mode`, `start_dim` (snake_case),
TableGen에서는 `$kernelSize`, `$ceilMode`, `$startDim` (camelCase).
이 불일치를 emit 함수에서 수동 매핑한다.

---

## 7. parseShape: 타입 생성

```cpp
RankedTensorType MLIREmitter::parseShape(const llvm::json::Array *shape) {
  SmallVector<int64_t> dims;
  for (const auto &dim : *shape) {
    dims.push_back(*dim.getAsInteger());
  }
  return RankedTensorType::get(dims, Float32Type::get(ctx));
}
```

JSON의 `"shape": [1, 64, 112, 112]`를 `tensor<1x64x112x112xf32>`로 변환한다.
모든 텐서가 `f32` 타입이라고 가정한다.

---

## 8. gawee-translate 도구

```cpp
int main(int argc, char **argv) {
  // (1) 커맨드라인 파싱 (입력 파일, 출력 파일)
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, ...);

  // (2) JSON 파일 읽기
  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFilename);
  auto jsonOrErr = json::parse(fileOrErr.get()->getBuffer());

  // (3) MLIR 컨텍스트 생성 + dialect 로드
  MLIRContext context;
  context.loadDialect<gawee::GaweeDialect>();
  // ... 기타 dialect

  // (4) emit
  gawee::MLIREmitter emitter(&context);
  auto module = emitter.emit(*graph);

  // (5) 출력
  module->print(output);
}
```

### gawee-opt과의 차이

| | gawee-translate | gawee-opt |
|---|---|---|
| 입력 | JSON 파일 | MLIR 텍스트 |
| 역할 | JSON → Gawee MLIR 생성 | MLIR에 pass 적용 |
| 핵심 | MLIREmitter 클래스 | MlirOptMain 함수 |

---

## 핵심 개념 정리

- **2-Pass 구조**: (1) weight 수집 → (2) op emit
- **valueMap**: JSON 텐서 이름 → MLIR SSA Value 매핑
- **OpBuilder**: op 생성 + 삽입 위치 관리
- **llvm::json API**: `getArray`, `getString`, `getInteger`, `getBoolean`
- **parseShape**: JSON shape 배열 → `RankedTensorType`
- **Weight는 함수 인자**: 상수가 아닌 함수 매개변수로 전달
- **snake_case ↔ camelCase**: JSON과 TableGen 간 이름 규칙 불일치에 주의
