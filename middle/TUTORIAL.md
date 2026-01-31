# Gawee Middle Layer 튜토리얼

이 문서는 C++ 초보자를 위한 Gawee 파서 코드 설명서입니다.

## 목차

1. [빌드 시스템 (CMake/Make)](#1-빌드-시스템-cmakemake)
2. [프로젝트 구조](#2-프로젝트-구조)
3. [C++ 기초 개념](#3-c-기초-개념)
4. [Graph.h 분석](#4-graphh-분석)
5. [Parser.h 분석](#5-parserh-분석)
6. [Parser.cpp 분석](#6-parsercpp-분석)
7. [main.cpp 분석](#7-maincpp-분석)

---

## 1. 빌드 시스템 (CMake/Make)

### CMakeLists.txt 분석

```cmake
# 최소 CMake 버전 요구
cmake_minimum_required(VERSION 3.16)

# 프로젝트 이름과 버전, 사용 언어 정의
project(GaweeMiddle VERSION 0.1.0 LANGUAGES CXX)

# C++17 표준 사용 (std::filesystem 등 최신 기능)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 헤더 파일 검색 경로
# #include "gawee/Graph.h" → include/gawee/Graph.h 에서 찾음
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# 컴파일할 소스 파일 목록
set(SOURCES
    src/Graph.cpp
    src/Parser.cpp
    src/main.cpp
)

# 실행파일 생성: gawee_parser라는 이름으로
add_executable(gawee_parser ${SOURCES})
```

### Make란?

Make는 Makefile을 읽어서 실제 컴파일을 수행합니다.
CMake가 생성한 Makefile에는 컴파일 명령어들이 들어있습니다.

### 빌드 과정

```bash
# 1. build 디렉토리 생성 (소스와 빌드 파일 분리)
mkdir build && cd build

# 2. CMake 실행: Makefile 생성
cmake ..
# ".."는 상위 디렉토리(CMakeLists.txt 위치)를 의미

# 3. Make 실행: 컴파일
make
# 결과물: ./gawee_parser 실행파일

# 4. 실행
./gawee_parser ../../jsondata/graph.json
```

### 유용한 명령어

```bash
make clean      # 빌드 결과물 삭제
make -j4        # 4개 코어로 병렬 컴파일 (빠름)
cmake --build . # make 대신 사용 가능 (플랫폼 독립적)
```

---

## 2. 프로젝트 구조

```
middle/
├── CMakeLists.txt          # 빌드 설정
├── include/                # 헤더 파일 (.h)
│   └── gawee/
│       ├── Graph.h         # 데이터 구조 선언
│       └── Parser.h        # 파서 인터페이스 선언
├── src/                    # 소스 파일 (.cpp)
│   ├── Graph.cpp           # Graph 메서드 구현
│   ├── Parser.cpp          # 파서 구현
│   └── main.cpp            # 프로그램 진입점
└── third_party/
    └── json.hpp            # 외부 라이브러리
```

### 헤더(.h)와 소스(.cpp) 분리 이유

C++에서는 **선언(declaration)**과 **구현(definition)**을 분리합니다.

- **헤더 파일 (.h)**: "이런 것들이 있다"고 선언
- **소스 파일 (.cpp)**: 실제 코드 구현

```cpp
// Graph.h - 선언
class Graph {
    void dump() const;  // "dump라는 함수가 있다"
};

// Graph.cpp - 구현
void Graph::dump() const {
    // 실제 코드
    std::cout << "=== Graph ===" << std::endl;
}
```

이렇게 분리하면:
1. 컴파일 시간 단축 (변경된 .cpp만 재컴파일)
2. 인터페이스와 구현 분리 (사용자는 .h만 보면 됨)

---

## 3. C++ 기초 개념

### 3.1 #include와 헤더 가드

```cpp
// Parser.h

#ifndef GAWEE_PARSER_H    // 만약 GAWEE_PARSER_H가 정의 안 되어있으면
#define GAWEE_PARSER_H    // GAWEE_PARSER_H를 정의하고

// ... 코드 ...

#endif // GAWEE_PARSER_H  // 여기서 끝
```

이것을 **헤더 가드(Header Guard)**라고 합니다.
같은 헤더가 여러 번 include 되는 것을 방지합니다.

```cpp
#include "gawee/Graph.h"  // 내 프로젝트 헤더 (따옴표)
#include <iostream>       // 표준 라이브러리 (꺾쇠)
```

### 3.2 namespace (네임스페이스)

이름 충돌을 방지하기 위한 "이름 공간"입니다.

```cpp
namespace gawee {
    class Graph { ... };  // gawee::Graph
}

namespace other {
    class Graph { ... };  // other::Graph (다른 클래스)
}
```

Python의 모듈과 비슷합니다: `gawee.Graph` vs `other.Graph`

### 3.3 클래스와 구조체

```cpp
// struct: 기본이 public (데이터 묶음에 적합)
struct Value {
    std::string id;
    std::vector<int64_t> shape;
};

// class: 기본이 private (메서드가 많을 때 적합)
class Parser {
public:                   // 외부에서 접근 가능
    static std::unique_ptr<Graph> load(const std::string& path);

private:                  // 클래스 내부에서만 접근
    static bool parseValue(const void* json, Value& value);
};
```

### 3.4 포인터와 참조

C++에서 가장 중요하고 어려운 개념입니다.

```cpp
int x = 10;

// 값 복사
int y = x;        // y는 x의 복사본 (별개의 변수)

// 참조 (Reference) - 별명
int& ref = x;     // ref는 x의 별명 (같은 메모리)
ref = 20;         // x도 20이 됨

// 포인터 (Pointer) - 주소
int* ptr = &x;    // ptr은 x의 메모리 주소를 저장
*ptr = 30;        // x가 30이 됨 (*는 주소의 값에 접근)
```

함수 파라미터에서:

```cpp
// 값 전달: 복사됨 (비효율적)
void func1(std::string s) { ... }

// 참조 전달: 복사 없이 원본 접근
void func2(std::string& s) { s = "modified"; }  // 원본 수정

// const 참조: 복사 없이 읽기만 (가장 흔함)
void func3(const std::string& s) { ... }  // 원본 수정 불가
```

### 3.5 스마트 포인터

C++에서는 `new`로 할당한 메모리를 `delete`로 해제해야 합니다.
실수로 해제를 안 하면 **메모리 누수(memory leak)**가 발생합니다.

```cpp
// 위험한 방식 (옛날 C++ 스타일)
Graph* g = new Graph();
// ... 사용 ...
delete g;  // 깜빡하면 메모리 누수!

// 안전한 방식 (현대 C++)
std::unique_ptr<Graph> g = std::make_unique<Graph>();
// 스코프를 벗어나면 자동으로 해제됨
```

`unique_ptr`의 특징:
- 자동으로 메모리 해제
- 복사 불가 (소유권 이동만 가능)
- `nullptr`과 비교 가능

```cpp
std::unique_ptr<Graph> graph = Parser::load("graph.json");
if (graph) {  // nullptr이 아니면
    graph->dump();
}
// 함수 끝나면 자동 해제
```

### 3.6 std::vector

Python의 list와 비슷한 동적 배열입니다.

```cpp
std::vector<int> nums;           // 빈 벡터
nums.push_back(1);               // [1]
nums.push_back(2);               // [1, 2]
nums.size();                     // 2
nums[0];                         // 1 (인덱스 접근)

// 초기화
std::vector<int> v = {1, 2, 3};

// 순회
for (const auto& n : v) {        // 범위 기반 for
    std::cout << n << std::endl;
}
```

### 3.7 std::unordered_map

Python의 dict와 비슷한 해시맵입니다.

```cpp
std::unordered_map<std::string, Value> values;

// 삽입
values["x"] = someValue;

// 검색
if (values.count("x") > 0) {     // 키가 있는지 확인
    Value& v = values["x"];
}

// 순회
for (const auto& [key, value] : values) {
    std::cout << key << std::endl;
}
```

### 3.8 auto 키워드

타입을 자동 추론합니다.

```cpp
// 명시적 타입
std::unordered_map<std::string, int>::iterator it = map.begin();

// auto 사용 (간결함)
auto it = map.begin();

// 복잡한 타입에 유용
auto graph = std::make_unique<Graph>();  // std::unique_ptr<Graph>
```

### 3.9 const 키워드

"변경 불가"를 의미합니다.

```cpp
const int x = 10;           // x 값 변경 불가
x = 20;                     // 컴파일 에러!

// 함수 파라미터
void print(const std::string& s);  // s를 읽기만 함

// 멤버 함수
void Graph::dump() const;   // 이 함수는 멤버 변수를 수정하지 않음
```

### 3.10 static 키워드

클래스에서 `static`은 "인스턴스 없이 호출 가능"을 의미합니다.

```cpp
class Parser {
public:
    // static 메서드: Parser 객체 없이 호출 가능
    static std::unique_ptr<Graph> load(const std::string& path);
};

// 사용
auto graph = Parser::load("graph.json");  // Parser 인스턴스 불필요
```

Python의 `@staticmethod`와 같습니다.

---

## 4. Graph.h 분석

```cpp
#ifndef GAWEE_GRAPH_H
#define GAWEE_GRAPH_H

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace gawee {
```

### Value 구조체

JSON의 "values" 섹션에 해당합니다.

```cpp
/**
 * 텐서 메타데이터 (shape, dtype)
 *
 * JSON 예시:
 * {"id": "x", "shape": [1, 3, 224, 224], "dtype": "float32"}
 */
struct Value {
    std::string id;                    // 텐서 이름
    std::vector<int64_t> shape;        // 텐서 shape
    std::string dtype;                 // 데이터 타입
    std::optional<std::string> path;   // 상수일 경우 파일 경로
};
```

`std::optional<T>`: 값이 있을 수도 없을 수도 있음 (Python의 `Optional[str]`)

### WeightRef 구조체

노드 속성 중 weight/bias 참조입니다.

```cpp
/**
 * 바이너리 weight 파일 참조
 */
struct WeightRef {
    std::vector<int64_t> shape;
    std::string dtype;
    std::string path;                  // "weights/conv1_weight_0.bin"
};
```

### Node 구조체

연산 노드입니다.

```cpp
struct Node {
    std::string opType;                        // "Conv", "Relu" 등
    std::string name;                          // "layer1.0.conv1"
    std::vector<std::string> inputs;           // 입력 텐서 ID들
    std::vector<std::string> outputs;          // 출력 텐서 ID들

    // 속성들 (타입별로 분리)
    std::unordered_map<std::string, int64_t> intAttrs;           // groups: 1
    std::unordered_map<std::string, double> floatAttrs;          // eps: 1e-5
    std::unordered_map<std::string, std::string> stringAttrs;    // padding: "zeros"
    std::unordered_map<std::string, std::vector<int64_t>> intArrayAttrs;  // kernel_size: [3, 3]
    std::unordered_map<std::string, WeightRef> weightAttrs;      // weight: {...}
```

헬퍼 메서드:

```cpp
    // 속성 접근 헬퍼
    int64_t getInt(const std::string& key, int64_t defaultVal = 0) const {
        auto it = intAttrs.find(key);          // 키 검색
        return it != intAttrs.end() ? it->second : defaultVal;
        // 찾았으면 값 반환, 아니면 기본값
    }
```

`it->second`: map의 iterator는 `{key, value}` 쌍이고, `second`가 value입니다.

### Graph 구조체

전체 그래프입니다.

```cpp
struct Graph {
    std::vector<std::string> inputs;               // 그래프 입력
    std::vector<std::string> outputs;              // 그래프 출력
    std::unordered_map<std::string, Value> values; // 모든 텐서
    std::vector<Node> nodes;                       // 모든 연산
    std::string baseDir;                           // JSON 파일 디렉토리

    // 메서드
    const Value* getValue(const std::string& id) const {
        auto it = values.find(id);
        return it != values.end() ? &it->second : nullptr;
        // 포인터 반환: 없으면 nullptr
    }

    void dump() const;  // 그래프 출력 (Graph.cpp에서 구현)
};
```

---

## 5. Parser.h 분석

```cpp
class Parser {
public:
    /**
     * JSON 파일에서 그래프 로드
     *
     * @param jsonPath  graph.json 경로
     * @return          성공시 Graph, 실패시 nullptr
     */
    static std::unique_ptr<Graph> load(const std::string& jsonPath);

private:
    // 내부 파싱 메서드 (외부에서 호출 불가)
    static bool parseValue(const void* json, Value& value);
    static bool parseNode(const void* json, Node& node);
    static bool parseWeightRef(const void* json, WeightRef& weight);
};
```

`const void*`: JSON 라이브러리의 타입을 헤더에 노출하지 않기 위한 기법입니다.
(실제로는 `nlohmann::json*`이지만, json.hpp를 include하지 않아도 됨)

### WeightLoader 클래스

```cpp
class WeightLoader {
public:
    /**
     * 바이너리 파일에서 데이터 로드
     *
     * @tparam T    데이터 타입 (float, int 등)
     * @param path  .bin 파일 경로
     * @return      로드된 데이터, 실패시 빈 벡터
     */
    template<typename T>
    static std::vector<T> load(const std::string& path);
};
```

`template<typename T>`: 제네릭/템플릿입니다.
Python의 `Generic[T]`와 비슷합니다.

```cpp
// 사용
auto floats = WeightLoader::load<float>("weight.bin");
auto ints = WeightLoader::load<int>("indices.bin");
```

---

## 6. Parser.cpp 분석

### JSON 라이브러리 사용

```cpp
#include "../third_party/json.hpp"

using json = nlohmann::json;  // 타입 별명 (타이핑 줄이기)
```

nlohmann/json은 헤더 하나로 된 JSON 라이브러리입니다.
Python의 `json` 모듈과 사용법이 비슷합니다.

### parseValue 함수

```cpp
bool Parser::parseValue(const void* jsonPtr, Value& value) {
    // void*를 json*로 캐스팅
    const json& j = *static_cast<const json*>(jsonPtr);

    try {
        // 필수 필드 (없으면 예외 발생)
        value.id = j.at("id").get<std::string>();
        value.dtype = j.at("dtype").get<std::string>();

        // shape 파싱 (null일 수 있음)
        if (j.contains("shape") && !j["shape"].is_null()) {
            for (const auto& dim : j["shape"]) {
                value.shape.push_back(dim.get<int64_t>());
            }
        }

        // 선택적 필드
        if (j.contains("path")) {
            value.path = j["path"].get<std::string>();
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Parser] Error: " << e.what() << std::endl;
        return false;
    }
}
```

Python 버전과 비교:
```python
def parse_value(j: dict) -> Value:
    return Value(
        id=j["id"],
        dtype=j["dtype"],
        shape=j.get("shape", []),
        path=j.get("path")
    )
```

### parseNode 함수 (속성 파싱)

```cpp
// 속성 타입에 따라 분기
if (val.is_number_integer()) {
    node.intAttrs[key] = val.get<int64_t>();
}
else if (val.is_number_float()) {
    node.floatAttrs[key] = val.get<double>();
}
else if (val.is_boolean()) {
    node.intAttrs[key] = val.get<bool>() ? 1 : 0;  // bool → int
}
else if (val.is_string()) {
    node.stringAttrs[key] = val.get<std::string>();
}
else if (val.is_array()) {
    std::vector<int64_t> arr;
    for (const auto& elem : val) {
        if (elem.is_number_integer()) {
            arr.push_back(elem.get<int64_t>());
        }
    }
    if (!arr.empty()) {
        node.intArrayAttrs[key] = arr;
    }
}
else if (val.is_object() && val.contains("path")) {
    // WeightRef 파싱
    WeightRef weight;
    if (parseWeightRef(&val, weight)) {
        node.weightAttrs[key] = weight;
    }
}
```

### load 함수 (메인 진입점)

```cpp
std::unique_ptr<Graph> Parser::load(const std::string& jsonPath) {
    // 파일 열기
    std::ifstream file(jsonPath);
    if (!file.is_open()) {
        std::cerr << "Cannot open: " << jsonPath << std::endl;
        return nullptr;  // 실패
    }

    // JSON 파싱
    json j;
    try {
        file >> j;  // 파일에서 JSON 읽기
    } catch (const std::exception& e) {
        std::cerr << "Parse error: " << e.what() << std::endl;
        return nullptr;
    }

    // Graph 생성
    auto graph = std::make_unique<Graph>();

    // 기본 디렉토리 저장 (weight 파일 경로용)
    std::filesystem::path p(jsonPath);
    graph->baseDir = p.parent_path().string();

    // 각 섹션 파싱
    // ... inputs, outputs, values, nodes ...

    return graph;  // 소유권 이동 (move)
}
```

### WeightLoader 템플릿 특수화

```cpp
template<>
std::vector<float> WeightLoader::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);  // 바이너리 모드
    if (!file.is_open()) {
        return {};  // 빈 벡터 반환
    }

    // 파일 크기 구하기
    file.seekg(0, std::ios::end);      // 파일 끝으로 이동
    size_t fileSize = file.tellg();    // 현재 위치 = 파일 크기
    file.seekg(0, std::ios::beg);      // 다시 처음으로

    // 원소 개수 계산
    size_t numElements = fileSize / sizeof(float);  // 4바이트씩

    // 데이터 읽기
    std::vector<float> data(numElements);
    file.read(reinterpret_cast<char*>(data.data()), fileSize);

    return data;
}
```

`reinterpret_cast<char*>`: float 배열을 바이트 배열로 해석합니다.
바이너리 I/O에서 자주 사용됩니다.

---

## 7. main.cpp 분석

```cpp
int main(int argc, char* argv[]) {
    // argc: 인자 개수
    // argv: 인자 배열 (argv[0]은 프로그램 이름)

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph.json>" << std::endl;
        return 1;  // 에러 코드
    }

    std::string jsonPath = argv[1];  // 첫 번째 인자

    // 그래프 로드
    auto graph = gawee::Parser::load(jsonPath);

    if (!graph) {  // nullptr 체크
        std::cerr << "Failed to load graph!" << std::endl;
        return 1;
    }

    // 그래프 출력
    graph->dump();

    // Weight 로드 예시
    for (const auto& node : graph->nodes) {
        if (node.opType == "Conv") {
            if (auto* weight = node.getWeight("weight")) {
                std::string weightPath = graph->baseDir + "/" + weight->path;
                auto data = gawee::WeightLoader::load<float>(weightPath);

                if (!data.empty()) {
                    std::cout << "Loaded " << data.size() << " floats" << std::endl;
                }
            }
            break;  // 첫 번째 Conv만
        }
    }

    return 0;  // 성공
}
```

---

## 부록: Python vs C++ 비교표

| 개념 | Python | C++ |
|------|--------|-----|
| 리스트 | `list` | `std::vector` |
| 딕셔너리 | `dict` | `std::unordered_map` |
| 문자열 | `str` | `std::string` |
| None | `None` | `nullptr` |
| Optional | `Optional[T]` | `std::optional<T>` |
| 타입 힌트 | `def f(x: int)` | `void f(int x)` |
| 클래스 | `class Foo:` | `class Foo { };` |
| 상속 | `class B(A):` | `class B : public A { };` |
| 반복문 | `for x in items:` | `for (auto& x : items) { }` |
| 파일 읽기 | `open(path)` | `std::ifstream file(path)` |
| 예외 | `try: except:` | `try { } catch { }` |
| 출력 | `print(x)` | `std::cout << x << std::endl;` |

---

## 다음 단계: MLIR 연동

이 파서로 로드한 Graph를 MLIR로 변환하려면:

1. MLIR 빌드 (LLVM 프로젝트의 일부)
2. Dialect 정의 (Gawee 연산들의 MLIR 표현)
3. Graph → MLIR 변환 코드 작성

```cpp
// 예시 (미래 작업)
mlir::ModuleOp convertToMLIR(const gawee::Graph& graph) {
    mlir::OpBuilder builder(context);

    for (const auto& node : graph.nodes) {
        if (node.opType == "Conv") {
            // gawee.conv 연산 생성
        }
    }

    return module;
}
```
