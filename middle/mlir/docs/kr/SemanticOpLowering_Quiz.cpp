// Semantic Op Lowering Quiz
//
// 목표:
// 1. semantic op를 언제 gawee에 남겨야 하는지 설명할 수 있는가?
// 2. Gather / Range / Split lowering 핵심을 직접 다시 쓸 수 있는가?
//
// 규칙:
// - TODO를 채운다.
// - 답은 "컴파일되는 정답"보다 "정확한 개념"이 더 중요하다.

#include <cstdint>
#include <vector>

namespace quiz {

// Q1. 아래 op 중 semantic op로 gawee에 남길 가치가 큰 것을 고르시오.
//
// 후보:
//   1. Gather
//   2. Neg
//   3. Range
//   4. Sin
//
// TODO:
// - semantic op 번호만 남기고 trivial op 번호는 지워라.
std::vector<int> semanticOps() {
  return {
      /* TODO */, /* TODO */
  };
}

// Q2. Gather의 axis가 1일 때,
// output index [b, i0, i1, c]를 data index로 바꾸는 개념을 적어라.
//
// 규칙:
// - axis 이전 dim은 그대로 사용
// - axis 위치는 indices에서 읽은 값 사용
// - axis 이후 dim은 indices rank만큼 밀린다
struct GatherIndexRule {
  bool keepPrefixDims = false;
  bool useIndicesValueAtAxis = false;
  bool shiftSuffixDims = false;
};

GatherIndexRule gatherRule() {
  GatherIndexRule rule;
  rule.keepPrefixDims = /* TODO */;
  rule.useIndicesValueAtAxis = /* TODO */;
  rule.shiftSuffixDims = /* TODO */;
  return rule;
}

// Q3. Range의 i번째 원소 식을 완성하라.
//
// ONNX Range:
//   output[i] = start + i * delta
int64_t rangeElement(int64_t start, int64_t delta, int64_t i) {
  return /* TODO */;
}

// Q4. positive constant delta를 가정할 때,
// dynamic range 길이를 계산하는 식을 완성하라.
//
// ceil((limit - start) / delta)
int64_t rangeLength(int64_t start, int64_t limit, int64_t delta) {
  int64_t diff = limit - start;
  int64_t adjusted = /* TODO */;
  return /* TODO */;
}

// Q5. Split lowering에서 axis 방향 offset은 어떻게 갱신되는가?
//
// 예:
// splitSizes = [3, 5, 2]
// offset sequence = 0 -> 3 -> 8 -> 10
int64_t nextSplitOffset(int64_t currentOffset, int64_t splitSize) {
  return /* TODO */;
}

// Q6. 빈칸 채우기
//
// "Emitter는 가능한 한 ONNX ______ 를 보존하고,
//  실제 decomposition은 ______ 에서 수행한다."
struct Philosophy {
  const char *preserve = nullptr;
  const char *decomposeAt = nullptr;
};

Philosophy fillPhilosophy() {
  return {
      /* TODO */,
      /* TODO */
  };
}

} // namespace quiz
