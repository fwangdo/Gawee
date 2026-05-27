//===----------------------------------------------------------------------===//
// Phase 10: Conv Consumer Chain Fusion Quiz
//===----------------------------------------------------------------------===//
//
// 아래 코드의 빈칸(_____)을 채워서 완성하세요.
// 각 빈칸에는 MLIR API 호출, 조건식, 또는 타입이 들어갑니다.
//
//===----------------------------------------------------------------------===//

// ============================================================
// Quiz 1: findElementwiseConsumerChain — consumer chain 탐색
//
// conv op에서 시작해서 single-use elementwise consumer chain을 수집합니다.
// chain 탐색이 멈추는 조건 3가지를 이해하세요.
// ============================================================

static SmallVector<Operation *> findElementwiseConsumerChain(Operation *root) {
  SmallVector<Operation *> chain;
  Operation *current = root;

  // Q1a: loop 조건 — current op의 result가 1개이고, 그 result의 user가 1개인 동안
  while (current->_____() == 1 && current->getResult(0)._____()) {
    Operation *user = *current->getResult(0).getUsers().begin();

    // Q1b: user가 elementwise generic인지 확인
    auto genericOp = dyn_cast<_____>(user);
    if (!genericOp || !isElementwiseGenericHeuristic(genericOp))
      break;

    chain.push_back(user);
    current = user;
  }
  return chain;
}


// ============================================================
// Quiz 2: Tiling root 선택
//
// consumer chain이 있으면 마지막 consumer를 root로,
// 없으면 conv 자체를 root로 사용합니다.
// 왜 마지막 consumer를 root로 쓰는지 생각하세요.
// ============================================================

// conv에서 consumer chain 탐색
SmallVector<Operation *> consumerChain =
    findElementwiseConsumerChain(plan.operation);

// Q2a: tiling root 결정 — chain이 비어있으면 conv, 아니면 chain의 _____
Operation *tilingRoot =
    consumerChain.empty() ? plan.operation : consumerChain._____();

// Q2b: tiling root를 TilingInterface로 cast
auto tilingIface = dyn_cast<_____>(tilingRoot);


// ============================================================
// Quiz 3: Tile sizes 결정
//
// chain이 있을 때와 없을 때 tile sizes가 다른 이유를 이해하세요.
// conv: 7 loops (N, C_out, H, W, C_in, KH, KW)
// elementwise generic: 4 loops (N, C, H, W)
// ============================================================

SmallVector<int64_t> fullTileSizes;
if (consumerChain.empty()) {
  // Q3a: chain이 없으면 conv의 전체 loop 수만큼 tile sizes를 만들고
  // parallel dims만 plan에서 복사, 나머지는 0 (no tiling)
  unsigned numLoops = cast<linalg::LinalgOp>(plan.operation)._____();
  fullTileSizes.assign(numLoops, _____);
  for (unsigned i = 0;
       i < plan.parallelTileSizes.size() && i < numLoops; ++i) {
    fullTileSizes[i] = plan.parallelTileSizes[i];
  }
} else {
  // Q3b: chain이 있으면 elementwise root는 parallel dims만 있으므로
  // parallelTileSizes를 _____ 사용
  fullTileSizes = plan._____;
}


// ============================================================
// Quiz 4: fusionControlFn — 어떤 producer를 fuse할 것인가
//
// 기존에는 destination operand의 fill만 fuse했지만,
// 이제는 conv와 elementwise generic도 fuse해야 합니다.
// 각 producer 종류별 판단 로직을 채우세요.
// ============================================================

tileAndFuseOptions.fusionControlFn =
    [](tensor::ExtractSliceOp, OpResult originalProducer,
       bool isDestinationOperand)
        -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
  Operation *producer = originalProducer.getOwner();

  // Q4a: fill은 항상 fuse
  if (isa<_____>(producer))
    return scf::SCFTileAndFuseOptions::ControlFnResult{false};

  // Q4b: conv도 fuse (bias_add의 input producer)
  if (isa<_____>(producer))
    return scf::SCFTileAndFuseOptions::ControlFnResult{false};

  // Q4c: elementwise generic도 fuse (chain의 intermediate ops)
  if (auto genericOp = dyn_cast<linalg::GenericOp>(producer)) {
    if (_____(genericOp))
      return scf::SCFTileAndFuseOptions::ControlFnResult{false};
  }

  // Q4d: 그 외는 fuse 거부 — 왜 nullopt을 반환하는가?
  // hint: nullopt은 "이 producer는 fuse하지 말라"는 의미
  return _____;
};


// ============================================================
// Quiz 5: Cleanup — dead op 정리
//
// fuse 후 원본 op들이 dead가 됩니다. 어떤 op들을 정리해야 하는지,
// 그리고 왜 use_empty() 체크가 필요한지 생각하세요.
// ============================================================

// Q5a: tiling root (relu)가 dead이면 삭제
if (tilingRoot->_____())
  rewriter.eraseOp(tilingRoot);

// Q5b: conv 원본이 tiling root와 다르고 dead이면 삭제
if (tilingRoot != plan.operation && plan.operation->_____())
  rewriter.eraseOp(plan.operation);

// Q5c: chain의 intermediate ops (bias_add 등) 정리
for (Operation *op : consumerChain) {
  if (op != _____ && op->use_empty())
    rewriter.eraseOp(op);
}

// Q5d: API가 fuse한 producer들 (fill 등) 정리
for (Operation *fusedProducer : result->_____) {
  if (fusedProducer->use_empty())
    rewriter.eraseOp(fusedProducer);
}


// ============================================================
// Quiz 6: 개념 질문 (코드 없음, 서술형)
//
// 아래 질문에 간단히 답하세요.
// ============================================================

// Q6a: 왜 multi-use인 경우 consumer chain을 끊는가?
//       (hint: tile loop 안에서만 결과가 계산되면 loop 밖의 다른 user는?)
//
// 답: _____

// Q6b: tileConsumerAndFuseProducersUsingSCF API는 어떤 방향으로 탐색하는가?
//       (forward: root → producers? 또는 backward: root → producers?)
//
// 답: _____

// Q6c: 기존 코드에서 isDestinationOperand 체크가 있었던 이유는?
//       그리고 이번에 제거한 이유는?
//
// 답 (있었던 이유): _____
// 답 (제거한 이유): _____

// Q6d: tile-and-fuse 후 원본 conv/bias_add가 IR에 남아있을 수 있다.
//       이들이 최종적으로 사라지는 시점은?
//       (hint: pipeline의 다음 pass들을 생각)
//
// 답: _____
