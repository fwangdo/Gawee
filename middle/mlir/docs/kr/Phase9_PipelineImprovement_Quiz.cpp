//===----------------------------------------------------------------------===//
// Phase 9: Pipeline Improvement Quiz
//===----------------------------------------------------------------------===//
//
// 아래 코드의 빈칸(_____)을 채워서 완성하세요.
// 각 빈칸에는 MLIR API 호출, 조건식, 또는 타입이 들어갑니다.
//
//===----------------------------------------------------------------------===//

// ============================================================
// Quiz 1: Vectorization — vectorize 결과 처리
//
// linalg::vectorize()는 FailureOr<VectorizationResult>를 반환합니다.
// 이 결과를 어떻게 처리해야 원래 op이 vector op으로 교체되는지 생각하세요.
// ============================================================

static void vectorizeEligibleOps(ModuleOp module) {
  SmallVector<linalg::LinalgOp> candidates;
  // ... (후보 수집 생략) ...

  IRRewriter rewriter(module.getContext());
  for (linalg::LinalgOp op : candidates) {
    rewriter.setInsertionPoint(op);

    // Q1a: vectorize의 반환 타입을 채우세요
    _____<linalg::VectorizationResult> result =
        linalg::vectorize(rewriter, op);

    // Q1b: 성공 여부를 확인하고, 원래 op을 대체하세요
    // hint: result->replacements 필드를 사용
    if (_____(result)) {
      rewriter._____(op, result->_____);
    }
  }
}


// ============================================================
// Quiz 2: isVectorizableElementwise — projected permutation 필터
//
// broadcast map (d0,d1,d2,d3) -> (d1) 은 vectorize 가능합니다.
// 하지만 0-result map (d0,d1,d2) -> () 는 특별 처리가 필요합니다.
// ============================================================

static bool isVectorizableElementwise(linalg::LinalgOp op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op.getOperation());
  if (!genericOp)
    return false;

  // Q2a: 모든 iterator가 parallel인지 확인
  auto iteratorTypes = op.getIteratorTypesArray();
  if (!llvm::all_of(iteratorTypes, [](utils::IteratorType t) {
        return t == utils::IteratorType::_____;
      }))
    return false;

  // Q2b: indexing map 필터 — projected permutation 허용
  for (AffineMap map : genericOp.getIndexingMapsArray()) {
    // Q2c: 0-result map은 scalar broadcast. 왜 별도 처리가 필요한가?
    // hint: isProjectedPermutation()이 0-result map에 대해 _____를 반환하기 때문
    if (map._____() == 0)
      continue;
    if (!map._____())
      return false;
  }
  return true;
}


// ============================================================
// Quiz 3: Tile-and-Fuse — fusionControlFn 설계
//
// conv의 tile-and-fuse에서 어떤 producer를 fuse하고 어떤 것을 거부해야 하는지.
// ============================================================

scf::SCFTileAndFuseOptions tileAndFuseOptions;
tileAndFuseOptions.setTilingOptions(tilingOptions);
tileAndFuseOptions.fusionControlFn =
    [](tensor::ExtractSliceOp, OpResult originalProducer,
       bool isDestinationOperand)
        -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {

      // Q3a: input producer는 왜 fuse하면 안 되는가?
      // hint: conv의 input은 _____ 때문에 output tile과 다른 크기의 slice 필요
      if (!_____)
        return std::nullopt;

      // Q3b: destination operand 중에서도 fill만 fuse. 왜?
      Operation *producer = originalProducer.getOwner();
      if (!isa<linalg::_____>(producer))
        return std::nullopt;

      // Q3c: yieldProducerReplacement를 false로 설정하는 이유는?
      // hint: fill은 conv의 _____ 로만 사용되므로 loop 밖에서 대체할 필요 없음
      return scf::SCFTileAndFuseOptions::ControlFnResult{
          /*yieldProducerReplacement=*/_____};
    };


// ============================================================
// Quiz 4: Vector lowering pipeline 순서
//
// vectorize 후 LLVM까지 내리려면 어떤 pass가 어떤 순서로 필요한지.
// ============================================================

// Q4: 올바른 순서로 배열하세요 (번호를 매기세요)
//
// ___: createConvertVectorToSCFPass()     // vector.transfer_read → SCF
// ___: bufferization (OneShotBufferize)   // tensor → memref
// ___: createConvertVectorToLLVMPass()    // vector → LLVM
// ___: createUBToLLVMConversionPass()     // ub.poison → LLVM
// ___: createConvertLinalgToLoopsPass()   // linalg → SCF loops
//
// 왜 VectorToSCF가 bufferization 후에 와야 하는가?
// hint: vector.transfer_read는 _____ 타입 operand가 필요


// ============================================================
// Quiz 5: opt vs llc — LLVM 도구의 역할 구분
//
// AOT 파이프라인에서 mlir-translate로 생성한 LLVM IR을 기계어로 변환할 때,
// 어떤 도구가 어떤 역할을 하는지 이해해야 합니다.
// ============================================================

// Q5a: 빈칸을 채우세요.
//
// _____: LLVM middle-end optimizer.
//        LoopVectorize, SLPVectorize, GVN, LICM 등 수행.
//        LLVM IR → 최적화된 LLVM IR.
//
// _____: LLVM backend code generator.
//        instruction selection, register allocation 수행.
//        LLVM IR → 기계어 (object file).

// Q5b: 올바른 AOT 파이프라인 순서를 완성하세요.
//
// mlir-translate --mlir-to-llvmir input.mlir -o lowered.ll
// _____ -O2 -S lowered.ll -o optimized.ll     // middle-end 최적화
// _____ -O2 -filetype=obj optimized.ll -o lowered.o  // codegen
// clang++ -O2 launcher.cpp lowered.o -o runner       // link

// Q5c: llc -O2만 사용했을 때 성능이 변하지 않는 이유는?
// hint: llc는 _____ 만 수행하고, _____ 는 수행하지 않기 때문.
//       scalar loop를 SIMD화하는 것은 _____ 의 역할.


// ============================================================
// Quiz 6: MLIR Fusion과 LLVM Auto-Vectorization의 간섭
//
// elementwise fusion이 LLVM LoopVectorize와 어떻게 충돌하는지.
// ============================================================

// Q6a: fusion이 bert_tiny를 느리게 만든 메커니즘:
//
// fusion 전: add, mul, div 각각이 별도의 _____ 루프
//            → LLVM이 각 루프를 독립적으로 _____
// fusion 후: 하나의 linalg.generic에 복잡한 _____ map
//            → LLVM LoopVectorize가 loop body를 분석하지 못해 _____

// Q6b: fusion을 다시 활성화하려면 어떤 조건이 필요한가?
// hint: LLVM auto-vectorization에 의존하지 않으려면
//       MLIR 수준에서 fuse한 뒤 _____도 함께 수행해야 함


// ============================================================
// 정답은 Summary 문서와 실제 소스 코드를 참고하세요:
//   - lib/Conversion/LinalgVectorization.cpp
//   - lib/Conversion/LinalgTransform.cpp
//   - lib/Conversion/LinalgFusion.cpp
//   - back/gawee_aot.cpp
//   - tools/gawee-opt.cpp
// ============================================================
