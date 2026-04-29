# cmake_exercise

이 디렉토리는 `middle/mlir` 본체와 분리된 **독립 MLIR 따라치기 연습장**이다.

핵심 원칙:
- production 코드를 읽는다
- 여기서 같은 파일을 직접 다시 친다
- 파일 하나 끝날 때마다 바로 build한다

자세한 순서와 빌드 방법은 [order.md](./order.md)에 정리되어 있다.

현재 원칙은 단순하다:
- `src/main.cpp`만 reference를 남긴다
- 나머지는 가능한 한 `# TODO`만 남긴다
- pass 파일들은 부분 scaffold가 아니라 **통째로 다시 치는 방식**으로 연습한다
