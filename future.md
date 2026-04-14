# Future Directions

This project began as an AI compiler prototype focused on graph lowering and optimization, using a Python FX-based frontend and an MLIR-based middle-end.

For a paper-oriented problem definition grounded in prior work, see:

- `docs/research_direction.md`

The long-term goal is to evolve it into a full-stack inference system that includes a lightweight runtime and a system-level execution engine.

Upcoming development is centered on two main areas:

1. Inference runtime implementation
2. System-level execution and concurrency

## 1. Inference Runtime

The current compiler produces an intermediate representation and applies optimization passes. The next step is to build a runtime executor capable of executing compiled operators efficiently.

### Execution Engine

A graph execution engine will be implemented to execute compiled operators in dependency order.

Core components:

- Operator scheduler
- Kernel dispatcher
- Execution graph traversal

Example structure:

```text
ExecutionEngine
├─ Graph
├─ NodeScheduler
├─ KernelDispatcher
└─ Profiler
```

The runtime will initially target CPU execution, with possible future extensions to GPU backends.

### Memory Planning

Efficient tensor memory management will be a core part of the runtime.

Goals:

- Reduce peak memory usage
- Reuse buffers across operators
- Minimize allocations during execution

Possible techniques:

- Static memory planning
- Lifetime analysis
- Memory pool allocator

### Profiling

Profiling support will be integrated to analyze runtime performance.

Potential metrics:

- Operator execution latency
- Memory usage
- Scheduling overhead

Tools under consideration:

- `perf` on Linux
- Custom runtime instrumentation

## 2. System-Level Execution

The runtime is intended to become a system-level execution engine rather than a simple operator interpreter.

This includes:

- Multi-threaded execution
- Asynchronous task scheduling
- Efficient synchronization

### Thread Pool Execution

A custom thread pool will be implemented to execute operators in parallel.

Possible architecture:

```text
ThreadPool
├─ WorkerThreads
├─ TaskQueue
└─ WorkStealingScheduler
```

Expected benefits:

- Parallel operator execution
- Improved CPU utilization
- Lower scheduling overhead

### Task Graph Scheduling

Execution will follow a dependency-graph model.

Example operator graph:

```text
Conv -> ReLU -> Add
     \
      -> BatchNorm
```

Operators can be executed concurrently once their dependencies are satisfied.

### Low-Overhead Synchronization

To reduce synchronization overhead, the runtime will explore:

- Lock-free queues
- Atomic operations
- Fine-grained synchronization

## 3. Linux System Programming

The runtime will be implemented with a strong focus on Linux system-level performance.

Topics to explore include:

- Thread scheduling
- Memory mapping (`mmap`)
- CPU affinity
- NUMA awareness

This direction helps the runtime behave more like a high-performance system component than a simple library.

## 4. Long-Term Goals

The broader objective is to bridge the gap between the following layers:

```text
ML Compiler
    ↓
Inference Runtime
    ↓
System-Level Execution
```

By gradually extending the project into a complete ML execution stack, it can explore the intersection of:

- Compiler engineering
- Runtime systems
- High-performance computing
- System programming
