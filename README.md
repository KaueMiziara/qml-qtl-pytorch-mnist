# QML Benchmarking: PyTorch, PennyLane & Qiskit

This project implements a **Hybrid Quantum-Classical Neural Network** to solve a
binary classification task on the MNIST dataset.

The primary goal is to benchmark the integration of Quantum Machine
Learning (QML) frameworks with PyTorch, comparing a Classical Baseline
against **PennyLane** and **Qiskit** implementations.

The focus is not just on accuracy, but on the computational implications
of simulating quantum circuits on classical hardware.

## Project Architecture

The model follows a modular architecture, where the central processing
unit (Backbone) can be swapped via dependency injection without
altering the input/output pipelines.

### 1. Data Processing

- **Source:** MNIST Dataset.
- **Filter:** Binary classification (digits `0` and `1` only).
- **Transformation:** Resized to `4x4` pixels (16 input features) and normalized.

### 2. The Hybrid Classifier

The model consists of three distinct modules:

- **Pre-Net (Classical):** A Linear layer that compresses the 16 input pixels
  into 4 features and scales them to the range $[-\pi, \pi]$ for angle embedding.
- **Backbone (Modular):** The core processing unit.
  - **Classical Backbone:** A standard PyTorch `nn.Linear` layer used
    as a baseline surrogate.
  - **PennyLane Backbone:** A Variational Quantum Circuit (VQC)
    using `AngleEmbedding` and `StronglyEntanglingLayers`.
  - **Qiskit Backbone:** A VQC using `RX` gates for encoding and the
    `EfficientSU2` Ansatz with circular entanglement.
- **Post-Net (Classical):** A final Linear layer and Sigmoid activation to map the
  backbone's measurement/output to a class probability.

## Phase 1: Simulation Benchmarks

In this phase, all models were trained and evaluated on a CPU.

### Results

- **Accuracy:** Both quantum backbones achieved parity with the classical
  baseline, reaching **~99% accuracy**. This validates that the hybrid architecture
  successfully backpropagates gradients through the quantum layers.
- **Runtime:** Significant disparity was observed between frameworks.
  - **Classical:** Instantaneous.
  - **PennyLane:** Fast. Used internal backpropagation (adjoint differentiation)
    , treating the simulator as a differentiable tensor engine.
  - **Qiskit:** Slow (~ orders of magnitude). Used
    **Parameter Shift Rules** ($2N$ evaluations per step).

### Discussion

The runtime difference is a feature of the methodology, not a bug.

- **PennyLane** was optimized for **Simulation Speed**, leveraging mathematical
  "shortcuts" available only on classical CPUs (Backpropagation).
- **Qiskit** was configured for **Hardware Fidelity**. It used the Parameter
  Shift rule, which is the only valid method for calculating gradients
  on real quantum hardware (where the state vector is not observable).

This highlights the massive computational overhead required to simulate
faithful quantum mechanics versus using mathematical optimizations.

## Phase 2: Hardware Execution

_Work in progress._

The objective of Phase 2 is to deploy the trained Qiskit
and PennyLane backbones onto real Quantum Processing Units (QPUs)
to evaluate noise resilience and actual hardware runtime performance.
