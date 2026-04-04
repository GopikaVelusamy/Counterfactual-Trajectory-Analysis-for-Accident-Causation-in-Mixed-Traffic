# Counterfactual Trajectory Analysis for Accident Causation in Mixed Traffic

---

## Overview

Road traffic accidents cause approximately **1.35 million deaths annually** (WHO). Traditional accident prediction models treat each vehicle in isolation and cannot explain *why* a scene is dangerous or *what should be done* to prevent it.

This project proposes a **three-phase Graph Neural Network pipeline** that:
1. Predicts whether a traffic scene is accident-prone or normal
2. Improves prediction using attention-based graph learning and feature engineering
3. Identifies **which specific vehicles** are responsible and **what action** (braking, speed reduction, lane change) would prevent the accident — using counterfactual reasoning

---

## Project Structure

```
counterfactual-traffic-accident-prediction/
├── counterfactual_traffic_accident_gnn.ipynb   # Main code (all 3 phases)
├── requirements.txt                            # Dependencies
├── results/
│   ├── run1_distributed_risk.png               # Counterfactual output - Run 1
│   ├── run2_concentrated_risk.png              # Counterfactual output - Run 2
│   └── run3_safe_scenario.png                  # Counterfactual output - Run 3
├── docs/
│   ├── dissertation.pdf                        # Full project report
│   └── presentation.pptx                      # Final review slides
└── README.md
```

---

## Three-Phase Methodology

### Phase 1 — Graph Convolutional Network (GCN)
- Each traffic scene is modelled as a graph: agents = nodes, interactions = edges (KNN, k=5)
- Node features: `[x, y, vx, vy]` — 4 dimensions
- 2-layer GCN with global mean pooling and class-weighted cross-entropy loss
- **Test Accuracy: 75.90%**

### Phase 2 — Graph Attention Network (GAT)
- Upgraded to multi-head GAT (4 heads) with enriched **11-dimensional features**:
  `[x, y, vx, vy, speed, heading, is_car, is_truck, is_ped, length, width]`
- Attention mechanism assigns different importance to each neighbouring vehicle
- **Test Accuracy: 77.91% | Best Validation Accuracy: 80.32%**

### Phase 3 — Counterfactual Explainability
- For each accident-prone scene, three interventions are applied to each vehicle independently:
  - **Speed Reduction** — scale velocity by 0.8
  - **Early Braking** — reduce velocity magnitude
  - **Lane Change** — shift lateral position
- A vehicle is **causal** if: `Δp > 0.4` OR `(prediction flips AND Δp > 0.3)`
- Transforms the model from a passive classifier into an **active decision-support tool**

---

## Results

### Model Comparison

| Model | Input Features | Best Val Accuracy | Test Accuracy |
|-------|---------------|-------------------|---------------|
| GCN   | 4 (raw)       | 73.09%            | 75.90%        |
| GAT   | 11 (engineered) | **80.32%**      | **77.91%**    |

> Majority class baseline (always predict Normal) = 60%

### Counterfactual Analysis

| Run | Prediction | Accident Probability | Agents | Causal Vehicles |
|-----|-----------|----------------------|--------|-----------------|
| 1   | Accident  | 0.537                | 18     | None (distributed risk) |
| 2   | Accident  | 0.560                | 17     | 8 vehicles [0,1,4,7,8,9,11,13] |
| 3   | Normal    | 0.002                | 18     | None (safe scene) |

**Key finding:** Early braking was the most effective intervention. Lane change was consistently ineffective — confirming that velocity-based risk dominates over positional changes in dense traffic.

---

## Key Contributions

1. End-to-end GNN pipeline in PyTorch + PyTorch Geometric for accident scene classification
2. Quantitative proof of GAT superiority over GCN — **+7.23% validation accuracy**
3. Novel **dual-threshold causal criterion** for counterfactual vehicle identification
4. Empirical discovery of two distinct accident risk patterns:
   - **Concentrated risk** — specific vehicles are individually causal (fixable)
   - **Distributed risk** — risk is collectively emergent (requires system-level intervention)

---

## Dataset

**DeepAccident** — A simulation-based benchmark using the CARLA autonomous driving simulator.

- 1,658 scenario frames (subset used for this project)
- Class distribution: ~40% accident, ~60% normal
- Each frame: complete per-agent state (position, velocity, type, dimensions)
- Split: 70% train / 15% validation / 15% test (stratified)

Download: [DeepAccident — arXiv:2304.01168](https://arxiv.org/abs/2304.01168)

> **Note:** The dataset is not included in this repository due to size and licensing. Download it separately from the link above.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| PyTorch | Deep learning framework |
| PyTorch Geometric (PyG) | GCN and GAT layers, graph data handling |
| scikit-learn | KNN graph construction, class weighting, metrics |
| pandas / numpy | Data loading and feature engineering |
| matplotlib / networkx | Visualisation of interaction graphs |

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/counterfactual-traffic-accident-prediction.git
cd counterfactual-traffic-accident-prediction

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook counterfactual_traffic_accident_gnn.ipynb
```

> The notebook is structured in order — run Phase 1 cells first, then Phase 2, then Phase 3.  
> Update the `dataset_path` variable in Cell 2 to point to your local DeepAccident dataset folder.

---

## Future Work

- Extend counterfactual module to **multi-vehicle simultaneous interventions** (to handle distributed risk scenarios)
- Incorporate **temporal GNNs** (STGCN / Temporal Graph Networks) for multi-frame accident prediction
- Add **physics-based trajectory simulation** into the intervention module
- Validate on real-world datasets: **NGSIM, HighD, nuScenes**
- Incorporate **edge features** (inter-vehicle distance, relative speed, angle of approach)
- Explore **heterogeneous graphs** with type-specific encoders for cars, trucks, and pedestrians

---

## References

1. Wang et al. — *DeepAccident: A Motion and Accident Prediction Benchmark for V2X Autonomous Driving*, AAAI 2024
2. Kipf & Welling — *Semi-Supervised Classification with Graph Convolutional Networks*, 2017
3. Veličković et al. — *Graph Attention Networks*, 2018
4. Wachter et al. — *Counterfactual Explanations without Opening the Black Box*, 2017
