# Meta-Learning — Learning to Learn
### How MAML Learns an Initialisation That Adapts to Any Task in a Few Gradient Steps

**MLNN Tutorial | University of Hertfordshire | 2025**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## What is this?

This repository contains the complete tutorial submission for the MLNN Assignment at the University of Hertfordshire. The tutorial teaches **meta-learning**, with a focus on **MAML (Model-Agnostic Meta-Learning)** — the most influential optimisation-based meta-learning algorithm.

**Core question MAML answers:** What initialisation θ* allows a model to fine-tune to any new task in the fewest possible gradient steps?

---

## Repository Contents

| File | Description |
|------|-------------|
| `meta_learning_tutorial.pdf` | PDF tutorial (< 2000 words) — main submission |
| `meta_learning_tutorial.ipynb` | Jupyter notebook — full runnable code |
| `README.md` | This file |
| `LICENSE` | MIT licence |

---

## Tutorial Overview

### Section 1 — The Meta-Learning Framework
Taxonomy of three families: optimisation-based (MAML, Reptile), metric-based (ProtoNet, MatchNet), model-based (SNAIL, NTM). MAML algorithm steps with formal equations.

### Section 2 — MAML Deep Dive
Formal objective, inner loop implementation with `create_graph=True`, outer meta-update, why second-order gradients matter.

### Section 3 — Experiments: Sinusoid Regression
Standard MAML benchmark. Real training results: meta-loss 5.65 → 1.26 over 400 iterations. Adaptation comparison showing MAML vs random initialisation on unseen tasks.

### Section 4 — K-Shot Analysis
Adaptation curves for K=1,5,10,20 support examples. MAML vs random across gradient step budgets.

### Section 5 — Algorithm Comparison
MAML vs Reptile vs ProtoNet vs MatchNet across key dimensions.

### Section 6 — Why MAML Works
Geometric interpretation (flat regions) and Bayesian interpretation (amortised inference). Connection to regularisation, transfer learning, and Siamese networks.

---

## Key Results

| Metric | MAML | Random Init |
|--------|------|-------------|
| 1-step MSE (K=5) | 3.36 | 4.74 |
| 10-step MSE (K=5) | 3.94 | 4.44 |
| Meta-loss (start) | 5.65 | — |
| Meta-loss (final) | 1.26 | — |

MAML consistently outperforms random initialisation across all gradient step budgets.

---

## How to Run

### Requirements

```bash
pip install torch numpy matplotlib
```

Python 3.8+. No GPU required — training runs on CPU in ~3 minutes.

### Steps

```bash
git clone https://github.com/yourusername/meta-learning-tutorial.git
cd meta-learning-tutorial
pip install torch numpy matplotlib
jupyter notebook meta_learning_tutorial.ipynb
```

All data is **generated synthetically** (sinusoid tasks) — no external downloads needed.

### Running order

| Cell | Content |
|------|---------|
| Cell 2 | Task distribution visualisation |
| Cell 3 | SineNet model + functional forward |
| Cell 4 | MAML inner loop (`create_graph=True`) |
| Cell 5 | Meta-training loop (400 iterations) |
| Cell 6 | Figure 2: task distribution + training loss + adaptation quality |
| Cell 7 | Figure 3: K-shot curves + MAML vs random |
| Cell 8 | Reptile: first-order alternative |

---

## Accessibility

- **Colourblind-safe** — tab10 palette throughout
- **Dual encoding** — line plots use colour + distinct styles/markers
- **Diagram accessibility** — taxonomy boxes contain text labels, readable without colour
- **Structured headings** — H1/H2/H3 for screen reader navigation
- **Figure captions** — full descriptive captions on every figure
- **Code comments** — every non-obvious line explained inline

---

## References

1. Finn, C., Abbeel, P. and Levine, S. (2017) 'Model-agnostic meta-learning for fast adaptation of deep networks', *ICML 2017*. https://arxiv.org/abs/1703.03400

2. Nichol, A., Achiam, J. and Schulman, J. (2018) 'On first-order meta-learning algorithms', *arXiv:1803.02999*. https://arxiv.org/abs/1803.02999

3. Snell, J., Swersky, K. and Zemel, R. (2017) 'Prototypical networks for few-shot learning', *NeurIPS 2017*. https://arxiv.org/abs/1703.05175

4. Vinyals, O., Blundell, C., Lillicrap, T., Kavukcuoglu, K. and Wierstra, D. (2016) 'Matching networks for one shot learning', *NeurIPS 2016*. https://arxiv.org/abs/1606.04080

5. Hospedales, T., Antoniou, A., Micaelli, P. and Storkey, A. (2022) 'Meta-learning in neural networks: A survey', *IEEE TPAMI*, 44(9). https://arxiv.org/abs/2004.05439

---

## Licence

MIT — see [LICENSE](LICENSE). Free to use, adapt, and build on with attribution.

---

**Author:** University of Hertfordshire — MLNN Assignment 2025
