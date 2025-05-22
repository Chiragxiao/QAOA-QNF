### ✅ `README.md`

```markdown
# Quantum-Enhanced Normalizing Flow and QAOA for Portfolio Optimization

This project presents a hybrid quantum-classical framework that combines **Quantum Approximate Optimization Algorithm (QAOA)** with a **Quantum-Enhanced Normalizing Flow (QNF)** model to solve financial **portfolio optimization** problems.

By leveraging the power of **variational quantum circuits** and **probabilistic deep learning**, this approach aims to outperform traditional classical optimization methods in high-dimensional, constraint-heavy financial environments.

---

## 📌 Key Features

- 🧠 **QNF-based modeling**: Captures complex distributions over portfolio weights.
- ⚛️ **QAOA integration**: Optimizes asset selection using quantum variational circuits (Qiskit).
- 🔄 **Hybrid workflow**: Combines classical deep learning (PyTorch) with quantum computing (Qiskit).
- 📊 **Portfolio performance evaluation**: Includes Sharpe ratio, expected returns, and risk assessment.

---

## 📂 Project Structure

```

.
├── src/                        # Source code
│   ├── collect\_data.py
│   ├── load\_data.py
│   ├── normalizing\_flow\.py
│   ├── quantum\_portfolio\_optimizer.py
│   └── quantum\_portfolio\_validator.py
│
├── data/                      # Input datasets
├── results/                   # Generated figures and logs
├── validation/                # Cleaned performance reports
├── requirements.txt
├── LICENSE
└── README.md

````

---

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Chiragxiao/qnf-qaoa.git
cd qnf-qaoa-portfolio-opt
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

> ⚠️ Python 3.8+ is recommended. Make sure you have Qiskit and PyTorch properly installed.

---

## 🚀 Usage

Run the hybrid optimization workflow:

```bash
python src/quantum_portfolio_optimizer.py
```

Evaluate results:

```bash
python src/quantum_portfolio_validator.py
```

---

## 📈 Results

Key benchmarks, graphs, and results can be found in the [`results/`](./results/) and [`validation_results/`](./validation_results/) folders.

* Improved portfolio return vs. classical models
* Higher Sharpe ratios
* QAOA parameter convergence plots

---

## 🧪 Dependencies

* `qiskit`
* `torch`
* `numpy`
* `scipy`
* `matplotlib`
* `pandas`
* `seaborn`

All dependencies are listed in [`requirements.txt`](./requirements.txt)

---

## 🙋‍♂️ Contact

For questions or collaborations, reach out via [LinkedIn](https://www.linkedin.com/in/chirag-solanki98) or create a GitHub issue.
