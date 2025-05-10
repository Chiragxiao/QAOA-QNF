# inference_quantum_flow_qaoa.py
"""
Inference-only pipeline for research: loads pretrained models, samples portfolios,
optimizes selection via QAOA, and visualizes results with concise computations.
"""
import argparse
import logging

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform

from quantum_layer import QuantumLayer
from qiskit_aer import Aer
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

# Logging config
logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_returns(path):
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    return np.log(df / df.shift(1)).dropna()

def load_models(num_features, flow_path, quantum_path):
    flow = Flow(
        transform=MaskedAffineAutoregressiveTransform(features=num_features, hidden_features=32),
        distribution=StandardNormal([num_features])
    )
    qlayer = QuantumLayer()
    flow.load_state_dict(torch.load(flow_path))
    qlayer.load_state_dict(torch.load(quantum_path))
    flow.eval(); qlayer.eval()
    return flow, qlayer

def sample_and_stats(flow, qlayer, n):
    X = flow.sample(n).detach().numpy()
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    return mu, cov

def build_qubo(mu, cov, lam):
    Q = lam * cov
    np.fill_diagonal(Q, lam * np.diag(cov) - mu)
    return Q

def solve_qaoa(Q):
    algorithm_globals.random_seed = 42
    n = Q.shape[0]
    qp = QuadraticProgram()
    qp.binary_var_list([f'x{i}' for i in range(n)])
    linear = list(np.diag(Q))
    quadratic = {(i, j): 2 * Q[i, j] for i in range(n) for j in range(i+1, n)}
    qp.minimize(linear=linear, quadratic=quadratic)
    qubo = QuadraticProgramToQubo().convert(qp)

    backend = Aer.get_backend('aer_simulator_statevector')
    sampler = Sampler()
    qaoa = QAOA(optimizer=COBYLA(maxiter=100), reps=2, sampler=sampler)
    solver = MinimumEigenOptimizer(qaoa)
    res = solver.solve(qubo)
    logging.info(f"QAOA objective: {res.fval:.4f}")
    return np.array(res.x, dtype=int)

def visualize(returns, sel, mu):
    sns.set(style='whitegrid', font_scale=1.2)
    selected = np.where(sel == 1)[0]
    df = returns.iloc[:, selected]
    pr = df.mean(axis=1)
    cum = (1 + pr).cumprod()
    peak = cum.cummax()
    draw = (cum - peak) / peak
    sharpe = pr.rolling(21).mean() / pr.rolling(21).std()

    # Compute proportional weights based on expected return
    asset_mu = mu[selected]
    # Use softmax to ensure positive weights that sum to 1
    weights = np.exp(asset_mu) / np.sum(np.exp(asset_mu))
    weights_percent = weights * 100
    labels = returns.columns[selected]

    # Calculate equal-weighted portfolio returns for comparison
    equal_returns = returns.mean(axis=1)
    equal_cum = (1 + equal_returns).cumprod()

    # Create a custom color palette
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f', '#1abc9c', 
              '#34495e', '#e67e22', '#95a5a6', '#16a085']

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Cumulative Returns with Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(equal_cum, label='Equal Weight', color='#95a5a6', linestyle='--', linewidth=2)
    ax1.plot(cum, label='Quantum Optimized', color="#3498db", linewidth=2.5)
    ax1.set_title("Cumulative Return Comparison", fontsize=14, pad=15)
    ax1.set_ylabel("Portfolio Value", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_facecolor("#f8f9fa")
    ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    
    # Add performance metrics annotation
    final_quantum_return = (cum.iloc[-1] - 1) * 100
    final_equal_return = (equal_cum.iloc[-1] - 1) * 100
    outperformance = final_quantum_return - final_equal_return
    annotation_text = f"Quantum: {final_quantum_return:.1f}%\nEqual: {final_equal_return:.1f}%\nOutperformance: {outperformance:.1f}%"
    ax1.annotate(annotation_text, xy=(0.02, 0.98), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                verticalalignment='top', fontsize=10)

    # Drawdown
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(draw.index, draw, color="#e74c3c", alpha=0.6)
    ax2.set_title("Drawdown Curve", fontsize=14, pad=15)
    ax2.set_ylabel("Drawdown", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_facecolor("#f8f9fa")

    # Rolling Sharpe Ratio
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(sharpe, color="#f39c12", linewidth=2.5)
    ax3.axhline(1, color='gray', linestyle='--', linewidth=1)
    ax3.set_title("21-Day Rolling Sharpe Ratio", fontsize=14, pad=15)
    ax3.set_ylabel("Sharpe Ratio", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_facecolor("#f8f9fa")

    # Pie Chart for Asset Allocation
    ax4 = fig.add_subplot(gs[1, 1])
    # Create labels with percentages
    pie_labels = [f'{label}\n({weight:.1f}%)' for label, weight in zip(labels, weights_percent)]
    wedges, texts, autotexts = ax4.pie(weights_percent, labels=pie_labels, 
                                      colors=colors[:len(weights_percent)],
                                      autopct='', startangle=90,
                                      wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 2})
    
    # Add percentage labels inside the pie chart
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax4.set_title("Asset Allocation Distribution", fontsize=14, pad=15)
    
    # Add a legend
    ax4.legend(wedges, labels, title="Assets", 
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # Add a title to the entire figure
    fig.suptitle("📊 Portfolio Performance & Asset Allocation Dashboard", 
                fontsize=18, fontweight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='portfolio_prices.csv')
    p.add_argument('--split_date', default='2023-01-01')
    p.add_argument('--flow_model', default='trained_flow.pth')
    p.add_argument('--quantum_model', default='trained_quantum.pth')
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--lambda', dest='risk_aversion', type=float, default=0.3)
    args = p.parse_args()

    ret = load_returns(args.data)
    train = ret[ret.index < args.split_date]

    flow, qlayer = load_models(train.shape[1], args.flow_model, args.quantum_model)
    mu, cov = sample_and_stats(flow, qlayer, args.samples)

    Q = build_qubo(mu, cov, args.risk_aversion)
    selection = solve_qaoa(Q)

    visualize(ret, selection, mu)
