# portfolio_validator.py
"""
Portfolio Optimization Validator

This script validates the results of the quantum portfolio optimizer by:
1. Loading the quantum optimizer results
2. Implementing several classical portfolio optimization methods for comparison
3. Visualizing performance metrics across all methods
4. Creating a comparative analysis dashboard
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.optimize import minimize
from typing import Dict
from datetime import datetime
from scipy import stats

# Classical optimization libraries
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Import your quantum optimizer API (adjust filename/module as needed)
from quantum_portfolio_optimizer import (
    QuantumEnhancedFlow,
    train_quantum_flow,
    generate_scenarios,
    build_qubo,
    solve_with_qaoa,
    evaluate_portfolio
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PortfolioValidator")

np.random.seed(42)
torch.manual_seed(42)


def load_optimizer_results(results_dir: str) -> Dict:
    """Load results from the quantum portfolio optimizer."""
    logger.info(f"Loading optimizer results from {results_dir}")
    results = {}

    # 1) Load selection vector
    sel_path = os.path.join(results_dir, 'asset_selection.csv')
    if os.path.exists(sel_path):
        df = pd.read_csv(sel_path, header=None)
        # assume single-row or single-column of 0/1 flags
        arr = df.values.flatten()
        results['selection'] = arr.astype(int)
        logger.info(f"Loaded selection vector → {int(arr.sum())} assets selected")

    # 2) Load performance metrics
    log_path = os.path.join(results_dir, 'optimization.log')
    if os.path.exists(log_path):
        with open(log_path) as f:
            text = f.read()
        try:
            ann_ret = float(text.split('Annualized Return:')[1].split('%')[0].strip()) / 100
            vol    = float(text.split('Volatility:')[1].split('%')[0].strip()) / 100
            sharpe = float(text.split('Sharpe:')[1].split()[0])
            results['metrics'] = {
                'annualized_return': ann_ret,
                'volatility':        vol,
                'sharpe_ratio':      sharpe
            }
            logger.info(f"Loaded metrics → {results['metrics']}")
        except Exception as e:
            logger.warning(f"Could not parse metrics from log: {e}")

    return results


def load_and_prepare_data(data_file: str):
    """
    Load and prepare price data for validation.
    Returns the full returns, train & test splits.
    """
    logger.info(f"Loading price data from {data_file}")
    prices = pd.read_csv(data_file, index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()
    split = int(len(returns) * 0.8)
    train = returns.iloc[:split]
    test  = returns.iloc[split:]
    logger.info(f"Data: {len(returns)} days, train={len(train)}, test={len(test)}, assets={returns.shape[1]}")
    return returns, train, test


def run_equal_weight(returns: pd.DataFrame) -> np.ndarray:
    n = returns.shape[1]
    w = np.ones(n) / n
    return w


def run_minimum_variance(returns: pd.DataFrame) -> np.ndarray:
    C = returns.cov().values + 1e-6 * np.eye(returns.shape[1])
    invC = np.linalg.inv(C)
    ones = np.ones(len(C))
    w = invC @ ones
    w = np.maximum(w, 0)
    w /= w.sum()
    return w


def run_maximum_sharpe(returns: pd.DataFrame, rf=0.02) -> np.ndarray:
    mu  = returns.mean().values
    C   = returns.cov().values + 1e-6 * np.eye(returns.shape[1])
    invC= np.linalg.inv(C)
    ex = mu - rf
    w  = invC @ ex
    w  = np.maximum(w, 0)
    w /= w.sum()
    return w


def calculate_annualized_metrics(r: pd.Series) -> Dict:
    # daily → annual
    ann_ret = (1 + r).prod() ** (252/len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = (ann_ret - 0.02) / ann_vol if ann_vol>0 else 0
    return {
        'annualized_return': ann_ret,
        'volatility':        ann_vol,
        'sharpe_ratio':      sharpe
    }


def evaluate_portfolio_performance(returns: pd.DataFrame, weights: np.ndarray) -> Dict:
    w = weights / weights.sum()
    port = returns.dot(w)
    return calculate_annualized_metrics(port)


def compare_methods(returns: pd.DataFrame, opt_results: Dict) -> pd.DataFrame:
    rows = []
    # Quantum result
    if 'metrics' in opt_results:
        rows.append({'Method':'Quantum-QAOA', **opt_results['metrics']})
    # Classical
    for name, func in [
        ('Equal-Weight', run_equal_weight),
        ('Min-Variance',  run_minimum_variance),
        ('Max-Sharpe',    run_maximum_sharpe)
    ]:
        w = func(returns)
        m = evaluate_portfolio_performance(returns, w)
        rows.append({'Method':name, **m})
    return pd.DataFrame(rows)


def perform_stat_tests(returns: pd.DataFrame, opt_results: Dict, n=1000) -> pd.DataFrame:
    if 'metrics' not in opt_results:
        return None
    qsh = opt_results['metrics']['sharpe_ratio']
    rand = []
    for _ in range(n):
        w = np.random.rand(returns.shape[1])
        w /= w.sum()
        m = returns.dot(w)
        rand.append(calculate_annualized_metrics(m)['sharpe_ratio'])
    rand = np.array([r for r in rand if np.isfinite(r)])
    pval = 1 - stats.percentileofscore(rand, qsh)/100
    return pd.DataFrame({
        'Test':['Sharpe'],
        'Quantum':[qsh],
        'Random_Mean':[rand.mean()],
        'Random_Std':[rand.std()],
        'P-Value':[pval],
        'Significant (5%)':[pval<0.05]
    })


def generate_visualizations(df_cmp: pd.DataFrame, df_test: pd.DataFrame, out: str):
    os.makedirs(out, exist_ok=True)
    sns.set_style("whitegrid")

    # Bar plot
    plt.figure(figsize=(8,4))
    df_cmp.set_index('Method')[['sharpe_ratio','annualized_return','volatility']] \
          .plot(kind='bar', rot=0)
    plt.title("Performance Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'performance.png'))
    plt.close()

    # Stat test plot
    if df_test is not None:
        plt.figure(figsize=(6,4))
        plt.hist(df_test['Random_Mean'], bins=30, alpha=0.7,
                 label='Random Sharpe Mean')
        plt.axvline(df_test['Quantum'].iloc[0], color='red',
                    label=f"Quantum (p={df_test['P-Value'].iloc[0]:.3f})")
        plt.legend()
        plt.title("Sharpe Ratio Significance")
        plt.tight_layout()
        plt.savefig(os.path.join(out, 'stat_test.png'))
        plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_file',       required=True)
    p.add_argument('--optimizer_dir',   required=True)
    p.add_argument('--output_dir',      default='validation_results')
    p.add_argument('--n_trials',  type=int, default=1000)
    args = p.parse_args()

    # 1) Load data
    prices = pd.read_csv(args.data_file, index_col=0, parse_dates=True)
    returns, train, test = load_and_prepare_data(args.data_file)

    # 2) Load quantum results
    qr = load_optimizer_results(args.optimizer_dir)

    # 3) Compare
    cmp_df = compare_methods(returns, qr)
    logger.info("\n" + cmp_df.to_string())

    # 4) Stats
    stat_df = perform_stat_tests(returns, qr, args.n_trials)
    if stat_df is not None:
        logger.info("\n" + stat_df.to_string())

    # 5) Plots
    generate_visualizations(cmp_df, stat_df, args.output_dir)

    # 6) Save
    cmp_df.to_csv(os.path.join(args.output_dir, 'comparison.csv'), index=False)
    if stat_df is not None:
        stat_df.to_csv(os.path.join(args.output_dir, 'stat_tests.csv'), index=False)

    logger.info("✅ Validation complete")


if __name__ == '__main__':
    main()
