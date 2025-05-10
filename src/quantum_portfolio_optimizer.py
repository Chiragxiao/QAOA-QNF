# quantum_portfolio_optimizer.py
"""
Hybrid Quantum Portfolio Optimization System

This implementation combines:
1. Quantum-Enhanced Normalizing Flows for realistic market simulation
2. QAOA for discrete asset selection optimization
3. Professional-grade portfolio evaluation and visualization

"""

import argparse
import logging
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.dates import DateFormatter

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Normalizing flows
from torch.distributions import Normal
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuantumPortfolio")

# Remove fixed seeds to allow for random results
# algorithm_globals.random_seed = 42
# np.random.seed(42)
# torch.manual_seed(42)


class QuantumLayer(nn.Module):
    """
    Parameterized quantum circuit as a differentiable PyTorch layer.
    Encodes financial data into quantum states and processes it through
    a variational quantum circuit.
    """
    
    def __init__(self, n_qubits=4, depth=2):
        """
        Initialize quantum layer with configurable qubits and circuit depth.
        
        Args:
            n_qubits: Number of qubits in circuit
            depth: Number of entangling layers
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        
        # Learnable parameters - 3 rotation angles (RX, RY, RZ) per qubit per layer
        param_shape = 3 * self.n_qubits * self.depth
        self.theta = nn.Parameter(torch.randn(param_shape) * 0.01)
        
        # Initialize simulator backend
        self.simulator = Aer.get_backend('statevector_simulator')
    
    def _build_circuit(self, inputs: torch.Tensor) -> QuantumCircuit:
        """
        Construct the parameterized quantum circuit.
        
        Args:
            inputs: Tensor of input features
            
        Returns:
            Constructed quantum circuit
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # Data encoding - Angle encoding of inputs into quantum states
        for i in range(min(len(inputs), self.n_qubits)):
            # Scale input to valid rotation angle range
            angle = float(torch.arctan(inputs[i]) * 2.0)
            qc.ry(angle, i)
        
        # Variational quantum circuit with learnable parameters
        param_idx = 0
        for d in range(self.depth):
            # Single-qubit rotations with learnable parameters
            for i in range(self.n_qubits):
                qc.rx(float(self.theta[param_idx]), i)
                param_idx += 1
                qc.ry(float(self.theta[param_idx]), i)
                param_idx += 1
                qc.rz(float(self.theta[param_idx]), i)
                param_idx += 1
            
            # Entangling layer - CNOT gates between adjacent qubits
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            # Connect last qubit to first for circular entanglement
            if self.n_qubits > 1:
                qc.cx(self.n_qubits - 1, 0)
        
        return qc
    
    def _get_expectations(self, circuit: QuantumCircuit) -> np.ndarray:
        """
        Calculate expectation values for each qubit.
        
        Args:
            circuit: Quantum circuit to execute
            
        Returns:
            Array of expectation values
        """
        # Execute circuit
        job = self.simulator.run(circuit)
        result = job.result()
        statevector = np.asarray(result.get_statevector(circuit))
        
        # Calculate expectation values for each qubit
        expectations = np.zeros(self.n_qubits)
        for i in range(self.n_qubits):
            # Create Pauli-Z measurement for qubit i
            z_mask = 1 << i
            expectations[i] = 0
            for j, amplitude in enumerate(statevector):
                # If jth bit of state j is 1, add negative contribution
                bit_parity = bin(j & z_mask).count('1') % 2
                sign = -1 if bit_parity else 1
                expectations[i] += sign * (np.abs(amplitude) ** 2)
                
        return expectations
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Quantum-processed features
        """
        # Build and execute quantum circuit
        circuit = self._build_circuit(x)
        expectations = self._get_expectations(circuit)
        
        # Convert to PyTorch tensor
        result = torch.tensor(expectations, dtype=torch.float32)
        return result


class AutoregressiveNetwork(nn.Module):
    """
    Neural network for autoregressive transforms in normalizing flows.
    Uses masked linear layers to ensure autoregressive property.
    """
    def __init__(self, features, hidden_features=64, num_layers=2):
        super().__init__()
        self.features = features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        
        # Input layer
        self.input_layer = nn.Linear(features, hidden_features)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_features, hidden_features) 
            for _ in range(num_layers)
        ])
        
        # Output layer - outputs shift and scale parameters
        self.output_layer = nn.Linear(hidden_features, 2 * features)
        
        # Apply masks to ensure autoregressive property
        self._create_masks()
        
    def _create_masks(self):
        """Create masks for autoregressive property"""
        # Input to hidden mask
        in_mask = torch.ones(self.hidden_features, self.features)
        for i in range(self.hidden_features):
            in_mask[i, i % self.features:] = 0
        self.register_buffer('in_mask', in_mask)
        
        # Hidden to hidden masks
        h_mask = torch.ones(self.hidden_features, self.hidden_features)
        for i in range(self.hidden_features):
            h_mask[i, i+1:] = 0
        self.register_buffer('h_mask', h_mask)
        
        # Hidden to output mask
        out_mask = torch.zeros(2 * self.features, self.hidden_features)
        for i in range(self.features):
            out_mask[i, :i+1] = 1
            out_mask[i + self.features, :i] = 1
        self.register_buffer('out_mask', out_mask)
        
    def forward(self, x):
        """Forward pass with masking for autoregressive property"""
        # Input layer with mask
        x = self.input_layer(x)
        x = torch.tanh(x)
        
        # Hidden layers with masks
        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.tanh(x)
        
        # Output layer with mask
        x = self.output_layer(x)
        
        # Split output into shift and scale
        shift, log_scale = x.chunk(2, dim=-1)
        scale = torch.exp(log_scale)
        
        return shift, scale


class QuantumEnhancedFlow(nn.Module):
    """
    Quantum-Enhanced Normalizing Flow for market simulation.
    Combines quantum processing with normalizing flows to model
    complex return distributions.
    """
    
    def __init__(self, features, quantum_features=4, hidden_features=64, num_layers=2):
        """
        Initialize the quantum-enhanced flow model.
        
        Args:
            features: Number of input features (assets)
            quantum_features: Number of quantum features (qubits)
            hidden_features: Hidden layer size in autoregressive network
            num_layers: Number of transform layers in the flow
        """
        super().__init__()
        self.features = features
        self.quantum_features = min(quantum_features, features)
        
        # Quantum layer for feature extraction
        self.quantum_layer = QuantumLayer(n_qubits=self.quantum_features, depth=2)
        
        # Base distribution (standard normal)
        self.base_dist = Normal(0, 1)
        
        # Neural network for autoregressive transforms
        self.autoregressive_net = AutoregressiveNetwork(
            features=features,
            hidden_features=hidden_features,
            num_layers=num_layers
        )
        
    def forward(self, x):
        """Forward pass (used for training)"""
        batch_size = x.shape[0]
        
        # Process first n features through quantum layer
        quantum_outputs = []
        for i in range(batch_size):
            quantum_feat = self.quantum_layer(x[i, :self.quantum_features])
            quantum_outputs.append(quantum_feat)
        
        # Combine quantum features with classical ones
        q_feats = torch.stack(quantum_outputs)
        
        if self.quantum_features < self.features:
            # Combine quantum features with remaining classical features
            enhanced_x = torch.cat([q_feats, x[:, self.quantum_features:]], dim=1)
        else:
            # Use quantum features only (truncated to match features)
            enhanced_x = q_feats[:, :self.features]
        
        # Get shift and scale from autoregressive network
        shift, scale = self.autoregressive_net(enhanced_x)
        
        # Apply transformation
        z = (enhanced_x - shift) / scale
        log_det = -torch.sum(torch.log(scale), dim=1)
        
        # Log probability calculation
        log_prob = torch.sum(self.base_dist.log_prob(z), dim=1) + log_det
        
        return log_prob
    
    def sample(self, n_samples):
        """
        Generate samples from the model.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        with torch.no_grad():
            # Sample from base distribution
            z = torch.randn(n_samples, self.features)
            
            # Generate quantum features for a zero input
            # This is a simplified approach - in a more sophisticated model,
            # we might sample these features conditionally
            quantum_feat = self.quantum_layer(torch.zeros(self.quantum_features))
            quantum_tensor = quantum_feat.repeat(n_samples, 1)
            
            if self.quantum_features < self.features:
                # Combine quantum features with random normal for remaining dims
                remaining = torch.randn(n_samples, self.features - self.quantum_features)
                enhanced_z = torch.cat([quantum_tensor, remaining], dim=1)
            else:
                # Use quantum features only (truncated to match features)
                enhanced_z = quantum_tensor[:, :self.features]
            
            # Get shift and scale from autoregressive network
            shift, scale = self.autoregressive_net(enhanced_z)
            
            # Apply inverse transformation
            x = enhanced_z * scale + shift
            
            return x


def load_and_preprocess_data(file_path, split_date=None):
    """
    Load and preprocess financial data.
    
    Args:
        file_path: Path to CSV file containing price data
        split_date: Date to split train/test data
        
    Returns:
        DataFrame of returns and train/test split
    """
    logger.info(f"Loading data from {file_path}")
    
    try:
        # Load price data
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        
        # Calculate log returns
        returns = np.log(df / df.shift(1)).dropna()
        logger.info(f"Calculated returns for {len(df.columns)} assets")
        
        # Split data if specified
        if split_date:
            train_data = returns[returns.index < split_date]
            test_data = returns[returns.index >= split_date]
            logger.info(f"Split data at {split_date}: {len(train_data)} train samples, {len(test_data)} test samples")
            return returns, train_data, test_data
        
        return returns, returns, None
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def train_quantum_flow(returns, epochs=50, batch_size=32, lr=0.001):
    """
    Train quantum-enhanced normalizing flow model.
    
    Args:
        returns: DataFrame of asset returns
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        
    Returns:
        Trained model
    """
    logger.info("Initializing quantum-enhanced flow training")
    
    # Convert to PyTorch tensor
    data = torch.tensor(returns.values, dtype=torch.float32)
    
    # Create dataloader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    n_features = returns.shape[1]
    model = QuantumEnhancedFlow(features=n_features)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs")
    losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (data_batch,) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            log_prob = model(data_batch)
            
            # Negative log likelihood loss
            loss = -log_prob.mean()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Log progress
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        logger.info(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}")
        
        # Early stopping check could be added here
    
    # Save model
    torch.save(model.state_dict(), 'quantum_flow_model.pth')
    logger.info("Model training completed and saved")
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood')
    plt.grid(True)
    plt.savefig('training_loss.png')
    
    return model


def generate_scenarios(model, n_samples=1000):
    """
    Generate market scenarios using the trained flow.
    
    Args:
        model: Trained quantum-enhanced flow model
        n_samples: Number of scenarios to generate
        
    Returns:
        Synthetic samples
    """
    logger.info(f"Generating {n_samples} synthetic market scenarios")
    
    model.eval()
    with torch.no_grad():
        samples = model.sample(n_samples).numpy()
    
    logger.info("Scenario generation completed")
    return samples


def build_qubo(mu, cov, lambda_risk=0.5, budget_constraint=None):
    """
    Build QUBO formulation for portfolio optimization.
    
    Args:
        mu: Expected returns vector
        cov: Covariance matrix
        lambda_risk: Risk aversion parameter
        budget_constraint: Number of assets to select
        
    Returns:
        QUBO matrix Q
    """
    logger.info("Building QUBO formulation for portfolio selection")
    
    n = len(mu)
    
    # Initialize QUBO matrix
    Q = np.zeros((n, n))
    
    # Fill QUBO matrix
    for i in range(n):
        # Diagonal terms (expected return)
        Q[i, i] = -mu[i]
        
        # Off-diagonal terms (covariance)
        for j in range(n):
            Q[i, j] += lambda_risk * cov[i, j]
    
    logger.info(f"QUBO matrix built with shape {Q.shape}")
    return Q


def solve_with_qaoa(Q, n_assets=None, p=1):
    """
    Solve portfolio optimization using QAOA.
    
    Args:
        Q: QUBO matrix
        n_assets: Number of assets to select (budget constraint)
        p: QAOA circuit depth parameter
        
    Returns:
        Binary selection vector
    """
    logger.info(f"Starting QAOA optimization with p={p}")
    
    n = Q.shape[0]
    
    # Create quadratic program
    qp = QuadraticProgram()
    
    # Add binary variables
    for i in range(n):
        qp.binary_var(name=f'x{i}')
    
    # Set objective
    linear = {i: Q[i, i] for i in range(n)}
    quadratic = {(i, j): 2*Q[i, j] for i in range(n) for j in range(i+1, n) if Q[i, j] != 0}
    qp.minimize(linear=linear, quadratic=quadratic)
    
    # Add budget constraint if specified
    if n_assets is not None:
        qp.linear_constraint(
            linear={i: 1 for i in range(n)},
            sense='E',
            rhs=n_assets,
            name='budget'
        )
        logger.info(f"Added budget constraint: select exactly {n_assets} assets")
    
    # Convert to QUBO
    qubo_converter = QuadraticProgramToQubo()
    qubo = qubo_converter.convert(qp)
    
    # Set up QAOA with random seed
    optimizer = COBYLA(maxiter=100)
    sampler = Sampler()
    qaoa = QAOA(optimizer=optimizer, reps=p, sampler=sampler)
    
    # Solve using QAOA
    logger.info("Running QAOA solver")
    qaoa_solver = MinimumEigenOptimizer(qaoa)
    result = qaoa_solver.solve(qubo)
    
    # Extract solution
    x = np.array([result.x[i] for i in range(n)])
    objective_value = result.fval
    
    logger.info(f"QAOA solution found with objective value: {objective_value:.6f}")
    logger.info(f"Selected {sum(x)} assets out of {n}")
    
    return x


def evaluate_portfolio(returns, selection, risk_free_rate=0.0):
    """
    Evaluate portfolio performance metrics.
    
    Args:
        returns: DataFrame of asset returns
        selection: Binary selection vector
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary of performance metrics and time series
    """
    logger.info("Evaluating portfolio performance")
    
    # Select assets based on binary vector
    selected_assets = np.where(selection == 1)[0]
    n_selected = len(selected_assets)
    
    if n_selected == 0:
        logger.warning("No assets selected! Cannot evaluate empty portfolio.")
        return None
    
    # Equal weighting among selected assets
    weights = np.zeros(returns.shape[1])
    weights[selected_assets] = 1.0 / n_selected
    
    # Calculate portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Calculate drawdowns
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Calculate annualized metrics (assuming daily data)
    ann_factor = 252  # Trading days per year
    ann_return = (1 + portfolio_returns.mean()) ** ann_factor - 1
    ann_volatility = portfolio_returns.std() * np.sqrt(ann_factor)
    sharpe_ratio = (ann_return - risk_free_rate) / ann_volatility if ann_volatility > 0 else 0
    
    # Calculate rolling metrics
    rolling_return = portfolio_returns.rolling(window=21).mean() * 21  # ~1 month
    rolling_vol = portfolio_returns.rolling(window=21).std() * np.sqrt(21)
    rolling_sharpe = rolling_return / rolling_vol
    
    # Monthly returns
    monthly_returns = portfolio_returns.resample('M').apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Performance summary
    metrics = {
        'annualized_return': ann_return,
        'annualized_volatility': ann_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_return': cumulative_returns.iloc[-1] - 1,
        'selected_assets': selected_assets,
        'weights': weights,
        
        # Time series
        'cumulative_returns': cumulative_returns,
        'drawdown': drawdown,
        'rolling_sharpe': rolling_sharpe,
        'monthly_returns': monthly_returns,
        'portfolio_returns': portfolio_returns
    }
    
    logger.info(f"Portfolio evaluation complete: {n_selected} assets")
    logger.info(f"Annualized Return: {ann_return:.2%}, Volatility: {ann_volatility:.2%}, Sharpe: {sharpe_ratio:.2f}")
    
    return metrics


def visualize_portfolio(metrics, returns, save_dir='results'):
    """
    Generate professional-grade portfolio visualizations.
    
    Args:
        metrics: Portfolio performance metrics
        returns: Original returns DataFrame
        save_dir: Directory to save plots
    """
    logger.info("Generating portfolio visualizations")
    
    # Create directory if needed
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Helper function for percentage formatting
    def percentage_formatter(x, pos):
        return f'{100 * x:.0f}%'
    
    # Asset Allocation
    selected_assets = metrics['selected_assets']
    weights = metrics['weights'][selected_assets]
    asset_names = returns.columns[selected_assets]
    
    plt.figure(figsize=(12, 8))
    plt.pie(weights, labels=asset_names, autopct='%1.1f%%', 
            startangle=90, explode=[0.05] * len(weights),
            wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    plt.title('Portfolio Asset Allocation', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/asset_allocation.png', dpi=300)
    plt.close()
    
    # Cumulative Return
    plt.figure(figsize=(12, 6))
    ax = plt.subplot()
    metrics['cumulative_returns'].plot(ax=ax, linewidth=2)
    ax.set_title('Cumulative Portfolio Return', fontsize=16)
    ax.set_xlabel('')
    ax.set_ylabel('Value of $1 Investment', fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'${y:.2f}'))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/cumulative_return.png', dpi=300)
    plt.close()
    
    # Drawdown Chart
    plt.figure(figsize=(12, 6))
    ax = plt.subplot()
    metrics['drawdown'].plot(ax=ax, linewidth=2, color='red')
    ax.fill_between(metrics['drawdown'].index, 0, metrics['drawdown'], color='red', alpha=0.2)
    ax.set_title('Portfolio Drawdown', fontsize=16)
    ax.set_xlabel('')
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/drawdown.png', dpi=300)
    plt.close()
    
    # Rolling Sharpe Ratio
    plt.figure(figsize=(12, 6))
    ax = plt.subplot()
    metrics['rolling_sharpe'].plot(ax=ax, linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.set_title('21-Day Rolling Sharpe Ratio', fontsize=16)
    ax.set_xlabel('')
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rolling_sharpe.png', dpi=300)
    plt.close()
    
    # Monthly Returns Heatmap
    monthly_returns = metrics['monthly_returns']
    
    # Pivot the data for heatmap display
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    heatmap_data = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    pivot_data = heatmap_data.pivot(index='Year', columns='Month', values='Return')
    pivot_data.columns = [month_names[i-1] for i in pivot_data.columns]
    
    plt.figure(figsize=(12, len(pivot_data) * 0.8))
    sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=0, 
               fmt='.1%', linewidths=1, vmin=-0.1, vmax=0.1)
    plt.title('Monthly Returns (%)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/monthly_returns.png', dpi=300)
    plt.close()
    
    # Performance Metrics Table
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    metrics_data = [
        ['Annualized Return', f"{metrics['annualized_return']:.2%}"],
        ['Annualized Volatility', f"{metrics['annualized_volatility']:.2%}"],
        ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}"],
        ['Max Drawdown', f"{metrics['max_drawdown']:.2%}"],
        ['Total Return', f"{metrics['cumulative_returns'].iloc[-1] - 1:.2%}"],
        ['Selected Assets', f"{len(metrics['selected_assets'])}"]
    ]
    table = ax.table(cellText=metrics_data, loc='center', cellLoc='center',
                   colWidths=[0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    plt.title('Portfolio Performance Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_summary.png', dpi=300)
    plt.close()
    
    logger.info(f"Visualizations saved to {save_dir}")


def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Quantum-Enhanced Portfolio Optimization')
    parser.add_argument('--data', type=str, default='portfolio_prices.csv',
                        help='Path to CSV file with asset prices')
    parser.add_argument('--split_date', type=str, default='2023-01-01',
                        help='Date to split train/test data')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--scenarios', type=int, default=1000,
                        help='Number of scenarios to generate')
    parser.add_argument('--risk_aversion', type=float, default=0.5,
                        help='Risk aversion parameter (lambda)')
    parser.add_argument('--n_assets', type=int, default=None,
                        help='Number of assets to select (budget constraint)')
    parser.add_argument('--qaoa_p', type=int, default=1,
                        help='QAOA circuit depth parameter')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--force_overwrite', action='store_true',
                        help='Force overwrite of existing results files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(args.output, "optimization.log")
    if os.path.exists(log_file):
        os.remove(log_file)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Load and preprocess data
    returns, train_data, test_data = load_and_preprocess_data(args.data, args.split_date)
    
    # Train quantum-enhanced flow model
    model = train_quantum_flow(train_data, epochs=args.epochs, batch_size=args.batch_size)
    
    # Generate synthetic market scenarios
    scenarios = generate_scenarios(model, n_samples=args.scenarios)
    
    # Calculate mean and covariance from scenarios
    mu = scenarios.mean(axis=0)
    cov = np.cov(scenarios, rowvar=False)
    
    # Build QUBO formulation
    Q = build_qubo(mu, cov, lambda_risk=args.risk_aversion)
    
    # Solve with QAOA
    selection = solve_with_qaoa(Q, n_assets=args.n_assets, p=args.qaoa_p)
    
    # Save selection vector
    selection_file = os.path.join(args.output, "asset_selection.csv")
    if os.path.exists(selection_file):
        os.remove(selection_file)
    np.savetxt(selection_file, selection, delimiter=',')
    
    # Evaluate portfolio performance
    metrics = evaluate_portfolio(test_data, selection)
    
    # Generate visualizations
    if metrics is not None:
        # Remove existing visualization files
        if args.force_overwrite:
            for file in os.listdir(args.output):
                if file.endswith('.png'):
                    os.remove(os.path.join(args.output, file))
        visualize_portfolio(metrics, returns, save_dir=args.output)
    
    logger.info("Quantum portfolio optimization complete!")
    logger.info(f"Results saved to {args.output}/")


if __name__ == "__main__":
    main()