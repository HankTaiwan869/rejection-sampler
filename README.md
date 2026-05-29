# rejection-sampler
A small Python package for validating rejection sampling setups and computing the optimal rejection constant (M).
The package supports both callable Python functions and symbolic SymPy expressions for target and proposal PDFs.
# Installation
```bash
pip install rejection-sampler
# or 
uv add rejection-sampler
```
# Usage
Import the main function:
```python
from rejection_sampler import find_optimal_M
```
## Example 1: Callable input
```python
def target_pdf(x):
    return 2 * x if 0 <= x <= 1 else 0.0

def proposal_pdf(x):
    return 1.0 if 0 <= x <= 1 else 0.0

M = find_optimal_M(
    target_pdf=target_pdf,
    target_support=(0.0, 1.0),
    proposal_pdf=proposal_pdf,
    proposal_support=(0.0, 1.0),
)

print(M)
```
## Example 2: SymPy input
```python
import sympy as sp

x = sp.Symbol("x", real=True)

target_pdf = 2 * x
proposal_pdf = sp.Integer(1)

M = find_optimal_M(
    target_pdf=target_pdf,
    target_support=(0, 1),
    proposal_pdf=proposal_pdf,
    proposal_support=(0, 1),
)

print(M)
```
## Note
When writing mathemtatical expressions (eg. `exp`, `log`, `sqrt`, `inf`), use `numpy` instead of the built-in `math` module.

# Infinite support
For numerical inputs with infinite support, provide finite optimization bounds:
```python
import numpy as np
from rejection_sampler import find_optimal_M

def target_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)

def proposal_pdf(x):
    return 1.0 / (np.pi * (1 + x * x))

M = find_optimal_M(
    target_pdf=target_pdf,
    target_support=(-np.inf, np.inf),
    # or use (-float("inf"), float("inf")) for infinite support
    proposal_pdf=proposal_pdf,
    proposal_support=(-np.inf, np.inf),
    bounds=(-10.0, 10.0),
)

print(M)
```


# Parameters
- `target_pdf`: target probability density function, either callable or SymPy expression
- `target_support`: support of the target PDF
- `proposal_pdf`: proposal probability density function, either callable or SymPy expression
- `proposal_support`: support of the proposal PDF
- `error`: numerical tolerance for validation
- `bounds`: finite search interval for numerical optimization with infinite support

# License
This project is licensed under the MIT License.