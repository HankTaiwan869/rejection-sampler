import warnings

import numpy as np
import pytest
import sympy as sp

from rejection_sampler.core import find_optimal_M
from rejection_sampler.exceptions import PDFValidationError

# -----------------------------------------------------------------------------
# Valid SymPy cases with known analytical M
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_pdf, proposal_pdf, expected_M",
    [
        pytest.param(
            2 * sp.Symbol("x", real=True),
            sp.Integer(1),
            2.0,
            id="triangular_increasing_vs_uniform",
        ),
        pytest.param(
            12 * sp.Symbol("x", real=True) ** 2 * (1 - sp.Symbol("x", real=True)),
            6 * sp.Symbol("x", real=True) * (1 - sp.Symbol("x", real=True)),
            2.0,
            id="beta_3_2_vs_beta_2_2",
        ),
        pytest.param(
            30 * sp.Symbol("x", real=True) ** 2 * (1 - sp.Symbol("x", real=True)) ** 2,
            sp.Integer(1),
            1.875,
            id="beta_3_3_vs_uniform",
        ),
        pytest.param(
            20 * sp.Symbol("x", real=True) ** 3 * (1 - sp.Symbol("x", real=True)),
            6 * sp.Symbol("x", real=True) * (1 - sp.Symbol("x", real=True)),
            10 / 3,
            id="beta_4_2_vs_beta_2_2",
        ),
    ],
)
def test_sympy_known_optimal_M(target_pdf, proposal_pdf, expected_M):
    M = find_optimal_M(
        target_pdf=target_pdf,
        target_support=(0, 1),
        proposal_pdf=proposal_pdf,
        proposal_support=(0, 1),
    )

    assert M == pytest.approx(expected_M)


# -----------------------------------------------------------------------------
# Valid callable cases with known analytical M
# -----------------------------------------------------------------------------


def test_callable_triangular_vs_uniform_returns_known_M():
    def target_pdf(x):
        return 4 * x if x <= 0.5 else 4 * (1 - x)

    def proposal_pdf(x):
        return 1.0

    M = find_optimal_M(
        target_pdf=target_pdf,
        target_support=(0.0, 1.0),
        proposal_pdf=proposal_pdf,
        proposal_support=(0.0, 1.0),
    )

    assert M == pytest.approx(2.0)


def test_callable_normal_vs_cauchy_unbounded_support_returns_known_M():
    sqrt_2pi = np.sqrt(2 * np.pi)

    def target_pdf(x):
        return np.exp(-0.5 * x * x) / sqrt_2pi

    def proposal_pdf(x):
        return 1.0 / (np.pi * (1 + x * x))

    expected_M = np.sqrt(2 * np.pi) * np.exp(-0.5)

    M = find_optimal_M(
        target_pdf=target_pdf,
        target_support=(-np.inf, np.inf),
        proposal_pdf=proposal_pdf,
        proposal_support=(-np.inf, np.inf),
        bounds=(-10.0, 10.0),
    )

    assert M == pytest.approx(expected_M, rel=1e-5)


def test_callable_exponential_vs_laplace_returns_known_M():
    def target_pdf(x):
        return np.exp(-x) if x >= 0 else 0.0

    def proposal_pdf(x):
        return 0.5 * np.exp(-abs(x))

    M = find_optimal_M(
        target_pdf=target_pdf,
        target_support=(0, np.inf),
        proposal_pdf=proposal_pdf,
        proposal_support=(-np.inf, np.inf),
        bounds=(0, 20),
    )

    assert M == pytest.approx(2.0, rel=1e-5)


def test_callable_triangular_vs_beta_2_2_returns_known_M():
    def target_pdf(x):
        if 0 <= x <= 0.5:
            return 4 * x
        if 0.5 < x <= 1:
            return 4 * (1 - x)
        return 0.0

    def proposal_pdf(x):
        if 0 <= x <= 1:
            return 6 * x * (1 - x)
        return 0.0

    M = find_optimal_M(
        target_pdf=target_pdf,
        target_support=(0, 1),
        proposal_pdf=proposal_pdf,
        proposal_support=(0, 1),
    )

    assert M == pytest.approx(4 / 3, rel=1e-5)


def test_callable_boundary_maximum_is_found():
    def target_pdf(x):
        return 2 * x if 0 <= x <= 1 else 0.0

    def proposal_pdf(x):
        return 1.0 if 0 <= x <= 1 else 0.0

    M = find_optimal_M(
        target_pdf=target_pdf,
        target_support=(0, 1),
        proposal_pdf=proposal_pdf,
        proposal_support=(0, 1),
    )

    assert M == pytest.approx(2.0)


def test_callable_oscillating_pdf_global_maximum_is_found():
    def target_pdf(x):
        if 0 <= x <= 1:
            return 1.0 + 0.5 * np.sin(20 * np.pi * x)
        return 0.0

    def proposal_pdf(x):
        return 1.0 if 0 <= x <= 1 else 0.0

    M = find_optimal_M(
        target_pdf=target_pdf,
        target_support=(0, 1),
        proposal_pdf=proposal_pdf,
        proposal_support=(0, 1),
    )

    assert M == pytest.approx(1.5, rel=1e-4)


# -----------------------------------------------------------------------------
# Invalid PDF validation
# -----------------------------------------------------------------------------


def test_sympy_negative_target_pdf_is_rejected():
    x = sp.Symbol("x", real=True)

    invalid_target = 3 * x - sp.Rational(1, 2)

    with pytest.raises(PDFValidationError, match="Target pdf is not a valid pdf"):
        find_optimal_M(
            target_pdf=invalid_target,
            target_support=(0.0, 1.0),
            proposal_pdf=sp.Integer(1),
            proposal_support=(0.0, 1.0),
        )


def test_sympy_unnormalized_target_pdf_is_rejected():
    with pytest.raises(PDFValidationError, match="Target pdf is not a valid pdf"):
        find_optimal_M(
            target_pdf=sp.Integer(2),
            target_support=(0.0, 1.0),
            proposal_pdf=sp.Integer(1),
            proposal_support=(0.0, 1.0),
        )


def test_sympy_piecewise_negative_target_pdf_is_rejected():
    x = sp.Symbol("x", real=True)

    invalid_target = sp.Piecewise(
        (-1, x < sp.Rational(1, 4)),
        (sp.Rational(5, 3), True),
    )

    with pytest.raises(PDFValidationError):
        find_optimal_M(
            target_pdf=invalid_target,
            target_support=(0, 1),
            proposal_pdf=sp.Integer(1),
            proposal_support=(0, 1),
        )


def test_callable_unnormalized_proposal_pdf_is_rejected():
    def target_pdf(x):
        return 3 * x**2

    def invalid_proposal(x):
        return 2.0

    with pytest.raises(PDFValidationError, match="Proposal pdf is not a valid pdf"):
        find_optimal_M(
            target_pdf=target_pdf,
            target_support=(0, 1),
            proposal_pdf=invalid_proposal,
            proposal_support=(0, 1),
        )


def test_callable_negative_proposal_pdf_is_rejected():
    def target_pdf(x):
        return 1.0

    def invalid_proposal(x):
        return 3 * x - 0.5

    with pytest.raises(PDFValidationError, match="Proposal pdf is not a valid pdf"):
        find_optimal_M(
            target_pdf=target_pdf,
            target_support=(0, 1),
            proposal_pdf=invalid_proposal,
            proposal_support=(0, 1),
        )


# -----------------------------------------------------------------------------
# Support and domination failures
# -----------------------------------------------------------------------------


def test_proposal_support_smaller_than_target_support_is_rejected():
    x = sp.Symbol("x", real=True)

    with pytest.raises(PDFValidationError, match="Target support is not a subset"):
        find_optimal_M(
            target_pdf=sp.Integer(1),
            target_support=(0, 1),
            proposal_pdf=2 * x,
            proposal_support=(0, 0.5),
        )


def test_sympy_unbounded_ratio_due_to_proposal_zero_is_rejected():
    x = sp.Symbol("x", real=True)

    with pytest.raises(PDFValidationError):
        find_optimal_M(
            target_pdf=sp.Integer(1),
            target_support=(0, 1),
            proposal_pdf=2 * x,
            proposal_support=(0, 1),
        )


# -----------------------------------------------------------------------------
# Lecture note examples
# -----------------------------------------------------------------------------


def test_sympy_skew_normal_vs_cauchy_returns_finite_M():
    x = sp.Symbol("x", real=True)

    lam = 1
    phi = sp.exp(-(x**2) / 2) / sp.sqrt(2 * sp.pi)
    Phi = sp.Rational(1, 2) * (1 + sp.erf(lam * x / sp.sqrt(2)))

    # target: skew-normal with lambda = 1
    target_pdf = 2 * phi * Phi

    # proposal: Cauchy(0, 1)
    proposal_pdf = 1 / (sp.pi * (1 + x**2))

    M = find_optimal_M(
        target_pdf=target_pdf,
        target_support=(-np.inf, np.inf),
        proposal_pdf=proposal_pdf,
        proposal_support=(-np.inf, np.inf),
        bounds=(-10, 10),
    )

    assert M > 0
    assert np.isfinite(M)


def test_sympy_inverse_gaussian_vs_exponential_returns_finite_M():
    x = sp.Symbol("x", positive=True)

    theta = 1
    lam = 1

    target_pdf = sp.sqrt(lam / (2 * sp.pi * x**3)) * sp.exp(
        -lam * (x - theta) ** 2 / (2 * theta**2 * x)
    )

    proposal_pdf = sp.Rational(1, 2) * sp.exp(-x / 2)

    # suppress divide by zero warning raised by sympy
    # the warning is irrelevant to the functionality
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        M = find_optimal_M(
            target_pdf=target_pdf,
            target_support=(0, np.inf),
            proposal_pdf=proposal_pdf,
            proposal_support=(0, np.inf),
            bounds=(1e-6, 30),
        )

    assert M > 0
    assert np.isfinite(M)


def test_scipy_generalized_error_vs_cauchy_returns_finite_M():
    from scipy.stats import gennorm

    nu = 1.5
    scale = 1.0

    # target: generalized error / generalized normal distribution
    def target_pdf(x):
        return gennorm.pdf(x, beta=nu, scale=scale)

    # proposal: Cauchy(0, 1)
    def proposal_pdf(x):
        return 1.0 / (np.pi * (1 + x * x))

    M = find_optimal_M(
        target_pdf=target_pdf,
        target_support=(-np.inf, np.inf),
        proposal_pdf=proposal_pdf,
        proposal_support=(-np.inf, np.inf),
        bounds=(-20, 20),
    )

    assert M > 0
    assert np.isfinite(M)
