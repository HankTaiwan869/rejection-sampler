import math

import pytest
import sympy as sp

from rejection_sampler.core import find_optimal_M
from rejection_sampler.exceptions import PDFValidationError


def test_find_optimal_M_simple_sympy_expression():
    x = sp.Symbol("x", real=True)

    # target: triangular increasing density on [0, 1]
    target_pdf = 2 * x

    # proposal: uniform density on [0, 1]
    proposal_pdf = sp.Integer(1)

    M = find_optimal_M(
        target_pdf=target_pdf,
        target_support=(0.0, 1.0),
        proposal_pdf=proposal_pdf,
        proposal_support=(0.0, 1.0),
    )

    assert M == pytest.approx(2.0)


def test_find_optimal_M_simple_callable_functions():
    # target: symmetric triangular density on [0, 1]
    def target_pdf(x):
        return 4 * x if x <= 0.5 else 4 * (1 - x)

    # proposal: uniform density on [0, 1]
    def proposal_pdf(x):
        return 1.0

    M = find_optimal_M(
        target_pdf=target_pdf,
        target_support=(0.0, 1.0),
        proposal_pdf=proposal_pdf,
        proposal_support=(0.0, 1.0),
    )

    assert M == pytest.approx(2.0)


def test_rejects_negative_pdf():
    x = sp.Symbol("x", real=True)

    # Integrates to 1 on [0, 1], but is negative near x = 0.
    invalid_pdf = 3 * x - sp.Rational(1, 2)

    proposal_pdf = sp.Integer(1)

    with pytest.raises(PDFValidationError, match="Target pdf is not a valid pdf"):
        find_optimal_M(
            target_pdf=invalid_pdf,
            target_support=(0.0, 1.0),
            proposal_pdf=proposal_pdf,
            proposal_support=(0.0, 1.0),
        )


def test_rejects_pdf_that_does_not_integrate_to_one():
    # Nonnegative but integrates to 2 on [0, 1].
    invalid_pdf = sp.Integer(2)

    proposal_pdf = sp.Integer(1)

    with pytest.raises(PDFValidationError, match="Target pdf is not a valid pdf"):
        find_optimal_M(
            target_pdf=invalid_pdf,
            target_support=(0.0, 1.0),
            proposal_pdf=proposal_pdf,
            proposal_support=(0.0, 1.0),
        )


def test_find_optimal_M_complicated_sympy_expression():
    x = sp.Symbol("x", real=True)

    # target: Beta(3, 2), f(x) = 12x^2(1-x)
    target_pdf = 12 * x**2 * (1 - x)

    # proposal: Beta(2, 2), g(x) = 6x(1-x)
    proposal_pdf = 6 * x * (1 - x)

    # ratio = 2x, max on [0, 1] is 2.
    M = find_optimal_M(
        target_pdf=target_pdf,
        target_support=(0.0, 1.0),
        proposal_pdf=proposal_pdf,
        proposal_support=(0.0, 1.0),
    )

    assert M == pytest.approx(2.0)


def test_find_optimal_M_complicated_callable_functions():
    sqrt_2pi = math.sqrt(2 * math.pi)

    # target: standard normal
    def target_pdf(x):
        return math.exp(-0.5 * x * x) / sqrt_2pi

    # proposal: Cauchy(0, 1)
    def proposal_pdf(x):
        return 1.0 / (math.pi * (1 + x * x))

    # ratio max occurs at x = ±1:
    # M = sqrt(2pi) * exp(-1/2)
    expected_M = math.sqrt(2 * math.pi) * math.exp(-0.5)

    M = find_optimal_M(
        target_pdf=target_pdf,
        target_support=(-math.inf, math.inf),
        proposal_pdf=proposal_pdf,
        proposal_support=(-math.inf, math.inf),
        bounds=(-10.0, 10.0),
    )

    assert M == pytest.approx(expected_M, rel=1e-5)


# harder tests


def test_valid_pdf_rejects_piecewise_negative_sympy_pdf():
    x = sp.Symbol("x", real=True)

    # Integrates to 1 but is negative on part of [0, 1].
    invalid_pdf = sp.Piecewise(
        (-1, x < sp.Rational(1, 4)),
        (sp.Rational(5, 3), True),
    )

    with pytest.raises(PDFValidationError):
        find_optimal_M(
            target_pdf=invalid_pdf,
            target_support=(0, 1),
            proposal_pdf=sp.Integer(1),
            proposal_support=(0, 1),
        )


def test_valid_pdf_rejects_callable_not_normalized_on_finite_support():
    def invalid_target(x):
        return 3 * x**2  # integrates to 1 on [0,1], valid

    def invalid_proposal(x):
        return 2.0  # integrates to 2 on [0,1], invalid

    with pytest.raises(PDFValidationError, match="Proposal pdf is not a valid pdf"):
        find_optimal_M(
            target_pdf=invalid_target,
            target_support=(0, 1),
            proposal_pdf=invalid_proposal,
            proposal_support=(0, 1),
        )


def test_valid_pdf_rejects_callable_negative_but_normalized():
    def target_pdf(x):
        return 1.0

    # Integral is 1 on [0,1], but negative near x = 0.
    def bad_proposal(x):
        return 3 * x - 0.5

    with pytest.raises(PDFValidationError, match="Proposal pdf is not a valid pdf"):
        find_optimal_M(
            target_pdf=target_pdf,
            target_support=(0, 1),
            proposal_pdf=bad_proposal,
            proposal_support=(0, 1),
        )


def test_rejects_proposal_support_smaller_than_target_support():
    x = sp.Symbol("x", real=True)

    with pytest.raises(PDFValidationError, match="Target support is not a subset"):
        find_optimal_M(
            target_pdf=sp.Integer(1),
            target_support=(0, 1),
            proposal_pdf=2 * x,
            proposal_support=(0, 0.5),
        )


def test_rejects_unbounded_ratio_due_to_bad_proposal_tail_sympy():
    x = sp.Symbol("x", real=True)

    # target = Uniform(0,1)
    # proposal = Beta(2,1), g(x)=2x
    # ratio = 1/(2x), which blows up as x -> 0+
    with pytest.raises(PDFValidationError):
        find_optimal_M(
            target_pdf=sp.Integer(1),
            target_support=(0, 1),
            proposal_pdf=2 * x,
            proposal_support=(0, 1),
        )


def test_main_function_beta_vs_uniform_sympy_known_M():
    x = sp.Symbol("x", real=True)

    # Beta(3,3): f(x)=30x^2(1-x)^2
    # proposal: Uniform(0,1)
    # max occurs at x=1/2: M = 30*(1/4)*(1/4)=30/16=1.875
    M = find_optimal_M(
        target_pdf=30 * x**2 * (1 - x) ** 2,
        target_support=(0, 1),
        proposal_pdf=sp.Integer(1),
        proposal_support=(0, 1),
    )

    assert M == pytest.approx(1.875)


def test_main_function_beta_vs_beta_sympy_known_M():
    x = sp.Symbol("x", real=True)

    # target: Beta(4,2), f(x)=20x^3(1-x)
    # proposal: Beta(2,2), g(x)=6x(1-x)
    # ratio = (10/3)x^2
    # max at x=1 gives M = 10/3
    M = find_optimal_M(
        target_pdf=20 * x**3 * (1 - x),
        target_support=(0, 1),
        proposal_pdf=6 * x * (1 - x),
        proposal_support=(0, 1),
    )

    assert M == pytest.approx(10 / 3)


def test_main_function_exponential_vs_laplace_callable_known_M():
    # target: Exponential(1) on [0, inf)
    def target_pdf(x):
        return math.exp(-x) if x >= 0 else 0.0

    # proposal: Laplace(0,1), g(x)=0.5 exp(-abs(x))
    def proposal_pdf(x):
        return 0.5 * math.exp(-abs(x))

    # On x >= 0, f/g = exp(-x)/(0.5 exp(-x)) = 2
    M = find_optimal_M(
        target_pdf=target_pdf,
        target_support=(0, math.inf),
        proposal_pdf=proposal_pdf,
        proposal_support=(-math.inf, math.inf),
        bounds=(0, 20),
    )

    assert M == pytest.approx(2.0, rel=1e-5)


def test_main_function_triangular_vs_beta_callable_known_M():
    # target: triangular density on [0,1], peak 2 at x=0.5
    def target_pdf(x):
        if 0 <= x <= 0.5:
            return 4 * x
        if 0.5 < x <= 1:
            return 4 * (1 - x)
        return 0.0

    # proposal: Beta(2,2), g(x)=6x(1-x)
    def proposal_pdf(x):
        if 0 <= x <= 1:
            return 6 * x * (1 - x)
        return 0.0

    # For x <= 0.5: ratio = 4x / [6x(1-x)] = 2/[3(1-x)]
    # For x > 0.5: ratio = 4(1-x) / [6x(1-x)] = 2/(3x)
    # Max occurs at x=0.5, M = 4/3
    M = find_optimal_M(
        target_pdf=target_pdf,
        target_support=(0, 1),
        proposal_pdf=proposal_pdf,
        proposal_support=(0, 1),
    )

    assert M == pytest.approx(4 / 3, rel=1e-5)


def test_find_optimal_M_callable_boundary_maximum():
    # target: Beta(2, 1), f(x) = 2x on [0, 1]
    def target_pdf(x):
        return 2 * x if 0 <= x <= 1 else 0.0

    # proposal: Uniform(0, 1)
    def proposal_pdf(x):
        return 1.0 if 0 <= x <= 1 else 0.0

    # ratio = 2x, so the maximum is attained at the right boundary x = 1.
    M = find_optimal_M(
        target_pdf=target_pdf,
        target_support=(0, 1),
        proposal_pdf=proposal_pdf,
        proposal_support=(0, 1),
    )

    assert M == pytest.approx(2.0)


# HARDER TESTS ACTUALLY HARD


def test_find_optimal_M_oscillating_callable_pdf():
    # target: oscillating but valid density on [0, 1]
    # Integral of sin(20*pi*x) over [0,1] is 0, so this integrates to 1.
    # Since 0.5 <= f(x) <= 1.5, it is nonnegative.
    def target_pdf(x):
        if 0 <= x <= 1:
            return 1.0 + 0.5 * math.sin(20 * math.pi * x)
        return 0.0

    # proposal: Uniform(0, 1)
    def proposal_pdf(x):
        return 1.0 if 0 <= x <= 1 else 0.0

    # ratio = target_pdf / proposal_pdf = target_pdf.
    # Max occurs when sin(20*pi*x) = 1, so M = 1.5.
    M = find_optimal_M(
        target_pdf=target_pdf,
        target_support=(0, 1),
        proposal_pdf=proposal_pdf,
        proposal_support=(0, 1),
        maxiter=2000,
    )

    assert M == pytest.approx(1.5, rel=1e-4)
