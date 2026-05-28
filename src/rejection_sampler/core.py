from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy import integrate, optimize

from rejection_sampler.exceptions import (
    BoundsNotProvidedError,
    PDFValidationError,
    ScipyFailedError,
    SympyFailedError,
)

PDF = Callable[[float], float] | sp.Expr


@dataclass(frozen=True, slots=True)
class _PDF:
    func: PDF
    support: tuple[float, float]
    is_symbolic: bool

    def __init__(self, pdf: PDF, support: tuple[float, float]):
        object.__setattr__(self, "func", pdf)
        object.__setattr__(self, "support", support)

        if isinstance(pdf, sp.Expr):
            is_symbolic = True
        elif callable(pdf):
            is_symbolic = False
        else:
            raise PDFValidationError(f"Invalid input type of {pdf}.")

        object.__setattr__(self, "is_symbolic", is_symbolic)


def find_optimal_M(
    target_pdf: PDF,
    target_support: tuple[float, float],
    proposal_pdf: PDF,
    proposal_support: tuple[float, float],
    error: float = 1e-6,
    bounds: None | tuple[float, float] = None,
) -> float:
    """
    Validate a rejection-sampling setup and compute the optimal rejection
    constant ``M`` such that

        target_pdf(x) <= M * proposal_pdf(x)

    for all ``x`` in the target support.

    Both symbolic (`sympy.Expr`) and numerical (`Callable`) probability
    density functions are supported. The function first validates the target
    and proposal distributions, then attempts to compute the supremum of the
    density ratio

        target_pdf(x) / proposal_pdf(x)

    using symbolic optimization via SymPy. If symbolic methods fail, numerical
    optimization via SciPy is used as a fallback.

    Parameters
    ----------
    target_pdf : PDF
        Target probability density function. May be either:
        - a callable ``f(x) -> float``
        - a symbolic ``sympy.Expr``

    target_support : tuple[float, float]
        Support interval of the target PDF.
        Use ``float("inf")`` or ``-float("inf")`` for infinite bounds.

    proposal_pdf : PDF
        Proposal probability density function used for rejection sampling.

    proposal_support : tuple[float, float]
        Support interval of the proposal PDF.

    error : float, default=1e-6
        Numerical tolerance used when checking:
        - PDF non-negativity
        - normalization
        - floating-point comparisons

    bounds : tuple[float, float] | None, default=None
        Possible area where max of f/g occurs.
        Provide this for complicated functions with infinite support for better accuracy.

    Returns
    -------
    float
        The optimal rejection constant ``M``.

    Examples
    --------
    Sympy example:

    >>> x = sp.Symbol("x", real=True)
    >>> target = sp.exp(-x)
    >>> proposal = sp.exp(-x / 2) / 2
    >>> find_optimal_M(
    ...     target,
    ...     (0, float("inf")),
    ...     proposal,
    ...     (0, float("inf")),
    ... )

    Callable functions example:

    >>> def target(x): return np.exp(-x)
    >>> def proposal(x): return 0.5 * np.exp(-x / 2)
    >>> find_optimal_M(
    ...     target,
    ...     (0, float("inf")),
    ...     proposal,
    ...     (0, float("inf")),
    ...     bounds=(0, 10),
    ... )

    Note
    ---------
    Use numpy for math expressions. Avoid using built-in math package.

    """

    x = sp.Symbol("x", real=True)

    def _as_callable(pdf: _PDF) -> Callable[[float], float]:
        # turns sp.Expr into a callable function for numerical approximation
        if pdf.is_symbolic:
            return sp.lambdify(x, pdf.func, modules=["numpy"])
        return pdf.func

    def _is_valid_pdf(pdf: _PDF) -> bool:
        f = _as_callable(pdf)
        a, b = pdf.support

        # check non-negativity using Sympy
        if pdf.is_symbolic:
            try:
                interval = sp.Interval(a, b)

                # handle piecewise functions separately
                if isinstance(pdf.func, sp.Piecewise):
                    for expr, cond in pdf.func.args:
                        if cond is True:
                            active_set = interval
                        else:
                            active_set = sp.solveset(cond, x, domain=interval)

                        if active_set is sp.EmptySet:
                            continue

                        # Constant negative branch over any non-empty active set.
                        if expr.is_number and float(expr) < -error:
                            return False

                        # Try symbolic minimum on the active set.
                        try:
                            min_value = sp.minimum(expr, x, domain=active_set)
                            if min_value.is_real and float(min_value) < -error:
                                return False
                        except Exception:
                            pass
                else:
                    min_value = sp.minimum(pdf.func, x, domain=interval)
                    if min_value.is_real and float(min_value) < -error:
                        return False

            except Exception:
                # Fall back to numerical checks below.
                pass

        # Check non-negativity numerically on finite support
        if np.isfinite(a) and np.isfinite(b):
            try:
                result = optimize.minimize_scalar(
                    lambda t: float(f(t)),
                    bounds=(a, b),
                    method="bounded",
                )
                if not result.success or result.fun < -error:
                    return False
            except Exception:
                return False

        # Check normalization
        try:
            integral, _ = integrate.quad(
                lambda t: float(f(t)),
                a,
                b,
                limit=200,
            )
        except Exception:
            return False

        return abs(float(integral) - 1.0) < error

    def _support_check(
        proposal_support: tuple[float, float],
        target_support: tuple[float, float],
    ) -> bool:
        p_left, p_right = proposal_support
        t_left, t_right = target_support

        return p_left <= t_left and p_right >= t_right

    def _check_boundaries(target: _PDF, proposal: _PDF) -> bool:

        a, b = target.support

        try:
            if target.is_symbolic and proposal.is_symbolic:
                ratio = sp.simplify(target.func / proposal.func)

                checks = [
                    (a, "+"),
                    (b, "-"),
                ]

                for boundary, direction in checks:
                    if boundary == -float("inf"):
                        value = sp.limit(ratio, x, -sp.oo)
                    elif boundary == float("inf"):
                        value = sp.limit(ratio, x, sp.oo)
                    else:
                        value = sp.limit(ratio, x, boundary, dir=direction)

                    if value in (sp.oo, -sp.oo, sp.zoo, sp.nan):
                        return False

                    if value.is_real is False:
                        return False

                return True
        except Exception:
            pass

        f = _as_callable(target)
        g = _as_callable(proposal)

        def neg_ratio(t: float) -> float:
            target_value = float(f(t))
            proposal_value = float(g(t))

            if proposal_value <= 0 or target_value <= 0:
                return float("inf")

            return -target_value / proposal_value

        # Case 1: finite boundaries
        if np.isfinite(a) and np.isfinite(b):
            result = optimize.differential_evolution(
                lambda z: neg_ratio(z[0]),
                bounds=[(a, b)],
                maxiter=1000,
            )

            return result.success
        # Case 2: infinite boundaries
        if bounds is None:
            raise BoundsNotProvidedError(
                "Please provide sufficiently large tail bounds for numerical "
                "optimization. Or try using Sympy expression for analytic optimization."
            )

        result = optimize.differential_evolution(
            lambda z: neg_ratio(z[0]),
            bounds=[bounds],
            maxiter=1000,
        )
        return result.success

    def _check_with_sympy(proposal: _PDF, target: _PDF) -> float:
        if not proposal.is_symbolic or not target.is_symbolic:
            raise TypeError
        a, b = target.support
        ratio = sp.simplify(target.func / proposal.func)
        derivative = sp.diff(ratio, x)

        critical_points = sp.solve(derivative, x)
        candidates: list[float] = []

        # evaluate ratio at all candidates to find global maximum
        for point in critical_points:
            try:
                if point.is_real and a <= float(point) <= b:
                    value = ratio.subs(x, point)
                    if value.is_real:
                        candidates.append(float(value))
            except Exception:
                continue

        for boundary, direction in ((a, "+"), (b, "-")):
            try:
                if boundary == -float("inf"):
                    value = sp.limit(ratio, x, -sp.oo)
                elif boundary == float("inf"):
                    value = sp.limit(ratio, x, sp.oo)
                else:
                    value = sp.limit(ratio, x, boundary, dir=direction)

                if value.is_real:
                    candidates.append(float(value))
            except Exception:
                continue

        candidates = [v for v in candidates if v > 0 and v != float("inf")]

        if not candidates:
            raise SympyFailedError

        return float(max(candidates))

    def _check_with_scipy(proposal: _PDF, target: _PDF) -> float:
        f = _as_callable(target)
        g = _as_callable(proposal)
        a, b = target.support

        def neg_ratio(t: float) -> float:
            target_value = float(f(t))
            proposal_value = float(g(t))

            if proposal_value <= 0 or target_value <= 0:
                return float("inf")

            return -target_value / proposal_value

        def _valid_M(value: float) -> bool:
            return value >= 1 - error and np.isfinite(value)

        # Case 1: finite target support
        if np.isfinite(a) and np.isfinite(b):
            candidates = []

            for t in (a, b):
                try:
                    value = -neg_ratio(t)
                    if _valid_M(value):
                        candidates.append(value)
                except Exception:
                    pass

            # use target support as `bounds` for better performance
            result = optimize.minimize_scalar(
                neg_ratio,
                bounds=(a, b),
                method="bounded",
            )

            if result.success:
                M = -float(result.fun)
                if _valid_M(M):
                    candidates.append(M)
                    return max(candidates)

            # Fallback for non-smooth callable PDFs.
            # handles more complicated functions
            result = optimize.differential_evolution(
                lambda z: neg_ratio(z[0]),
                bounds=[(a, b)],
                maxiter=1000,
            )

            if result.success:
                M = -float(result.fun)
                if _valid_M(M):
                    candidates.append(M)
                    return max(candidates)

            if candidates:
                return max(candidates)

            raise ScipyFailedError

        # Case 2: infinite support requires user-provided finite bounds.
        if bounds is None:
            raise BoundsNotProvidedError(
                "Please provide sufficiently large tail bounds for numerical "
                "optimization. Or try using Sympy expression for analytic optimization."
            )

        result = optimize.differential_evolution(
            lambda z: neg_ratio(z[0]),
            bounds=[bounds],
            maxiter=1000,
        )

        if not result.success:
            raise ScipyFailedError(
                "Try using Sympy expressions for analytic optimization."
            )

        M = -float(result.fun)

        if not _valid_M(M):
            raise ScipyFailedError("Return invalid M")

        return float(M)

    proposal = _PDF(proposal_pdf, proposal_support)
    target = _PDF(target_pdf, target_support)

    if not _support_check(proposal_support, target_support):
        raise PDFValidationError("Target support is not a subset of proposal support.")
    if not _is_valid_pdf(proposal):
        raise PDFValidationError("Proposal pdf is not a valid pdf.")
    if not _is_valid_pdf(target):
        raise PDFValidationError("Target pdf is not a valid pdf.")
    if not _check_boundaries(target, proposal):
        raise PDFValidationError(
            "PDF ratio at boundaries are not valid for rejection sampling."
        )

    try:
        M = _check_with_sympy(proposal, target)
    except Exception:
        M = _check_with_scipy(proposal, target)

    return M
