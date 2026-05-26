# basic exception definitions


class RejectionSamplerError(Exception):
    """Base exception for the package."""


class PDFValidationError(RejectionSamplerError):
    """Raised when a PDF is invalid."""


class SympyFailedError(RejectionSamplerError):
    """Raised when Sympy evaluation failed."""


class ScipyFailedError(RejectionSamplerError):
    """Raised when Scipy evaluation failed."""
