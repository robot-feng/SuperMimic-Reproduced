"""Small utilities for equal-budget paired score-gradient diagnostics."""


def combined_mean(fit_mean, validation_mean, fit_count, validation_count):
    if fit_count < 1 or validation_count < 1:
        raise ValueError('both batches must contain samples')
    return (fit_mean * fit_count + validation_mean * validation_count) / (fit_count + validation_count)


def crossfit_mean(real_a, real_b, zero_a, zero_b, coefficient_a, coefficient_b):
    """Apply each fold's coefficient to the other fold, for equal fold sizes."""
    return .5 * (real_a + coefficient_b * zero_a + real_b + coefficient_a * zero_b)
