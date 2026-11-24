import math
from statistics import NormalDist


def _gaussian_certificate(T, scale):
    """Exact response-domain TV diameter for isotropic Gaussian noise.

    The domain (Delta^{k-1})^T has Euclidean diameter sqrt(2T).
    Hence its Gaussian channel diameter is
    2 Phi(sqrt(2T) / (2 scale)) - 1 = erf(sqrt(T) / (2 scale)).
    A zero scale is the identity channel, whose diameter is one.
    """
    if scale == 0:
        return 1.0
    return math.erf(math.sqrt(T) / (2.0 * scale))


def _certificate(T, sigma, mediator, scale):
    if mediator == "none":
        return _gaussian_certificate(T, sigma), "exact_gaussian"
    if mediator == "gaussian":
        return _gaussian_certificate(T, math.hypot(sigma, scale)), "exact_gaussian"

    # Laplace(0, scale) has coordinate standard deviation sqrt(2) * scale.
    # Its convolution with observation noise is not a Gaussian channel.
    observation_bound = _gaussian_certificate(T, sigma)
    laplace_bound = 1.0 if scale == 0 else -math.expm1(-T / scale)
    return min(observation_bound, laplace_bound), "laplace_overlap_upper_bound"


def build_policies(T, sigma, mediators, noise_scales, privacy_budgets):
    """Declare single-batch policies using only public setup information.

    Gaussian certificates are exact domain-wide TV diameters. For iid
    Laplace coordinates of scale ell, the overlap of two one-dimensional
    shifted densities is exp(-abs(delta) / (2 ell)). For product densities,
    min(prod p_j, prod q_j) >= prod min(p_j, q_j), so their overlap is at
    least exp(-||delta||_1 / (2 ell)). Since two probability vectors have
    L1 distance at most two, ||delta||_1 <= 2T on the response domain.
    Therefore TV <= 1 - exp(-T / ell). This is an upper bound, not an
    equality. Convolution contracts TV, so the Gaussian-observation-plus-
    Laplace channel also satisfies both this bound and the observation-
    only Gaussian bound; we take their minimum.

    Privacy-budget inputs lie in (0, 1). A scale sweep sets the mediator
    scale directly; a budget sweep selects a sufficient scale before any
    model access. No surrogate estimates or retained-candidate information
    enter these policies. Each policy describes one complete kT-coordinate
    release, not repeated releases to the same auditor.
    """
    specifications = [("none", "baseline", 0, 0.0)]
    for mediator in mediators:
        specifications.extend((mediator, "scale", i, value)
                              for i, value in enumerate(noise_scales))
        specifications.extend((mediator, "budget", i, value)
                              for i, value in enumerate(privacy_budgets))

    policies = []
    observation_bound = _gaussian_certificate(T, sigma)
    for mediator, sweep, index, value in specifications:
        requested_budget = math.nan
        scale = value
        if sweep == "budget":
            requested_budget = value
            if mediator == "gaussian":
                quantile = NormalDist().inv_cdf((1.0 + value) / 2.0)
                required_total = math.sqrt(2.0 * T) / (2.0 * quantile)
                scale = math.sqrt(max(0.0, required_total ** 2 - sigma ** 2))
            else:
                scale = 0.0 if observation_bound <= value else -T / math.log1p(-value)

        b_cert, certificate_kind = _certificate(T, sigma, mediator, scale)
        policy_id = "unmediated" if sweep == "baseline" else f"{mediator}_{sweep}_{index:03d}"
        policies.append({
            "policy_id": policy_id,
            "mediator": mediator,
            "sweep": sweep,
            "sweep_value": value,
            "requested_budget": requested_budget,
            "noise_scale": scale,
            "b_cert": b_cert,
            "certificate_kind": certificate_kind,
            "audit_accuracy_ceiling": (1.0 + b_cert) / 2.0,
            "error_sum_floor": 1.0 - b_cert,
        })
    return policies


def add_mediator_noise(transcript, policy, rng):
    """Return one release; both auditor tests must reuse this same array."""
    if policy["mediator"] == "none":
        return transcript.copy()
    if policy["mediator"] == "gaussian":
        noise = rng.normal(0.0, policy["noise_scale"], transcript.shape)
    else:
        noise = rng.laplace(0.0, policy["noise_scale"], transcript.shape)
    return transcript + noise
