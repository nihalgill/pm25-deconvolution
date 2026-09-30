"""
==============================================================================
PHASE 1: TIME-DOMAIN DECONVOLUTION ENGINE
==============================================================================
Mobile Particulate Monitoring Platform, Quantitative Modeling Subsystem.

The engine recovers the latent ambient PM2.5 concentration X from
convolutionally smeared SPS30 telemetry Y via Tikhonov regularization, with
the regularization scalar selected automatically by Generalized
Cross-Validation. It operates in two execution modes:

    SIMULATION : synthetic ground-truth pipeline (digital twin
                 verification).
    PRODUCTION : ingests airborne survey CSV telemetry, validates sampling
                 uniformity, splits the flight into survey legs, gates each
                 leg on its own velocity stability, deconvolves each
                 segment independently, and emits a spatially bound cleaned
                 CSV ready for the Phase 2 Ordinary Kriging handoff.

Usage:
    python pm25_deconvolution.py --mode simulation
    python pm25_deconvolution.py --mode production --input flight_log.csv
    python pm25_deconvolution.py --help

Dependencies: numpy, scipy.optimize, matplotlib (no pandas; standard csv).

Author: Nihal Gill
==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


# =============================================================================
# CONFIGURATION
# =============================================================================
RNG_SEED: int | None = 42
# Set RNG_SEED = None to draw a fresh random dataset on every run, which
# allows Monte Carlo validation across many simulated flights.

# Temporal discretization (SIMULATION defaults; PRODUCTION derives dt from
# the CSV timestamps)
DT: float = 0.5                           # sampling interval [s]
T_TOTAL: float = 300.0                    # total survey duration [s]
N: int = int(T_TOTAL / DT)                # discrete sample count -> 600

# Sensor physics. TAU is a working estimate, not yet measured on this unit.
TAU: float = 2.0                          # SPS30 clearing constant [s]

# Platform kinematics. The cruise speed is a provisional airframe team
# estimate (8 to 13 m/s range), pending airframe finalization.
NOMINAL_CRUISE_SPEED: float = 10.0        # survey cruise speed [m/s]
DEFAULT_HEADING: np.ndarray = np.array([1.0, 0.0, 0.0])  # +X unit vector

# Validation gates. HEADING_BASELINE_SECONDS is the time span of GPS fixes
# used for each heading estimate; a longer baseline keeps normal GPS jitter
# from splitting legs. GAP_MAX_SECONDS = TAU/2 is converted into samples at
# ingest using the measured dt, so it holds at any logging rate.
DT_CV_MAX: float = 0.01                   # max coeff-of-variation of dt
VELOCITY_COVERAGE_MIN: float = 0.50       # hard gate: ≥50% deconvolvable
HEADING_TOLERANCE_DEG: float = 10.0       # max heading spread within a leg
HEADING_BASELINE_SECONDS: float = 6.0     # GPS time span per heading [s]
LEG_VELOCITY_TOLERANCE: float = 0.15      # ±15% of the leg's median speed
NAN_RATE_MAX: float = 0.05                # max acceptable NaN rate in PM2.5
GAP_MAX_SECONDS: float = 1.0              # max interpolable dropout [s]

# Default output locations
DEFAULT_OUTPUT_PATH: str = "cleaned_survey_data.csv"
DEFAULT_FIGURE_PATH: str = "docs/phase1_dashboard.png"

# Latent ground-truth signal (SIMULATION)
BASELINE: float = 4.0
EVENT_A = dict(mu=90.0, amp=55.0 - BASELINE, sigma=4.0)
EVENT_B = dict(mu=210.0, amp=85.0 - BASELINE, sigma=6.0)

# Observation corruption (SIMULATION)
NOISE_SIGMA: float = 1.2
DRIFT_AMP: float = 2.0
DRIFT_PERIOD: float = 60.0
DRIFT_PHASE: float = 0.0


# =============================================================================
# RESULT DATACLASSES
# =============================================================================
@dataclass
class DeconvolutionResult:
    """Complete artifact of a single Tikhonov deconvolution pass."""
    x_recon: np.ndarray        # (n,) reconstructed latent signal
    alpha_gcv: float           # GCV-selected regularization scalar
    alpha_lcurve: float        # L-curve-corner regularization scalar
    alphas: np.ndarray         # (m,) swept alpha grid
    rho: np.ndarray            # (m,) residual norms ||A X - Y||_2 over sweep
    eta: np.ndarray            # (m,) solution seminorms ||L X||_2 over sweep
    gcv_curve: np.ndarray      # (m,) GCV functional values over sweep
    idx_corner: int            # index of L-curve corner within alphas
    bound_note: str | None = None  # set if GCV hit a search-range bound


@dataclass
class SurveyData:
    """Parsed, validated, and uniformly sampled airborne survey telemetry."""
    t: np.ndarray              # (n,) seconds since survey start
    lat: np.ndarray            # (n,) latitude [deg]
    lon: np.ndarray            # (n,) longitude [deg]
    alt: np.ndarray            # (n,) altitude [m]
    raw_pm25: np.ndarray       # (n,) raw PM2.5 [ug/m^3], dropouts filled
    ground_speed: np.ndarray   # (n,) instantaneous ground speed [m/s]
    dt: float                  # uniform sampling interval [s]
    pm25_was_interpolated: np.ndarray  # (n,) True where raw_pm25 was filled


# =============================================================================
# OPERATION 1: LATENT GROUND TRUTH GENERATION (SIMULATION)
# =============================================================================
def generate_ground_truth(
    n: int = N,
    dt: float = DT,
    baseline: float = BASELINE,
    event_a: dict = EVENT_A,
    event_b: dict = EVENT_B,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the latent ambient PM2.5 concentration vector X(t).

    The signal is a constant ambient baseline with two superimposed Gaussian
    concentration events modeling localized plume traversals (e.g. crossing
    over a point emission source or an elevated urban pollution corridor).
    Amplitudes are specified as peak heights above baseline so resulting
    maxima match the specification exactly (55 and 85 ug/m^3).
    """
    t = np.arange(n) * dt

    def gauss(mu: float, amp: float, sigma: float) -> np.ndarray:
        return amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)

    x = baseline + gauss(**event_a) + gauss(**event_b)
    return t, x


# =============================================================================
# OPERATION 2: TOEPLITZ CONVOLUTION OPERATOR
# =============================================================================
def build_convolution_matrix(
    n: int,
    dt: float,
    tau: float = TAU,
) -> np.ndarray:
    """
    Construct the (n × n) lower-triangular Toeplitz convolution matrix A
    representing the sensor's causal exponential clearing response

            h(t) = exp(-t / tau)        for t >= 0.

    Each row is normalized to sum to unity so that a constant input vector
    maps exactly to itself (strict DC mass conservation). This induces a
    physically correct transient regime over the first ~4*tau/dt samples
    corresponding to sensor energization at t = 0.
    """
    kernel = np.exp(-np.arange(n) * dt / tau)
    i_idx, j_idx = np.indices((n, n))
    diff = i_idx - j_idx
    A = np.where(diff >= 0, kernel[np.clip(diff, 0, n - 1)], 0.0)
    A /= A.sum(axis=1, keepdims=True)
    return A


# =============================================================================
# OPERATION 3: OBSERVATION SIMULATION (SIMULATION mode)
# =============================================================================
def simulate_observation(
    A: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    noise_sigma: float = NOISE_SIGMA,
    drift_amp: float = DRIFT_AMP,
    drift_period: float = DRIFT_PERIOD,
    drift_phase: float = DRIFT_PHASE,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the corrupted telemetry vector Y from latent X:

        Y = A @ X  +  eps  +  d(t),

    where
        eps  ~ N(0, noise_sigma^2 I),
        d(t) = drift_amp * sin(2π t / drift_period + φ).
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    y_exact = A @ x
    noise = rng.normal(0.0, noise_sigma, size=x.shape)
    drift = drift_amp * np.sin(2.0 * np.pi * t / drift_period + drift_phase)
    return y_exact, drift, y_exact + noise + drift


# =============================================================================
# OPERATION 4: TIKHONOV INVERSION WITH AUTOMATED ALPHA SELECTION
# =============================================================================
def first_difference_operator(n: int) -> np.ndarray:
    """
    Build the (n-1, n) discrete first-derivative difference operator L:

            L[i, i] = -1,  L[i, i+1] = +1.

    No periodic or reflective boundary assumption. Acts as smoothness penalty
    ||L X||_2^2 = sum_i (X_{i+1} - X_i)^2.
    """
    L = np.zeros((n - 1, n))
    idx = np.arange(n - 1)
    L[idx, idx] = -1.0
    L[idx, idx + 1] = 1.0
    return L


def solve_tikhonov(
    A: np.ndarray,
    y: np.ndarray,
    L: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Solve the regularized normal equations

            (A^T A + alpha * L^T L) X = A^T y

    via direct LU-backed solve. Stable for alpha > 0.
    """
    AtA = A.T @ A
    Aty = A.T @ y
    LtL = L.T @ L
    return np.linalg.solve(AtA + alpha * LtL, Aty)


def _gcv_functional(
    log_alpha: float,
    A: np.ndarray,
    y: np.ndarray,
    AtA: np.ndarray,
    Aty: np.ndarray,
    LtL: np.ndarray,
) -> float:
    """
    Generalized Cross-Validation score for Tikhonov regularization:

                     n * || A X(α) - y ||_2^2
            GCV(α) = ------------------------
                        ( n - tr H(α) )^2

    where H(α) = A (A^T A + α L^T L)^{-1} A^T is the hat matrix mapping
    observations to fitted values. Optimized over log_alpha for robustness
    across many decades.
    """
    alpha = np.exp(log_alpha)
    n = A.shape[0]
    G = AtA + alpha * LtL
    x_alpha = np.linalg.solve(G, Aty)
    residual = A @ x_alpha - y
    rss = residual @ residual
    Z = np.linalg.solve(G, A.T)
    trace_H = np.einsum("ij,ji->", A, Z)
    return n * rss / (n - trace_H) ** 2


def _find_alpha_gcv(
    A: np.ndarray,
    y: np.ndarray,
    L: np.ndarray,
    log_alpha_bounds: tuple[float, float],
) -> float:
    """Locate argmin of GCV(α) on a log-bracketed scalar search (Brent)."""
    AtA = A.T @ A
    Aty = A.T @ y
    LtL = L.T @ L
    res = minimize_scalar(
        _gcv_functional,
        bounds=log_alpha_bounds,
        method="bounded",
        args=(A, y, AtA, Aty, LtL),
        options=dict(xatol=1e-4),
    )
    return float(np.exp(res.x))


def _sweep_regularization(
    A: np.ndarray,
    y: np.ndarray,
    L: np.ndarray,
    log_alpha_bounds: tuple[float, float],
    n_points: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep alpha geometrically across log_alpha_bounds and compute, for each:
        ρ(α) = ||A X(α) - y||_2,   η(α) = ||L X(α)||_2,   GCV(α).
    """
    alphas = np.geomspace(
        np.exp(log_alpha_bounds[0]),
        np.exp(log_alpha_bounds[1]),
        n_points,
    )
    AtA = A.T @ A
    Aty = A.T @ y
    LtL = L.T @ L

    rho = np.empty(n_points)
    eta = np.empty(n_points)
    gcv = np.empty(n_points)
    n = A.shape[0]
    for k, alpha in enumerate(alphas):
        G = AtA + alpha * LtL
        x_a = np.linalg.solve(G, Aty)
        residual = A @ x_a - y
        rho[k] = np.linalg.norm(residual)
        eta[k] = np.linalg.norm(L @ x_a)
        Z = np.linalg.solve(G, A.T)
        trace_H = np.einsum("ij,ji->", A, Z)
        gcv[k] = n * (residual @ residual) / (n - trace_H) ** 2
    return alphas, rho, eta, gcv


def _lcurve_corner(
    alphas: np.ndarray,
    rho: np.ndarray,
    eta: np.ndarray,
) -> tuple[float, int]:
    """
    Locate the L-curve corner as the point of maximum curvature of the
    log-log trajectory (log rho, log eta). Derivatives are estimated by
    finite differences (np.gradient); the two points at each end are masked
    because one-sided differences are unreliable there.
    """
    x = np.log(rho)
    y = np.log(eta)
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
    curvature[:2] = -np.inf
    curvature[-2:] = -np.inf
    idx = int(np.argmax(curvature))
    return float(alphas[idx]), idx


def deconvolve(
    A: np.ndarray,
    y: np.ndarray,
    L: np.ndarray,
    log_alpha_bounds: tuple[float, float] = (-8.0, 6.0),
) -> DeconvolutionResult:
    """
    Full Tikhonov deconvolution pass: sweep alpha, locate GCV optimum and
    L-curve corner, reconstruct X using GCV-selected alpha.
    """
    alphas, rho, eta, gcv = _sweep_regularization(A, y, L, log_alpha_bounds)
    alpha_gcv = _find_alpha_gcv(A, y, L, log_alpha_bounds)
    alpha_lcurve, idx_corner = _lcurve_corner(alphas, rho, eta)

    # Bound-saturation check: if GCV pins to an endpoint, the optimum lies
    # outside the search range. Benign for a flat, signal-free segment (GCV
    # correctly asks for maximum smoothing), but a warning sign if the
    # segment was expected to contain structure.
    lo, hi = np.exp(log_alpha_bounds[0]), np.exp(log_alpha_bounds[1])
    bound_note: str | None = None
    if alpha_gcv >= 0.99 * hi:
        bound_note = (
            f"GCV saturated at upper bound α={hi:.4g}; segment likely "
            f"signal-free, reconstruction ≈ smoothed baseline"
        )
    elif alpha_gcv <= 1.01 * lo:
        bound_note = (
            f"GCV saturated at lower bound α={lo:.4g}; search range "
            f"may be too narrow for this noise level"
        )

    x_recon = solve_tikhonov(A, y, L, alpha_gcv)
    return DeconvolutionResult(
        x_recon=x_recon,
        alpha_gcv=alpha_gcv,
        alpha_lcurve=alpha_lcurve,
        alphas=alphas,
        rho=rho,
        eta=eta,
        gcv_curve=gcv,
        idx_corner=idx_corner,
        bound_note=bound_note,
    )


# =============================================================================
# OPERATION 5: KINEMATIC SPATIAL COUPLING (SIMULATION only)
# =============================================================================
def map_temporal_to_spatial(
    t: np.ndarray,
    x_recon: np.ndarray,
    velocity_vector: np.ndarray = NOMINAL_CRUISE_SPEED * DEFAULT_HEADING,
    origin: np.ndarray = np.zeros(3),
) -> np.ndarray:
    """
    Map time-stamped synthetic samples to 3D Cartesian coordinates assuming
    constant-velocity straight-line transit:

            r(t) = origin + velocity_vector * t.

    Restricted to SIMULATION mode. PRODUCTION mode binds X_recon directly to
    GPS-stamped (lat, lon, alt) rows from the ingested CSV; calling this in
    PRODUCTION would silently overwrite measured positions with an idealized
    straight-line trajectory.

    Modeling scope: this function represents a single straight leg at
    constant velocity, appropriate for verifying the 1D deconvolution math
    on a digital twin, or for modeling one leg of a lawnmower survey
    pattern. It does not model multi-leg trajectories, banking turns, or
    altitude changes. For realistic multi-leg simulation, invoke this
    function once per leg and concatenate outputs.

    Parameters
    ----------
    t                : (n,) timestamps [s].
    x_recon          : (n,) reconstructed concentration values (passed for
                       symmetry with downstream pipelines; not consumed).
    velocity_vector  : (3,) constant velocity vector [m/s] in survey frame.
    origin           : (3,) starting position [m]. Default (0, 0, 0).

    Returns
    -------
    xyz : (n, 3) Cartesian coordinates [m].
    """
    if x_recon.shape[0] != t.shape[0]:
        raise ValueError(
            f"Length mismatch: t has {t.shape[0]} samples, "
            f"x_recon has {x_recon.shape[0]}."
        )
    return (origin[np.newaxis, :]
            + t[:, np.newaxis] * velocity_vector[np.newaxis, :])


# =============================================================================
# OPERATION 6: PRODUCTION CSV INGESTION AND VALIDATION
# =============================================================================
REQUIRED_COLUMNS = (
    "timestamp", "lat", "lon", "alt", "raw_pm25", "ground_speed",
)


def _parse_timestamp(raw: str) -> float:
    """
    Parse a single timestamp cell. Try float-seconds first; fall back to
    ISO 8601 via datetime.fromisoformat. Raises on failure.
    """
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError as e:
        raise RuntimeError(
            f"Unparseable timestamp '{raw}'. Expected float seconds or "
            f"ISO 8601 (e.g. '2026-06-29T14:30:00')."
        ) from e


def ingest_survey_telemetry(
    filepath: str,
    cruise_speed: float | None = None,
) -> SurveyData:
    """
    Parse an airborne survey CSV with header
        [timestamp, lat, lon, alt, raw_pm25, ground_speed]
    and return a validated SurveyData record.

    Validation pipeline, in the order it runs:
      (a) Schema check: all required columns present.
      (b) Sampling uniformity: std(dt) / mean(dt) < DT_CV_MAX (0.01),
          enforcing the Toeplitz convolution model's prerequisites. This
          runs before gap repair because the gap limit is converted from
          seconds into samples using the measured dt.
      (c) NaN rate audit: total raw_pm25 NaN rate <= NAN_RATE_MAX (5%).
      (d) Gap interpolation: raw_pm25 NaN runs no longer than
          GAP_MAX_SECONDS (1.0 s, half a sensor time constant) are
          linearly interpolated; longer runs trigger refusal. GPS columns
          (lat/lon/alt) and ground_speed are filled over the same span.
          The time axis is preserved intact; no rows are dropped.
      (e) Speed sanity check: the mean ground speed must lie between 0.4x
          and 2.5x the nominal cruise speed. This only catches a grossly
          misconfigured mission. Which samples are deconvolvable is decided
          downstream by per-leg relative gating, and the coverage gate
          (VELOCITY_COVERAGE_MIN) is enforced in deconvolve_segmented().

    Parameters
    ----------
    filepath     : path to the input CSV.
    cruise_speed : nominal cruise speed [m/s] used by the sanity check.
                   Defaults to NOMINAL_CRUISE_SPEED.

    Returns
    -------
    SurveyData with `pm25_was_interpolated` boolean mask flagging which
    raw_pm25 cells were filled.
    """
    if cruise_speed is None:
        cruise_speed = NOMINAL_CRUISE_SPEED

    # ---- 6.1 Read raw rows and check the schema -------------------------
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"CSV schema violation: missing required columns "
                f"{sorted(missing)}. Required header: "
                f"{list(REQUIRED_COLUMNS)}."
            )
        rows = list(reader)

    if len(rows) == 0:
        raise RuntimeError(f"Empty telemetry file: {filepath!r}.")

    # ---- 6.2 Parse columns ----------------------------------------------
    def col_float(name: str) -> np.ndarray:
        return np.array(
            [float(r[name]) if r[name] not in ("", "NaN", "nan") else np.nan
             for r in rows],
            dtype=float,
        )

    t_raw = np.array(
        [_parse_timestamp(r["timestamp"]) for r in rows], dtype=float
    )
    t = t_raw - t_raw[0]  # normalize to t0 = 0
    lat = col_float("lat")
    lon = col_float("lon")
    alt = col_float("alt")
    raw_pm25 = col_float("raw_pm25")
    ground_speed = col_float("ground_speed")

    n_samples = len(t)

    # ---- 6.3 Sampling uniformity gate (must precede gap repair) ---------
    # The time axis is validated before anything that depends on it,
    # because the interpolable gap length is expressed in SECONDS and
    # resolved into samples using the measured dt.
    dts = np.diff(t)
    if dts.size == 0 or np.any(dts <= 0):
        raise RuntimeError(
            "Telemetry timestamps non-monotonic or degenerate."
        )
    dt_mean = float(dts.mean())
    dt_cv = float(dts.std() / dt_mean)
    if dt_cv > DT_CV_MAX:
        raise RuntimeError(
            f"Sampling-uniformity violation: dt coefficient-of-variation = "
            f"{dt_cv:.4f} exceeds threshold {DT_CV_MAX}. Toeplitz "
            f"convolution model is invalid under non-uniform sampling. "
            f"Resample upstream to a uniform grid via linear interpolation "
            f"before re-running."
        )

    # Gap threshold in samples, derived from the measured logging rate so
    # the physical justification (half a sensor time constant) holds at
    # any rate.
    gap_max_samples = max(1, int(round(GAP_MAX_SECONDS / dt_mean)))

    # ---- 6.4 NaN audit on raw_pm25 --------------------------------------
    nan_rate = float(np.isnan(raw_pm25).mean())
    if nan_rate > NAN_RATE_MAX:
        raise RuntimeError(
            f"Telemetry corruption: raw_pm25 NaN rate = {nan_rate:.1%} "
            f"exceeds acceptance threshold of {NAN_RATE_MAX:.1%}. Inspect "
            f"sensor health or upstream logging integrity."
        )

    # ---- 6.5 Linear interpolation of raw_pm25 dropouts ------------------
    # Time axis is preserved (no row drops). Raises if any NaN run is
    # longer than gap_max_samples.
    pm25_was_interpolated = _interpolate_short_nan_runs(
        raw_pm25, gap_max_samples, field_name="raw_pm25",
    )
    n_interp = int(pm25_was_interpolated.sum())

    # ---- 6.6 GPS / ancillary fill for short gaps ------------------------
    for arr in (lat, lon, alt):
        _forward_fill_short_gaps(arr, gap_max_samples)
    # ground_speed dropouts also filled; needed for velocity gating
    _forward_fill_short_gaps(ground_speed, gap_max_samples)

    # Any column still carrying NaN after these recoveries indicates a gap
    # longer than the recoverable threshold in a non-PM2.5 column. Refuse.
    for name, arr in (("lat", lat), ("lon", lon), ("alt", alt),
                      ("ground_speed", ground_speed)):
        if np.any(np.isnan(arr)):
            n_nan = int(np.isnan(arr).sum())
            raise RuntimeError(
                f"Unrecoverable {name} dropout: {n_nan} NaN samples remain "
                f"after filling (gap exceeds {gap_max_samples} samples "
                f"= {GAP_MAX_SECONDS:.2f} s at dt = {dt_mean:.3f} s). "
                f"Resample upstream or refuse survey."
            )

    # ---- 6.7 Speed sanity check (wind-robust) ---------------------------
    # Over a lawnmower survey the along-track wind adds to ground speed on
    # one leg and subtracts on the reciprocal, so it cancels under
    # AVERAGING. The MEAN ground speed therefore approximates airspeed; the
    # median does not, because the distribution is bimodal (an upwind
    # cluster and a downwind cluster) and the median simply selects
    # whichever side is heavier.
    #
    # A per-sample absolute gate would be wind-biased and is NOT applied
    # here; deconvolvability is decided downstream by per-leg relative
    # gating. This check only catches a grossly wrong mission (e.g. the
    # aircraft never reached survey speed, or the cruise speed setting is
    # misconfigured).
    mean_speed = float(np.mean(ground_speed))
    if not (0.4 * cruise_speed <= mean_speed <= 2.5 * cruise_speed):
        raise RuntimeError(
            f"Implausible survey speed: mean ground speed = "
            f"{mean_speed:.2f} m/s against a nominal cruise of "
            f"{cruise_speed} m/s. Check the cruise speed setting against "
            f"the actual flight envelope, or inspect the telemetry."
        )

    print(f"  [ingest] {n_samples} samples retained, dt = {dt_mean:.4f} s "
          f"(CV = {dt_cv:.2e})")
    print(f"  [ingest] raw_pm25 NaN rate = {nan_rate:.2%}, "
          f"{n_interp} samples linearly interpolated "
          f"(gap limit {gap_max_samples} sample(s) = "
          f"{GAP_MAX_SECONDS:.2f} s)")
    print(f"  [ingest] mean ground speed = {mean_speed:.2f} m/s "
          f"(nominal cruise {cruise_speed} m/s)")

    return SurveyData(
        t=t, lat=lat, lon=lon, alt=alt,
        raw_pm25=raw_pm25, ground_speed=ground_speed, dt=dt_mean,
        pm25_was_interpolated=pm25_was_interpolated,
    )


def _forward_fill_short_gaps(arr: np.ndarray, max_gap: int) -> None:
    """
    In-place fill of NaN runs of length <= max_gap; longer runs are left as
    NaN so the caller can refuse them. Interior and trailing runs are
    forward-filled from the last valid value. A leading run (NaN at the
    very start, e.g. before the first GPS fix) has no earlier value, so it
    is back-filled from the first valid value instead.
    """
    n = arr.size
    i = 0
    while i < n:
        if np.isnan(arr[i]):
            j = i
            while j < n and np.isnan(arr[j]):
                j += 1
            run_len = j - i
            if run_len <= max_gap:
                if i > 0:
                    arr[i:j] = arr[i - 1]
                elif j < n:
                    arr[i:j] = arr[j]
            i = j
        else:
            i += 1


def _interpolate_short_nan_runs(
    arr: np.ndarray,
    max_gap: int,
    field_name: str = "field",
) -> np.ndarray:
    """
    Linearly interpolate NaN runs of length <= max_gap in `arr` (in-place).

    Returns a boolean mask flagging which indices were filled. Raises
    RuntimeError if any NaN run exceeds max_gap samples: short dropouts are
    recoverable, longer ones are not.

    Edge runs (NaN at the leading or trailing boundary) are filled by
    constant extrapolation from the single available anchor, since linear
    interpolation requires two endpoints. These are still gap-length
    checked.
    """
    n = arr.size
    was_interp = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if np.isnan(arr[i]):
            j = i
            while j < n and np.isnan(arr[j]):
                j += 1
            run_len = j - i
            if run_len > max_gap:
                raise RuntimeError(
                    f"{field_name} NaN run of length {run_len} starting at "
                    f"index {i} exceeds interpolation gap threshold of "
                    f"{max_gap} samples. Inspect sensor uptime; resample "
                    f"upstream or refuse survey."
                )
            left_anchor = arr[i - 1] if i > 0 else np.nan
            right_anchor = arr[j] if j < n else np.nan
            if not np.isnan(left_anchor) and not np.isnan(right_anchor):
                # Interior run: true two-sided linear interpolation
                ramp = np.linspace(left_anchor, right_anchor, run_len + 2)
                arr[i:j] = ramp[1:-1]
            elif not np.isnan(left_anchor):
                # Trailing edge run: extrapolate from left anchor
                arr[i:j] = left_anchor
            elif not np.isnan(right_anchor):
                # Leading edge run: extrapolate from right anchor
                arr[i:j] = right_anchor
            else:
                raise RuntimeError(
                    f"{field_name} array contains no valid anchor for "
                    f"interpolation (entirely NaN)."
                )
            was_interp[i:j] = True
            i = j
        else:
            i += 1
    return was_interp


# =============================================================================
# OPERATION 7: SURVEY-LEG DETECTION AND VELOCITY-STABLE SEGMENTATION
# =============================================================================
def compute_track_headings(
    lat: np.ndarray,
    lon: np.ndarray,
    dt: float,
    baseline_s: float = HEADING_BASELINE_SECONDS,
) -> np.ndarray:
    """
    Derive per-sample ground-track heading [deg, 0-360) from GPS fixes
    using a local planar (equirectangular) approximation, which is accurate
    to well under a degree over a 1 km x 1 km survey box.

    Each heading is computed across a centred window spanning about
    `baseline_s` seconds (k samples either side, k = baseline_s / (2 dt)),
    rather than between adjacent fixes. At 1 Hz and 10 m/s, adjacent fixes
    are only 10 m apart, so normal GPS fix-to-fix jitter of 0.5 to 2 m
    would swing the heading by more than the leg tolerance and break the
    survey into fragments. A 6 s window spreads the same jitter over a
    60 m baseline. The window is clipped at the ends of the record.

        dN = R * d(lat)                     [m, northward]
        dE = R * d(lon) * cos(lat)          [m, eastward]
        heading = atan2(dE, dN)             [deg clockwise from North]

    Returns
    -------
    headings : (n,) ground-track heading in degrees, 0 <= h < 360.
    """
    R_EARTH = 6_371_000.0
    n = lat.size
    k = max(1, int(round(baseline_s / (2.0 * dt))))
    idx = np.arange(n)
    lo = np.clip(idx - k, 0, n - 1)
    hi = np.clip(idx + k, 0, n - 1)
    d_north = R_EARTH * np.radians(lat[hi] - lat[lo])
    d_east = (R_EARTH * np.radians(lon[hi] - lon[lo])
              * np.cos(np.radians(lat)))
    return np.degrees(np.arctan2(d_east, d_north)) % 360.0


def _angular_deviation(headings: np.ndarray, reference: float) -> np.ndarray:
    """
    Smallest absolute angular difference [deg] between each heading and a
    reference, correctly handling the 359 deg -> 1 deg wraparound.
    """
    diff = np.abs((headings - reference + 180.0) % 360.0 - 180.0)
    return diff


def detect_survey_legs(
    headings: np.ndarray,
    dt: float,
    tau: float = TAU,
    heading_tol_deg: float = HEADING_TOLERANCE_DEG,
) -> list[tuple[int, int]]:
    """
    Partition the flight into survey legs: maximal contiguous runs over
    which the ground-track heading stays within `heading_tol_deg` of the
    run's opening heading.

    Turns, which by definition sweep heading rapidly, terminate a run and
    are therefore excluded automatically. This supersedes the need for a
    separate turn-rate gate.

    Runs shorter than the sensor transient settling length
    max(16, ceil(4*tau/dt)) are discarded: such a run is entirely warm-up
    and contains no well-conditioned interior to deconvolve.

    Returns
    -------
    legs : list of half-open (start, end) index pairs.
    """
    n = headings.size
    min_length = max(16, int(np.ceil(4 * tau / dt)))
    legs: list[tuple[int, int]] = []

    start = 0
    while start < n:
        reference = headings[start]
        end = start + 1
        while end < n:
            deviation = _angular_deviation(headings[end:end + 1], reference)
            if deviation[0] > heading_tol_deg:
                break
            end += 1
        if end - start >= min_length:
            legs.append((start, end))
        start = end
    return legs


def find_velocity_stable_segments(
    ground_speed: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    dt: float,
    tau: float = TAU,
    heading_tol_deg: float = HEADING_TOLERANCE_DEG,
    leg_tol: float = LEG_VELOCITY_TOLERANCE,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Identify deconvolvable segments using PER-LEG RELATIVE velocity gating.

    Rationale. The sensor time constant tau is governed by airspeed, but
    only GPS ground speed is logged. Over a lawnmower survey the two differ
    by the along-track wind component, which reverses sign between
    reciprocal legs and can reach ±30% of cruise (10 m/s in a 3 m/s wind).
    Gating ground speed against a single global constant would therefore
    reject most of a perfectly valid flight.

    Within one leg, however, heading is fixed, so the wind contribution is
    constant. Deviation of a sample's ground speed from its OWN leg's
    median is therefore attributable to genuine airspeed variation (gusts,
    throttle excursions), not to wind. Gating per leg removes the wind bias
    entirely.

    Procedure:
      1. Derive ground-track headings from GPS over a
         HEADING_BASELINE_SECONDS window.
      2. Partition into survey legs by heading stability.
      3. Within each leg, take the median ground speed as that leg's
         reference.
      4. Keep maximal runs inside the leg where
         |v - median| / median <= leg_tol.
      5. Discard runs shorter than the 4*tau transient length.

    Returns
    -------
    segments : list of half-open (start, end) index pairs to deconvolve.
    legs     : list of the detected survey legs, for reporting.
    """
    min_length = max(16, int(np.ceil(4 * tau / dt)))
    headings = compute_track_headings(lat, lon, dt)
    legs = detect_survey_legs(headings, dt, tau, heading_tol_deg)

    segments: list[tuple[int, int]] = []
    for (ls, le) in legs:
        leg_speed = ground_speed[ls:le]
        leg_median = float(np.median(leg_speed))
        if leg_median <= 0:
            continue
        ok = np.abs(leg_speed - leg_median) / leg_median <= leg_tol

        # Collect maximal True runs within this leg
        in_run = False
        start = 0
        for i, flag in enumerate(ok):
            if flag and not in_run:
                start, in_run = i, True
            elif not flag and in_run:
                if i - start >= min_length:
                    segments.append((ls + start, ls + i))
                in_run = False
        if in_run and ok.size - start >= min_length:
            segments.append((ls + start, ls + ok.size))

    return segments, legs


def estimate_wind_from_reciprocal_legs(
    legs: list[tuple[int, int]],
    lat: np.ndarray,
    lon: np.ndarray,
    ground_speed: np.ndarray,
    dt: float,
    reciprocal_tol_deg: float = 20.0,
) -> tuple[float, float] | None:
    """
    Diagnostic only; does not gate anything.

    For a pair of legs flown on reciprocal headings, the along-track wind
    component reverses sign while the airspeed does not:

        |Vg_A| = sqrt(Vair^2 - w_perp^2) + w_par
        |Vg_B| = sqrt(Vair^2 - w_perp^2) - w_par

    Hence the half-sum estimates airspeed (biased low by any crosswind) and
    the half-difference recovers the along-track wind exactly. Averaging
    over all reciprocal pairs found in the flight gives a robust estimate.

    Returns
    -------
    (airspeed_estimate, along_track_wind) in m/s, or None if the flight
    contains no reciprocal leg pairs.
    """
    if len(legs) < 2:
        return None

    headings = compute_track_headings(lat, lon, dt)
    leg_info = [
        (float(np.median(headings[s:e])), float(np.median(ground_speed[s:e])))
        for (s, e) in legs
    ]

    half_sums: list[float] = []
    half_diffs: list[float] = []
    used: set[int] = set()
    for i, (h_i, v_i) in enumerate(leg_info):
        if i in used:
            continue
        for j in range(i + 1, len(leg_info)):
            if j in used:
                continue
            h_j, v_j = leg_info[j]
            # Reciprocal means headings differ by ~180 deg
            opposite = (h_i + 180.0) % 360.0
            if (_angular_deviation(np.array([h_j]), opposite)[0]
                    <= reciprocal_tol_deg):
                half_sums.append(0.5 * (v_i + v_j))
                half_diffs.append(0.5 * abs(v_i - v_j))
                used.update({i, j})
                break

    if not half_sums:
        return None
    return float(np.mean(half_sums)), float(np.mean(half_diffs))


def deconvolve_segmented(
    raw_pm25: np.ndarray,
    ground_speed: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    dt: float,
    tau: float = TAU,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Detect survey legs, isolate velocity-stable segments, and deconvolve
    each segment independently. Returns a full-length cleaned vector with
    NaN over excluded windows, plus the segment list for reporting.

    The coverage gate (VELOCITY_COVERAGE_MIN) is enforced here, after
    segmentation, because the deconvolvable fraction of the flight is only
    known once the segments have been isolated.
    """
    n = raw_pm25.size
    cleaned = np.full(n, np.nan)

    segments, legs = find_velocity_stable_segments(
        ground_speed, lat, lon, dt, tau,
    )

    if not segments:
        raise RuntimeError(
            f"No velocity-stable segments of sufficient length found "
            f"({len(legs)} survey legs detected, min segment length = "
            f"{max(16, int(np.ceil(4 * tau / dt)))} samples). "
            f"Trajectory cannot be deconvolved."
        )

    print(f"  [legs]    {len(legs)} survey leg(s) detected "
          f"(heading tolerance ±{HEADING_TOLERANCE_DEG:.0f}°)")

    wind = estimate_wind_from_reciprocal_legs(
        legs, lat, lon, ground_speed, dt,
    )
    if wind is not None:
        airspeed_est, wind_component = wind
        print(f"  [wind]    reciprocal-leg estimate: airspeed ≈ "
              f"{airspeed_est:.2f} m/s, along-track wind ≈ "
              f"{wind_component:.2f} m/s")

    print(f"  [segment] {len(segments)} velocity-stable segment(s) isolated "
          f"(±{LEG_VELOCITY_TOLERANCE:.0%} of each leg's median "
          f"ground speed)")

    for k, (s, e) in enumerate(segments):
        seg_len = e - s
        A_seg = build_convolution_matrix(seg_len, dt, tau)
        L_seg = first_difference_operator(seg_len)
        y_seg = raw_pm25[s:e]
        result = deconvolve(A_seg, y_seg, L_seg)
        cleaned[s:e] = result.x_recon
        print(f"    segment {k + 1}: indices [{s}:{e}] "
              f"({seg_len} samples, α* = {result.alpha_gcv:.4g})")
        if result.bound_note is not None:
            print(f"        [warn] {result.bound_note}")

    excluded = int(np.isnan(cleaned).sum())
    coverage = 1.0 - excluded / n
    if coverage < VELOCITY_COVERAGE_MIN:
        raise RuntimeError(
            f"Severe survey execution instability: only {coverage:.1%} of "
            f"samples fell inside a stable survey leg within "
            f"±{LEG_VELOCITY_TOLERANCE:.0%} of that leg's median ground "
            f"speed. Deployment validity requires "
            f"≥{VELOCITY_COVERAGE_MIN:.0%}. Survey rejected: re-fly with "
            f"steadier legs, or review the heading and velocity tolerances "
            f"against the actual flight profile."
        )
    if excluded > 0:
        print(f"  [segment] {excluded} samples ({excluded / n:.1%}) "
              f"excluded: turns, transients, or velocity excursions "
              f"({coverage:.1%} coverage, "
              f"gate ≥{VELOCITY_COVERAGE_MIN:.0%})")

    return cleaned, segments


# =============================================================================
# OPERATION 8: OUTPUT CSV EMISSION
# =============================================================================
def write_cleaned_csv(
    survey: SurveyData,
    cleaned_pm25: np.ndarray,
    output_path: str,
) -> None:
    """
    Emit cleaned spatial telemetry preserving the full input schema with
    two appended columns:
        - cleaned_pm25            : Tikhonov-deconvolved PM2.5 [ug/m^3];
                                    blank for samples outside
                                    velocity-stable segments.
        - pm25_was_interpolated   : "1" if raw_pm25 at this row was
                                    linearly interpolated from a short
                                    dropout, else "0". Phase 2 Kriging may
                                    downweight flagged samples in the
                                    variogram.

    Note: the `raw_pm25` column carries the post-interpolation values (NaN
    cells are filled). Original dropout positions are recoverable from the
    `pm25_was_interpolated` flag column.
    """
    header = list(REQUIRED_COLUMNS) + [
        "cleaned_pm25", "pm25_was_interpolated",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(survey.t.size):
            writer.writerow([
                f"{survey.t[i]:.6f}",
                f"{survey.lat[i]:.8f}",
                f"{survey.lon[i]:.8f}",
                f"{survey.alt[i]:.4f}",
                f"{survey.raw_pm25[i]:.6f}",
                f"{survey.ground_speed[i]:.4f}",
                "" if np.isnan(cleaned_pm25[i]) else f"{cleaned_pm25[i]:.6f}",
                "1" if survey.pm25_was_interpolated[i] else "0",
            ])
    print(f"  [output] wrote {survey.t.size} rows to {output_path}")


# =============================================================================
# OPERATION 9: VISUAL VERIFICATION DASHBOARD (SIMULATION)
# =============================================================================
def render_dashboard(
    t: np.ndarray,
    x_true: np.ndarray,
    y_obs: np.ndarray,
    result: DeconvolutionResult,
    savepath: str | None = None,
    show: bool = True,
) -> None:
    """
    4-panel verification dashboard for SIMULATION-mode runs. Saved to
    `savepath` if given (parent folders are created as needed). The plot
    window is opened only if `show` is True.
    """
    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1.0, 1.2])
    gcv_label = fr"$\alpha^*_{{\mathrm{{GCV}}}}={result.alpha_gcv:.3g}$"
    lc_label = fr"$\alpha_{{\mathrm{{Lcurve}}}}={result.alpha_lcurve:.3g}$"

    # Panel 1: signals overlay
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, x_true, color="black", lw=2.0, label="Ground truth X(t)")
    ax1.plot(t, y_obs, color="#d62728", lw=0.9, alpha=0.6,
             label="Observed telemetry Y(t)")
    ax1.plot(t, result.x_recon, color="#1f77b4", lw=1.6, ls="--",
             label=f"Tikhonov recon  ({gcv_label})")
    ax1.set_xlabel("Survey time t [s]")
    ax1.set_ylabel(r"PM2.5 concentration  [$\mu$g/m$^3$]")
    ax1.set_title("Phase 1: Time-domain deconvolution of SPS30 telemetry")
    ax1.legend(loc="upper left", framealpha=0.95)
    ax1.grid(alpha=0.3)

    # Panel 2: absolute error residuals
    ax2 = fig.add_subplot(gs[1, :])
    err_obs = np.abs(x_true - y_obs)
    err_rec = np.abs(x_true - result.x_recon)
    rec_label = r"$|X - X_{\mathrm{recon}}|$"
    ax2.plot(t, err_obs, color="#d62728", lw=0.9, alpha=0.7,
             label=fr"$|X - Y|$  (MAE = {err_obs.mean():.3f})")
    ax2.plot(t, err_rec, color="#1f77b4", lw=1.2,
             label=f"{rec_label}  (MAE = {err_rec.mean():.3f})")
    ax2.set_xlabel("Survey time t [s]")
    ax2.set_ylabel(r"Absolute error  [$\mu$g/m$^3$]")
    ax2.set_title(
        "Recovery fidelity: observation error vs. reconstruction error")
    ax2.legend(loc="upper right", framealpha=0.95)
    ax2.grid(alpha=0.3)

    # Panel 3a: L-curve diagnostic
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.loglog(result.rho, result.eta, color="#2ca02c", lw=1.4,
               marker="o", ms=3)
    ic = result.idx_corner
    ax3.loglog(result.rho[ic], result.eta[ic], "s",
               color="#ff7f0e", ms=11, mfc="none", mew=2.0,
               label=fr"L-curve corner  $\alpha={result.alpha_lcurve:.3g}$")
    log_gap = np.abs(np.log(result.alphas) - np.log(result.alpha_gcv))
    idx_gcv = int(np.argmin(log_gap))
    ax3.loglog(result.rho[idx_gcv], result.eta[idx_gcv], "D",
               color="#1f77b4", ms=9, mfc="none", mew=2.0,
               label=fr"GCV optimum  $\alpha={result.alpha_gcv:.3g}$")
    ax3.set_xlabel(r"Residual norm  $\|AX - Y\|_2$")
    ax3.set_ylabel(r"Solution seminorm  $\|LX\|_2$")
    ax3.set_title("L-curve diagnostic")
    ax3.legend(loc="lower left", framealpha=0.95, fontsize=9)
    ax3.grid(which="both", alpha=0.3)

    # Panel 3b: GCV functional
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.loglog(result.alphas, result.gcv_curve, color="#9467bd", lw=1.4)
    ax4.axvline(result.alpha_gcv, color="#1f77b4", ls="--", lw=1.5,
                label=gcv_label)
    ax4.axvline(result.alpha_lcurve, color="#ff7f0e", ls=":", lw=1.5,
                label=lc_label)
    ax4.set_xlabel(r"Regularization scalar  $\alpha$")
    ax4.set_ylabel(r"GCV$(\alpha)$")
    ax4.set_title("GCV functional")
    ax4.legend(loc="best", framealpha=0.95, fontsize=9)
    ax4.grid(which="both", alpha=0.3)

    if savepath is not None:
        folder = os.path.dirname(savepath)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(savepath, dpi=140, bbox_inches="tight")
        print(f"  [figure] dashboard saved to {savepath}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# EXECUTION-MODE DRIVER
# =============================================================================
def _run_simulation(
    savepath: str | None = None,
    show: bool = True,
    tau: float = TAU,
    cruise_speed: float = NOMINAL_CRUISE_SPEED,
    seed: int | None = RNG_SEED,
) -> DeconvolutionResult:
    """Synthetic digital twin pipeline: generate X, simulate Y, deconvolve."""
    print("=" * 72)
    print("MODE: SIMULATION  (synthetic digital twin verification)")
    print("=" * 72)

    rng = np.random.default_rng(seed)
    t, x_true = generate_ground_truth()
    A = build_convolution_matrix(N, DT, tau)
    y_exact, drift, y_obs = simulate_observation(A, x_true, t, rng=rng)
    L = first_difference_operator(N)
    result = deconvolve(A, y_obs, L)

    # Spatial coupling (SIMULATION exclusive)
    xyz = map_temporal_to_spatial(
        t, result.x_recon, velocity_vector=cruise_speed * DEFAULT_HEADING,
    )

    mae_obs = float(np.mean(np.abs(x_true - y_obs)))
    mae_rec = float(np.mean(np.abs(x_true - result.x_recon)))
    rmse_obs = float(np.sqrt(np.mean((x_true - y_obs) ** 2)))
    rmse_rec = float(np.sqrt(np.mean((x_true - result.x_recon) ** 2)))

    print(f"  Samples N                : {N}")
    print(f"  dt                       : {DT} s")
    print(f"  Sensor tau               : {tau} s")
    print(f"  Random seed              : {seed}")
    print(f"  Cruise speed             : {cruise_speed} m/s "
          f"(heading +X, provisional)")
    print(f"  Spatial lag τ·v          : {tau * cruise_speed:.1f} m")
    print(f"  Spatial extent           : X ∈ [{xyz[0, 0]:.1f}, "
          f"{xyz[-1, 0]:.1f}] m")
    print(f"  GCV α*                   : {result.alpha_gcv:.6g}")
    print(f"  L-curve corner α         : {result.alpha_lcurve:.6g}")
    print(f"  MAE  observation vs X    : {mae_obs:.4f} ug/m^3")
    print(f"  MAE  reconstruction vs X : {mae_rec:.4f} ug/m^3")
    print(f"  RMSE observation vs X    : {rmse_obs:.4f} ug/m^3")
    print(f"  RMSE reconstruction vs X : {rmse_rec:.4f} ug/m^3")
    print(f"  Recovery gain (RMSE)     : {rmse_obs / rmse_rec:.2f}×")
    print("=" * 72)

    render_dashboard(t, x_true, y_obs, result, savepath=savepath, show=show)
    return result


def _run_production(
    csv_path: str,
    output_path: str,
    tau: float = TAU,
    cruise_speed: float = NOMINAL_CRUISE_SPEED,
) -> np.ndarray:
    """CSV ingestion -> validation -> segmented deconvolution -> output."""
    print("=" * 72)
    print(f"MODE: PRODUCTION  (ingesting {csv_path})")
    print("=" * 72)

    survey = ingest_survey_telemetry(csv_path, cruise_speed=cruise_speed)
    cleaned, segments = deconvolve_segmented(
        survey.raw_pm25, survey.ground_speed,
        survey.lat, survey.lon, survey.dt, tau=tau,
    )
    write_cleaned_csv(survey, cleaned, output_path)
    kept = int((~np.isnan(cleaned)).sum())
    print("=" * 72)
    print(f"  Phase 2 handoff artifact: {output_path}")
    print("  Spatial binding         : raw GPS (lat, lon, alt) per row")
    print(f"  Deconvolved samples     : {kept} / {cleaned.size}")
    print("=" * 72)
    return cleaned


def main(
    mode: str = "SIMULATION",
    csv_path: str | None = None,
    output_path: str = DEFAULT_OUTPUT_PATH,
    figure_path: str | None = None,
    show: bool = True,
    tau: float = TAU,
    cruise_speed: float = NOMINAL_CRUISE_SPEED,
    seed: int | None = RNG_SEED,
) -> None:
    """
    Execution-mode dispatcher, usable from Python or through the command
    line interface below.

    Parameters
    ----------
    mode         : "SIMULATION" (synthetic) or "PRODUCTION" (CSV-driven);
                   case-insensitive.
    csv_path     : input telemetry CSV path. Required for PRODUCTION.
    output_path  : cleaned output CSV path (PRODUCTION). Defaults to
                   cleaned_survey_data.csv.
    figure_path  : optional save path for the SIMULATION dashboard.
    show         : open the dashboard plot window (SIMULATION).
    tau          : sensor time constant [s]. Defaults to TAU.
    cruise_speed : nominal cruise speed [m/s]. Defaults to
                   NOMINAL_CRUISE_SPEED.
    seed         : random seed for SIMULATION. Defaults to RNG_SEED.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}.")
    if cruise_speed <= 0:
        raise ValueError(
            f"cruise_speed must be positive, got {cruise_speed}.")

    mode = mode.upper()
    if mode == "SIMULATION":
        _run_simulation(savepath=figure_path, show=show, tau=tau,
                        cruise_speed=cruise_speed, seed=seed)
    elif mode == "PRODUCTION":
        if not csv_path:
            raise ValueError("PRODUCTION mode requires `csv_path`.")
        _run_production(csv_path, output_path, tau=tau,
                        cruise_speed=cruise_speed)
    else:
        raise ValueError(
            f"Unknown mode {mode!r}. Use 'SIMULATION' or 'PRODUCTION'.")


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    """Command-line options for both execution modes."""
    parser = argparse.ArgumentParser(
        prog="pm25_deconvolution.py",
        description=(
            "Phase 1 deconvolution engine: recovers PM2.5 concentration "
            "from lagged SPS30 telemetry using Tikhonov regularization."
        ),
    )
    parser.add_argument(
        "--mode", choices=["simulation", "production"],
        default="simulation",
        help="simulation (synthetic digital twin) or production (real "
             "flight CSV). Default: simulation.",
    )
    parser.add_argument(
        "--input", metavar="CSV",
        help="input flight telemetry CSV (required for production).",
    )
    parser.add_argument(
        "--output", metavar="CSV", default=DEFAULT_OUTPUT_PATH,
        help=f"cleaned output CSV (production). "
             f"Default: {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--figure", metavar="PNG", default=DEFAULT_FIGURE_PATH,
        help=f"where to save the dashboard (simulation). "
             f"Default: {DEFAULT_FIGURE_PATH}.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="save the dashboard without opening a plot window.",
    )
    parser.add_argument(
        "--tau", type=float, default=TAU,
        help=f"sensor time constant in seconds. Default: {TAU}.",
    )
    parser.add_argument(
        "--cruise-speed", type=float, default=NOMINAL_CRUISE_SPEED,
        help=f"nominal cruise speed in m/s. "
             f"Default: {NOMINAL_CRUISE_SPEED}.",
    )
    parser.add_argument(
        "--seed", type=int, default=RNG_SEED,
        help=f"random seed for the simulation. Default: {RNG_SEED}.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.mode == "production" and not args.input:
        parser.error("--mode production requires --input CSV")
    if args.tau <= 0:
        parser.error("--tau must be positive")
    if args.cruise_speed <= 0:
        parser.error("--cruise-speed must be positive")
    try:
        main(
            mode=args.mode,
            csv_path=args.input,
            output_path=args.output,
            figure_path=args.figure,
            show=not args.no_show,
            tau=args.tau,
            cruise_speed=args.cruise_speed,
            seed=args.seed,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as err:
        sys.exit(f"error: {err}")
