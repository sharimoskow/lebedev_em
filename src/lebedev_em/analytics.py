"""
analytics.py — Analytical electromagnetic Green's functions.

Provides closed-form solutions used to validate the numerical scheme.

The primary reference benchmark (DDH03 Figs. 2–3) is the magnetic field of a
magnetic dipole in a homogeneous isotropic whole space.

Reference: Davydycheva, Druskin & Habashy (2003), Geophysics 68(5):1525–1536.
"""

from __future__ import annotations

import numpy as np
from .media import MU0, EPS0


def _wavenumber(omega: float, sigma: float, mu: float, eps: float) -> complex:
    """
    Complex wavenumber k for a homogeneous isotropic medium.
    k² = iω μ (σ − iω ε) = iω μ σ̇
    Convention: time dependence exp(−iωt), so Im(k) > 0 for decaying field.
    """
    sigma_dot = complex(sigma) - 1j * omega * eps  # σ̇ = σ + iωε  (note sign)
    # Actually: σ̇ = σ + iωε in DDH03 (eq 1: ∇×H = σE − iωεE),
    # so effective conductivity is σ_eff = σ − iωε.  But k² = iωμσ̇:
    k_sq = 1j * omega * mu * (sigma - 1j * omega * eps)
    # Choose branch with Im(k) > 0 (fields decay away from source)
    k = np.sqrt(k_sq + 0j)
    if np.imag(k) < 0:
        k = -k
    return k


def magnetic_dipole_B(
    x: float,
    y: float,
    z: float,
    sigma: float,
    omega: float,
    mu: float = MU0,
    eps: float = EPS0,
    dipole_comp: int = 0,
    moment: float = 1.0,
) -> np.ndarray:
    """
    Magnetic field **B** of a magnetic dipole in a homogeneous isotropic medium.

    The dipole is located at the origin with moment  m = moment · ê_{dipole_comp}.

    Parameters
    ----------
    x, y, z : float
        Observation point (must not be exactly at the origin).
    sigma : float   [S/m]
    omega : float   [rad/s]
    mu    : float   [H/m]   default μ₀
    eps   : float   [F/m]   default ε₀
    dipole_comp : int   0=x, 1=y, 2=z
    moment : float

    Returns
    -------
    B : ndarray, shape (3,), complex
        (Bx, By, Bz) at the observation point.

    Notes
    -----
    For a magnetic dipole m at the origin the magnetic field is (SI, frequency domain):

        B = (μ/4π) { [3(m·r̂)r̂ − m] (k²r² − 3ikr − 3)/r³ + m k²(ikr − 1)/r } exp(ikr)/(k²r²)  × (−1/r)

    or more compactly using the dyadic Green's function:

        Bₐ = (μ m / 4π) Gₐᵦ mᵦ / (k²r²)  × f(kr) + ...

    We use the standard result for the αβ component of the magnetic Green's function:

        G_αβ(r) = exp(ikr)/(4π r³) { (3r̂ₐr̂ᵦ − δₐᵦ)(1 − ikr) + (r̂ₐr̂ᵦ − δₐᵦ) k²r² }   ... (*)
                  + δₐᵦ δ(r)/3

    Away from the source the delta term vanishes and we use (*).  See, e.g.,
    Chew (1995), "Waves and Fields in Inhomogeneous Media", §2.3.
    """
    r_vec = np.array([x, y, z], dtype=complex)
    r = float(np.real(np.sqrt(np.dot(r_vec, r_vec))))
    if r < 1e-30:
        raise ValueError("Observation point must not be at the origin (singularity).")

    k = _wavenumber(omega, sigma, mu, eps)
    kr = k * r
    r_hat = np.array([x, y, z]) / r

    # Scalar prefactor
    pref = mu * moment / (4.0 * np.pi * r**3)
    exp_ikr = np.exp(1j * kr)

    # Dipole direction vector
    m_hat = np.zeros(3)
    m_hat[dipole_comp] = 1.0

    # Cross-terms (3 (m·r̂) r̂ − m)
    m_dot_rhat = float(np.dot(m_hat, r_hat))
    cross_term = 3.0 * m_dot_rhat * r_hat - m_hat

    # B = (μ m / 4π r³) exp(ikr) { cross_term (1 − ikr − k²r²/3) + ... }
    # Full formula (see e.g. Griffiths / Chew):
    #   B = (μ/4π) exp(ikr)/r³ { cross_term [k²r² − 3ikr − 3]/k²r²
    #                            + m_hat [k²r²]/k²r² }  × k²
    # Simplifying using the textbook form:
    #   B_near/mid = pref · exp(ikr) { cross_term (1 − ikr) − m_hat (kr)² }  / k²r²  × k²
    # Use the exact formula:
    factor_near = (1.0 - 1j * kr) * exp_ikr / (kr**2)
    factor_far  = exp_ikr  # k²r² term

    B = pref * (
        cross_term * (1.0 - 1j * kr - (kr)**2 / 3.0) * exp_ikr / (kr**2 / 3.0 + 1e-300)
        # This is getting complicated; use the exact closed-form instead:
    )

    # ---- Exact closed-form (Chew 1995, eq. 2.3.8 or Jackson §9.1) ----
    # For magnetic dipole m = moment * ê_α in homogeneous medium:
    #
    #   B(r) = (μ / 4π) { k² (r̂ × m) × r̂ exp(ikr)/r
    #                    + [3r̂(r̂·m) − m](1/r³ − ik/r²) exp(ikr) }
    #
    # (SI, exp(−iωt) convention)

    rxm = np.cross(r_hat, m_hat)
    rxmxr = np.cross(rxm, r_hat)  # = m − (m·r̂)r̂ = transverse part of m

    term1 = k**2 * rxmxr * exp_ikr / r                          # radiation term
    term2 = (3.0 * m_dot_rhat * r_hat - m_hat) * (1.0 / r**3 - 1j * k / r**2) * exp_ikr

    B = (mu * moment / (4.0 * np.pi)) * (term1 + term2)
    return B


def electric_dipole_E(
    x: float,
    y: float,
    z: float,
    sigma: float,
    omega: float,
    mu: float = MU0,
    eps: float = EPS0,
    dipole_comp: int = 0,
    moment: float = 1.0,
) -> np.ndarray:
    """
    Electric field **E** of an electric current dipole in a homogeneous medium.

    Parameters
    ----------
    (same as magnetic_dipole_B but for an electric dipole)

    Returns
    -------
    E : ndarray, shape (3,), complex
    """
    r_vec = np.array([x, y, z], dtype=float)
    r = float(np.linalg.norm(r_vec))
    if r < 1e-30:
        raise ValueError("Observation point must not be at the source (singularity).")

    k = _wavenumber(omega, sigma, mu, eps)
    kr = k * r
    r_hat = r_vec / r

    m_hat = np.zeros(3)
    m_hat[dipole_comp] = 1.0
    m_dot_rhat = float(np.dot(m_hat, r_hat))

    exp_ikr = np.exp(1j * kr)
    sigma_dot = complex(sigma - 1j * omega * eps)

    # E-field of electric dipole (Hertz dipole):
    # E = (moment / (4π σ̇)) { k² (ê × r̂) × r̂ exp(ikr)/r + ...}
    # Near-field (static) + radiation terms:
    #   E_α = (moment / (4π σ̇ r³)) exp(ikr) { (3r̂_α r̂_β − δ_αβ) (1 − ikr) + δ_αβ k²r² } m̂_β
    rxm = np.cross(r_hat, m_hat)
    rxmxr = np.cross(rxm, r_hat)  # transverse component

    prefac = moment / (4.0 * np.pi * sigma_dot)
    term1 = k**2 * rxmxr * exp_ikr / r
    term2 = (3.0 * m_dot_rhat * r_hat - m_hat) * (1.0 / r**3 - 1j * k / r**2) * exp_ikr

    E = prefac * (term1 + term2)
    return E


# ---------------------------------------------------------------------------
# Two-layer (half-space contact) analytic — Sommerfeld integral
# ---------------------------------------------------------------------------

def electric_dipole_Ez_two_layer_onaxis(
    z_r: float,
    z_s: float,
    z_c: float,
    sigma1: float,
    sigma2: float,
    omega: float,
    mu: float = MU0,
    eps: float = EPS0,
    moment: float = 1.0,
) -> complex:
    """
    E_z on-axis (x=y=0) from a z-directed electric current dipole in a
    two-half-space model.

    Geometry
    --------
    Layer 1  :  z < z_c,   conductivity sigma1
    Layer 2  :  z ≥ z_c,   conductivity sigma2
    Source   :  VED at (0, 0, z_s),  z_s < z_c  (must be in layer 1)
    Receiver :  (0, 0, z_r),  z_r > z_s

    Method
    ------
    Sommerfeld / Hertz-potential integral.  For source in layer 1 and
    interface at z = z_c:

      γ_j(λ) = sqrt(λ² − k_j²),  Re(γ_j) > 0

    **Receiver in layer 1** (z_r < z_c):

      E_z = (moment / (4π σ̇₁))
            ∫₀^∞ λ³/γ₁ [exp(−γ₁ d₁) + R exp(−γ₁ d_R)] dλ

      d₁ = z_r − z_s   (source → receiver)
      d_R = 2(z_c−z_s) − (z_r−z_s)   (source → image receiver via interface)
      R  = (σ̇₂ γ₁ − σ̇₁ γ₂) / (σ̇₁ γ₂ + σ̇₂ γ₁)

    **Receiver in layer 2** (z_r ≥ z_c):

      E_z = (moment / (2π))
            ∫₀^∞ λ³ / (σ̇₁ γ₂ + σ̇₂ γ₁) exp(−γ₁ h) exp(−γ₂ (z_r−z_c)) dλ

      h = z_c − z_s

    At induction-logging frequencies (ω ≈ 2π·2500, σ ≈ 0.1–1 S/m):
    displacement currents are negligible (ωε₀/σ < 10⁻⁶), so σ̇ ≈ σ and
    k_j² ≈ iω μ σ_j.  The full k_j (from _wavenumber) is used for accuracy.

    Returns
    -------
    E_z : complex  [V/m] at (0, 0, z_r)
    """
    from scipy.integrate import quad

    if z_s >= z_c:
        raise ValueError("Source must be in layer 1 (z_s < z_c).")
    if z_r <= z_s:
        raise ValueError("Receiver must be above source (z_r > z_s).")

    k1 = _wavenumber(omega, sigma1, mu, eps)
    k2 = _wavenumber(omega, sigma2, mu, eps)

    # σ̇ = σ − iωε  (DDH03 convention, used in overall prefactor)
    sdot1 = complex(sigma1 - 1j * omega * eps)
    sdot2 = complex(sigma2 - 1j * omega * eps)

    h  = z_c - z_s          # source → interface
    d1 = z_r - z_s          # source → receiver

    def _gamma1(lam: float) -> complex:
        """Re(γ₁) > 0 by numpy principal branch."""
        return np.sqrt(complex(lam * lam - k1 * k1))

    def _gamma2(lam: float) -> complex:
        return np.sqrt(complex(lam * lam - k2 * k2))

    def _cquad(f, limit: int = 500, epsabs: float = 1.49e-9) -> complex:
        re = quad(lambda x: float(np.real(f(x))), 0.0, np.inf,
                  limit=limit, epsabs=epsabs)[0]
        im = quad(lambda x: float(np.imag(f(x))), 0.0, np.inf,
                  limit=limit, epsabs=epsabs)[0]
        return re + 1j * im

    if z_r < z_c:
        # Receiver in layer 1: primary + reflected
        dR = 2.0 * h - d1   # reflected path length (> 0 when z_r < z_c)

        def kernel(lam: float) -> complex:
            g1 = _gamma1(lam)
            g2 = _gamma2(lam)
            R = (sdot2 * g1 - sdot1 * g2) / (sdot1 * g2 + sdot2 * g1)
            primary   = np.exp(-g1 * d1)
            reflected = R * np.exp(-g1 * dR)
            return lam ** 3 / g1 * (primary + reflected)

        return moment / (4.0 * np.pi * sdot1) * _cquad(kernel)

    else:
        # Receiver in layer 2: transmitted field only
        d2 = z_r - z_c      # interface → receiver

        def kernel(lam: float) -> complex:
            g1 = _gamma1(lam)
            g2 = _gamma2(lam)
            denom = sdot1 * g2 + sdot2 * g1
            return (lam ** 3 / denom
                    * np.exp(-g1 * h)
                    * np.exp(-g2 * d2))

        return moment / (2.0 * np.pi) * _cquad(kernel)


def electric_dipole_Ez_homogeneous_onaxis(
    z_r: float,
    z_s: float,
    sigma: float,
    omega: float,
    mu: float = MU0,
    eps: float = EPS0,
    moment: float = 1.0,
) -> complex:
    """
    E_z on-axis (x=y=0) from a z-directed electric dipole at (0,0,z_s)
    in a homogeneous isotropic whole-space.

    Closed-form expression from `electric_dipole_E`:
        E_z = (moment / (4π σ̇)) · 2 exp(ik r) (1/r − ik) / r²
    where r = |z_r − z_s| and σ̇ = σ − iωε.

    Also equal to the Sommerfeld integral
    (Il/4π σ̇) ∫₀^∞ λ³/γ exp(−γr) dλ.
    """
    r = abs(z_r - z_s)
    if r < 1e-30:
        raise ValueError("Receiver and source must not coincide.")
    E_vec = electric_dipole_E(
        0.0, 0.0, float(z_r - z_s),
        sigma, omega, mu, eps, dipole_comp=2, moment=float(moment),
    )
    return complex(E_vec[2])


def Bxx_homogeneous(
    z_obs: np.ndarray,
    sigma: float,
    omega: float,
    mu: float = MU0,
    eps: float = EPS0,
) -> np.ndarray:
    """
    B_x component of an x-oriented magnetic dipole at the origin, evaluated
    at points (0, 0, z) on the borehole axis.  This reproduces the benchmark
    in DDH03 Fig. 2.

    Parameters
    ----------
    z_obs : 1-D array of floats
        z-coordinates of evaluation points (must be ≠ 0).

    Returns
    -------
    Bxx : ndarray of complex, same shape as z_obs.
    """
    z_obs = np.asarray(z_obs, dtype=float)
    Bxx = np.empty_like(z_obs, dtype=complex)
    for idx, z in enumerate(z_obs):
        if abs(z) < 1e-20:
            Bxx[idx] = np.nan
            continue
        B = magnetic_dipole_B(0.0, 0.0, z, sigma, omega, mu, eps, dipole_comp=0)
        Bxx[idx] = B[0]
    return Bxx
