"""
test_anisotropic_backus.py — the anisotropic Backus / laminate homogenization.

Locks in the July-2026 fix: the `backus` method must use the FULL tensor
laminate (DDH03 eq. 9), not the scalar ⅓·tr σ proxy which collapsed an
anisotropic layer to an over-conductive isotropic average.  For a TI layer
whose symmetry axis is the interface normal (σ_nn = σ_T/200), the effective
normal conductivity must be the *harmonic* mean of σ_nn — not the harmonic
mean of trace/3.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lebedev_em.grid import symmetric_uniform_grid
from lebedev_em.media import (
    _anisotropic_backus_tensor_3d,
    _standard_backus_tensor_3d,
    _nodal_eff_tensor_general,
    from_sigma_func,
)

N75 = np.array([np.sin(np.radians(75.0)), 0.0, np.cos(np.radians(75.0))])


def _ddh03_eq9_reference(s1, s2, f, n_hat):
    """Independent LITERAL transcription of DDH03 eq. (9) in the (T, N) frame,
    used to certify `_anisotropic_backus_tensor_3d` is that exact formula."""
    n = np.asarray(n_hat, float); n = n / np.linalg.norm(n)
    idx = int(np.argmin(np.abs(n))); v = np.zeros(3); v[idx] = 1.0
    m = v - np.dot(v, n) * n; m /= np.linalg.norm(m)
    q = np.cross(n, m); q /= np.linalg.norm(q)
    R = np.column_stack([m, q, n]).astype(complex)

    def blocks(s):
        A = R.T @ np.asarray(s, complex) @ R
        return A[:2, :2], A[:2, 2], A[2, :2], A[2, 2]   # σ_TT, σ_TN, σ_NT, σ_NN

    TT1, TN1, NT1, NN1 = blocks(s1)
    TT2, TN2, NT2, NN2 = blocks(s2)
    w1, w2 = complex(f), complex(1 - f)
    avg = lambda a, b: w1 * a + w2 * b
    SNN = 1.0 / avg(1 / NN1, 1 / NN2)                       # Σ_NN
    TN_invNN = avg(TN1 / NN1, TN2 / NN2)                    # ⟨σ_TN σ_NN⁻¹⟩
    invNN_NT = avg(NT1 / NN1, NT2 / NN2)                    # ⟨σ_NN⁻¹ σ_NT⟩
    TN_invNN_NT = avg(np.outer(TN1, NT1) / NN1, np.outer(TN2, NT2) / NN2)
    STT = avg(TT1, TT2) - TN_invNN_NT + np.outer(TN_invNN, invNN_NT) * SNN
    B = np.zeros((3, 3), complex)
    B[:2, :2] = STT; B[:2, 2] = TN_invNN * SNN
    B[2, :2] = SNN * invNN_NT; B[2, 2] = SNN
    return R @ B @ R.T


def test_is_literal_ddh03_eq9_for_general_anisotropic():
    """The averaging must be DDH03 eq. (9) itself, for arbitrary anisotropic
    layer tensors — not merely the TI special case."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(500):
        A = rng.normal(size=(3, 3)); s1 = A @ A.T
        B = rng.normal(size=(3, 3)); s2 = B @ B.T
        f = rng.uniform(0.05, 0.95); n = rng.normal(size=3)
        worst = max(worst, np.abs(
            _anisotropic_backus_tensor_3d(s1, s2, f, n)
            - _ddh03_eq9_reference(s1, s2, f, n)).max())
    assert worst < 1e-11, f"not eq.(9): worst diff {worst:.2e}"


def test_never_overshoots_for_clean_two_material_inputs():
    """Eq. (9) of a passive mixture must keep effective eigenvalues within the
    constituent range — no overshoot above the largest constituent eigenvalue.
    (Overshoots seen on the grid therefore come from dirty extraction, not the
    formula.)"""
    rng = np.random.default_rng(2)
    for _ in range(1000):
        A = rng.normal(size=(3, 3)); s1 = A @ A.T
        B = rng.normal(size=(3, 3)); s2 = B @ B.T
        f = rng.uniform(0.02, 0.98); n = rng.normal(size=3)
        lo = min(np.linalg.eigvalsh(s1).min(), np.linalg.eigvalsh(s2).min())
        hi = max(np.linalg.eigvalsh(s1).max(), np.linalg.eigvalsh(s2).max())
        ev = np.linalg.eigvalsh(_anisotropic_backus_tensor_3d(s1, s2, f, n).real)
        assert ev.max() <= hi * (1 + 1e-9) and ev.min() >= lo * (1 - 1e-9)


def test_reduces_to_scalar_backus_for_isotropic():
    a, b, f = 0.3, 0.05, 0.4
    T = _anisotropic_backus_tensor_3d(a, b, f, N75)
    ref = _standard_backus_tensor_3d(f * a + (1 - f) * b, f / a + (1 - f) / b, N75)
    assert np.abs(T - ref).max() < 1e-13


def test_equals_general_nodal_with_symmetric_fractions():
    SIG_T, SIG_N = 0.1, 0.1 / 200.0
    SA = SIG_T * np.eye(3) + (SIG_N - SIG_T) * np.outer(N75, N75)
    SB = 0.1 * np.eye(3)
    for f in (0.2, 0.5, 0.8):
        Tb = _anisotropic_backus_tensor_3d(SA, SB, f, N75)
        Tg = _nodal_eff_tensor_general(SA, SB, f, np.array([f, f, f]), N75)
        assert np.abs(Tb - Tg).max() < 1e-12


def test_normal_conductivity_is_harmonic_sigma_nn_not_trace():
    """The bug being fixed: scalar-proxy Backus used ⅓·tr σ (≈0.067) for the
    layer; correct Backus uses σ_nn = σ_N = 5e-4, giving a far smaller (and
    physical) harmonic normal conductivity."""
    SIG_T, SIG_N = 0.1, 0.1 / 200.0
    SA = SIG_T * np.eye(3) + (SIG_N - SIG_T) * np.outer(N75, N75)
    SB = 0.1 * np.eye(3)
    f = 0.3
    T = _anisotropic_backus_tensor_3d(SA, SB, f, N75)
    snn_correct = 1.0 / (f / SIG_N + (1 - f) / 0.1)      # harmonic of true σ_nn
    snn_proxy = 1.0 / (f / (np.trace(SA).real / 3) + (1 - f) / 0.1)
    assert abs(float((N75 @ T @ N75).real) - snn_correct) < 1e-9
    # eigenvalues physical: within [σ_N, σ_T]
    ev = np.sort(np.linalg.eigvals(T).real)
    assert ev[0] > 0 and ev[-1] <= SIG_T * (1 + 1e-9)
    # and the correct normal conductivity is far below the proxy's
    assert snn_correct < 0.1 * snn_proxy


def test_from_sigma_func_backus_sees_the_anisotropy():
    """End-to-end: method='backus' on a tilted TI layer must produce interface
    tensors whose normal conductivity is harmonic-σ_nn small, not trace/3."""
    SIG_T, SIG_N = 0.1, 0.1 / 200.0
    NH = N75
    SA = SIG_T * np.eye(3) + (SIG_N - SIG_T) * np.outer(NH, NH)
    d0 = NH[2] * (-1.2)

    def sf(X, Y, Z):
        X = np.asarray(X, float); Y = np.asarray(Y, float); Z = np.asarray(Z, float)
        o = np.zeros(np.broadcast(X, Y, Z).shape + (3, 3), complex)
        o[...] = 0.1 * np.eye(3)
        o[np.abs(NH[0] * X + NH[2] * Z - d0) < 0.25 / 2] = SA
        return o

    grid = symmetric_uniform_grid(16, 16, 40, 5.0, 5.0, 6.0)
    med = from_sigma_func(grid, sf, h_svd=0.05, n_line=20, n_vol=6, method="backus")
    s = np.array(med.sigma_R)
    assert s.ndim == 3, "anisotropic layer should force tensor storage"
    # at least one interface cell must have a normal conductivity well below
    # the trace/3 floor (0.067) — i.e. it resolved σ_nn, not the proxy.
    nn = np.einsum("i,nij,j->n", NH, s, NH).real
    assert nn.min() < 0.02, f"min n·σ·n = {nn.min():.4f}; Backus not seeing σ_nn"
