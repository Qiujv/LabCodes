import numpy as np
import pytest

from labcodes import tomo

qt = pytest.importorskip("qutip", reason="qutip is required for tomography cross-check tests")


def _qutip_probs(rho_qt, tomo_ops):
    probs = []
    for op in tomo_ops:
        op_qt = qt.Qobj(op)
        rotated = op_qt * rho_qt * op_qt.dag()
        probs.append(np.real(rotated.diag()))
    return np.concatenate(probs)


@pytest.mark.parametrize("n_qbs", [1, 2])
@pytest.mark.parametrize("fit_method", ["cvx", "lstsq"])
def test_qst_recovers_qutip_density_matrix(n_qbs, fit_method):
    dim = 2**n_qbs
    rho_qt = qt.rand_dm(dim, density=1.0, seed=1234 + n_qbs)
    rho_target = rho_qt.full()

    tomo_ops = tomo.qst_transform_matrix("ixy", n_qbs, return_tomo_ops=True)
    probs = _qutip_probs(rho_qt, tomo_ops)

    rho_fit = tomo.qst(probs, tomo_ops="ixy", fit_method=fit_method)
    np.testing.assert_allclose(np.asarray(rho_fit), rho_target, atol=1e-6)
