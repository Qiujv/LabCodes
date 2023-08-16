"""Module for processing data of Quantum State Tomography (QST) and Quantum Processs Tomography (QPT)

Adapted from pyle_tomo and the matlab code from Youpeng Zhong.


# Explanation of QST

For QST, one should measure qubit state probabilities after applying different `tomo_ops`,
then the density matrix is obtained by

`rho = qst(probs, tomo_ops="ixy")`,

where `tomo_ops` are list of matrices of the operations applied before measurement.
Typically it is [I, X/2, Y/2]. You can pass `tomo_ops="ixy"` to use it.
`rho` is reconstructed on basis of |0> and |1>.

For example, in 1-qubit case, the input probs should be like:
`probs = [i_p0, i_p1, x_p0, x_p1, y_p0, y_p1]`

In 2-qubit case, it is:
```
probs = [ii_p00, ii_p01, ii_p10, ii_p11,
         ix_p00, ix_p01, ix_p10, ix_p11,
         iy_p00, iy_p01, iy_p10, iy_p11,
         xi_p00, xi_p01, xi_p10, xi_p11,
         xx_p00, xx_p01, xx_p10, xx_p11,
         xy_p00, xy_p01, xy_p10, xy_p11,
         yi_p00, yi_p01, yi_p10, yi_p11,
         yx_p00, yx_p01, yx_p10, yx_p11,
         yy_p00, yy_p01, yy_p10, yy_p11,]
```
`rho` on basis states [00, 01, 10, 11].

In n-qubit case, it is:
- `probs` of length `(2*3)**n_qbs`, 
- operations labelled as itertools.product('ixy', repeat=n_qbs)
- and basis states organized as itertools.product('01', repeat=n_qbs).

Besides we provide:
- `qst_cvx`, `qst_lstsq`, `qst_transform` as lower-level API for reconstructing density matrices.
- `get_probs`, to calculate ideal measured probabilities from given density matrix.
- `random_density_matrix`, `warn_if_not_herm_unit_or_pos`... as their name indicates.


# Explanation of QPT

For QPT, one should prepare some qubit states as input state (usually "0xy1"), 
then measure the density matrix of output state. The process matrix is obtained by

`chi = qpt(rho_out, rho_in="0xy1", basis="sigma")`,

where `rho_in`, `rho_out` are list of density matrices of same shape, though it's not enforced,
and `basis` are list of matrices that specifies the basis reconstructed process matrix.
Typically it is [I, sigmaX, sigmaY, sigmaZ], you can pass `basis="sigma"` to use it.

Everything extending to multi-qubit case is just nested loops here, e.g.
- ixy -> [ii, ix, iy, xi, xx, xy, yi, yx, yy]
- 01 -> [00, 01, 10, 11] -> [000, 001, 010, 011, 100, 101, 110, 111]

So to the init_ops="0xy1" and chi basis="IXYZ" as well as general list of matrices.

Besides we provide:
- `qpt_cvx`, `qpt_lstsq`, `qpt_transform` as lower-level API.
- `get_rho_in`, `get_rho_out` to get ideal density matrices given init_ops and chi.
"""

import functools
import itertools
import logging
import math
from typing import Literal, Union

import numpy as np
import scipy

from labcodes import misc

logger = logging.getLogger(__name__)


def Rmat(axis: np.matrix, angle: float) -> np.matrix:
    """Return the rotation matrix.

    >>> Rmat(np.array([[0, -1j], [1j, 0]]), np.pi / 2)
    array([[ 0.70710678+0.j, -0.70710678+0.j],
           [ 0.70710678+0.j,  0.70710678+0.j]])
    """
    return scipy.linalg.expm(-1j * angle / 2 * axis)


sigmaI = np.eye(2, dtype=complex)
sigmaX = np.array([[0, 1], [1, 0]], dtype=complex)
sigmaY = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigmaZ = np.array([[1, 0], [0, -1]], dtype=complex)

sigmaP = (sigmaX - 1j * sigmaY) / 2
sigmaM = (sigmaX + 1j * sigmaY) / 2

Xpi2 = Rmat(sigmaX, np.pi / 2)  # np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)
Ypi2 = Rmat(sigmaY, np.pi / 2)  # np.array([[1, -1], [1, 1]]) / np.sqrt(2)
Zpi2 = Rmat(sigmaZ, np.pi / 2)  # np.array([[1-1j, 0], [0, 1+1j]]) / np.sqrt(2)

Xpi = Rmat(sigmaX, np.pi)
Ypi = Rmat(sigmaY, np.pi)
Zpi = Rmat(sigmaZ, np.pi)

Xmpi2 = Rmat(sigmaX, -np.pi / 2)
Ympi2 = Rmat(sigmaY, -np.pi / 2)
Zmpi2 = Rmat(sigmaZ, -np.pi / 2)

Xmpi = Rmat(sigmaX, -np.pi)
Ympi = Rmat(sigmaY, -np.pi)
Zmpi = Rmat(sigmaZ, -np.pi)


def tensor(matrix_list: list[np.matrix]) -> np.matrix:
    """Returns tensor product of given matrices.

    >>> mats = [np.eye(2), [[0,3],[3,0]]]
    >>> tensor(mats)
    array([[0., 3., 0., 0.],
           [3., 0., 0., 0.],
           [0., 0., 0., 3.],
           [0., 0., 3., 0.]])
    >>> mats.append(np.eye(2)*5)
    >>> tensor(mats)
    array([[ 0.,  0., 15.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0., 15.,  0.,  0.,  0.,  0.],
           [15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0., 15.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0., 15.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0., 15.],
           [ 0.,  0.,  0.,  0., 15.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0., 15.,  0.,  0.]])
    """
    return functools.reduce(np.kron, matrix_list)


def tensor_combinations(matrix_list: list[np.matrix], repeat: int) -> list[np.matrix]:
    """Return a list of all tensor combinations of the input matrices.

    >>> tensor_combinations([1, 2, 3], 2)
    [1, 2, 3, 2, 4, 6, 3, 6, 9]
    >>> tensor_combinations([[1], [2], [3]], 2)
    [array([1]), array([2]), array([3]), array([2]), array([4]), array([6]), array([3]), array([6]), array([9])]
    >>> tensor_combinations([1, 2, 3], 1)
    [1, 2, 3]
    """
    return [tensor(mats) for mats in itertools.product(matrix_list, repeat=repeat)]


def qst(
    probs: np.ndarray,
    tomo_ops: Union[list[np.matrix], Literal["ixy", "oct"]] = "ixy",
    fit_method: Literal["cvx", "lstsq"] = "cvx",
) -> np.matrix:
    """Find the density matrix given the measured probabilities.

    >>> rho_fit = qst([1. , 0. , 0.5, 0.5, 0.5, 0.5])
    >>> np.allclose([[1,0],[0,0]], rho_fit, atol=1e-5)
    True
    """
    # Infer number of qubits.
    if tomo_ops == "ixy":
        n_ops = 3
    elif tomo_ops == "oct":
        n_ops = 6
    elif isinstance(tomo_ops, str):
        raise ValueError(f"Unknown tomo_ops={tomo_ops}")
    else:
        n_ops = len(tomo_ops)

    n_qbs = int(math.log(len(probs), n_ops * 2))  # 2 for 2 states.
    if len(probs) != (2 * n_ops) ** n_qbs:
        raise ValueError(
            "Mismatch between probs and tomo_ops."
            f"len(probs)={len(probs)} is not a power of {2*n_ops}."
        )

    trans_mat = qst_transform_matrix(tomo_ops, n_qbs)

    # Fit the density matrix.
    if fit_method == "cvx":
        rho = qst_cvx(probs, trans_mat)
    elif fit_method == "lstsq":
        rho = qst_lstsq(probs, trans_mat)
    else:
        raise ValueError(f"Unknown fit_method: {fit_method}")

    # TODO: return the residue?
    return rho


@misc.cache_with_bypass()
def qst_transform_matrix(
    tomo_ops: Union[list[np.matrix], Literal["ixy", "oct"]] = "ixy",
    n_qbs: int = 1,
    return_tomo_ops: bool = False,
) -> np.matrix:
    """Returns the transformation matrix for state tomography:
    `transform @ rho.ravel() = probs`.

    >>> qst_transform_matrix('ixy')
    array([[ 1. +0.j ,  0. +0.j ,  0. +0.j ,  0. +0.j ],
           [ 0. +0.j ,  0. +0.j ,  0. +0.j ,  1. +0.j ],
           [ 0.5+0.j ,  0. +0.5j,  0. -0.5j,  0.5+0.j ],
           [ 0.5+0.j ,  0. -0.5j,  0. +0.5j,  0.5+0.j ],
           [ 0.5+0.j , -0.5-0.j , -0.5+0.j ,  0.5+0.j ],
           [ 0.5+0.j ,  0.5+0.j ,  0.5+0.j ,  0.5+0.j ]])
    >>> np.allclose(qst_transform_matrix('ixy'),
    ...             qst_transform_matrix([np.eye(2), Xpi2, Ypi2]))
    True
    """
    if tomo_ops == "ixy":
        tomo_ops = [np.eye(2), Xpi2, Ypi2]
    elif tomo_ops == "oct":
        tomo_ops = [np.eye(2), Xpi2, Ypi2, Xmpi2, Ympi2, Xpi]
    elif isinstance(tomo_ops, str):
        raise ValueError(f"Unknown tomo_ops={tomo_ops}")
    else:
        pass

    mq_tomo_ops = tensor_combinations(tomo_ops, n_qbs)
    if return_tomo_ops:
        return mq_tomo_ops

    trans_mat = _qst_transform_matrix(mq_tomo_ops)
    return trans_mat


def _qst_transform_matrix(tomo_ops: list[np.matrix]) -> np.matrix:
    tomo_ops = np.asarray(tomo_ops)  # For array indexing.
    n_ops = len(tomo_ops)
    n_states = len(tomo_ops[0])
    transform_shape = (n_ops * n_states, n_states**2)
    # TODO: use sparse matrix?
    # TODO: warn for large matrix.

    # TODO: try np.einsum?
    def trans_element(k: int, l: int):
        i, j = divmod(k, n_states)
        m, n = divmod(l, n_states)
        return tomo_ops[i, j, m] * tomo_ops[i, j, n].conj()

    if n_states <= 16:  # 1-4 qubits.
        trans_mat = np.fromfunction(trans_element, transform_shape, dtype=int)
    else:  # 5+ qubits.
        # Slower, but takes less memory, according to pyle.
        trans_mat = np.zeros(transform_shape, dtype=complex)
        for k in range(transform_shape[0]):
            for l in range(transform_shape[1]):
                trans_mat[k, l] = trans_element(k, l)

    return trans_mat


def qst_cvx(probs: np.ndarray, trans_mat: np.matrix) -> np.matrix:
    """Find the density matrix given the measured probabilities.

    CVXPY is used to constrain the density matrix to be hermitian, unit-trace and
    semi-positive-definite.

    Args:
        probs: 1d array of measured probabilities. Ordered as required by transform.
        trans_mat: Transformation matrix. `trans_mat @ rho.ravel() = probs`.

    Returns:
        the density matrix.

    Examples:
    >>> rho = np.array([[0,0,0,0],[0,.5,.5,0],[0,.5,.5,0],[0,0,0,0]])
    >>> trans_mat = qst_transform_matrix('ixy', 2)
    >>> probs = trans_mat @ rho.ravel()
    >>> rho_fit = qst_cvx(probs, trans_mat)
    >>> np.allclose(rho, rho_fit, atol=1e-7)
    True
    """
    rho_size = trans_mat.shape[1]
    n_states = int(np.sqrt(rho_size))  # 2 ** n_qbs
    if n_states**2 != rho_size:
        logger.warning(f"Cannot infer n_states from trans_mat.shape={trans_mat.shape}")

    probs = np.asarray(probs).ravel()

    import cvxpy as cp  # dll load failure reports at first import, but nothing wrong.

    # Installation see: https://www.iitk.ac.in/mwn/SRS/docs/cvx_installation.pdf
    # or https://www.cvxpy.org/index.html

    rho = cp.Variable((n_states, n_states), hermitian=True)  # Density matrix.
    # NOTE: cvxpy's default reshape order is "F", while numpy is "C".
    # Mistaking the reshape order can lead to wrong result, which is easy to see by test
    # fitting a random density matrix (as we did in test_qst or test_qpt).
    # Both pyle and ZYP's code admits the default reshape order. But here we follow pyle
    # to construct the transformation matrix, i.e. the reshape order should be "C".
    # Strange thing is: this fixs the problem in qpt but leads to error in qst.
    # So here we use the default of cvxpy, i.e. "F", but in qpt we use "C".
    objective = cp.Minimize(cp.norm(trans_mat @ cp.reshape(rho, rho_size) - probs))
    constraints = [
        cp.trace(rho) == 1,
        rho >> 0,
    ]  # See https://www.cvxpy.org/tutorial/advanced/index.html#semidefinite-matrices
    problem = cp.Problem(objective, constraints)
    problem.solve()
    warn_if_not_herm_unit_or_pos(rho.value.T)
    return rho.value.T  # Transpose as in ZYP's matlab code.


def qst_lstsq(probs: np.ndarray, trans_mat: np.matrix) -> np.matrix:
    """Find the density matrix given the measured probabilities.

    Using least squares and the return are NOT guaranteed to be hermitian, unit-trace,
    or semi-positive-definite.

    Args:
        probs: 1d array of measured probabilities. Ordered as required by transform.
        trans_mat: Transformation matrix. `trans_mat @ rho.ravel() = probs`.

    Returns:
        the density matrix.

    Examples:
    >>> rho = np.array([[0,0,0,0],[0,.5,.5,0],[0,.5,.5,0],[0,0,0,0]])
    >>> trans_mat = qst_transform_matrix('ixy', 2)
    >>> probs = trans_mat @ rho.ravel()
    >>> rho_fit = qst_lstsq(probs, trans_mat)
    >>> np.allclose(rho, rho_fit)
    True
    """
    probs = np.asarray(probs).ravel()
    rho: np.ndarray = np.linalg.lstsq(trans_mat, probs, rcond=-1)[0]

    rho_size = trans_mat.shape[1]
    n_states = int(np.sqrt(rho_size))  # 2 ** n_qbs
    if n_states**2 != rho_size:
        logger.warning(f"Cannot infer n_states from trans_mat.shape={trans_mat.shape}")
    try:
        rho = rho.reshape(n_states, -1)  # No transpose as in pyle's code.
        warn_if_not_herm_unit_or_pos(rho)
        return rho
    except:
        logger.exception(
            f"Returning a flatten rho of shape={rho.shape}, "
            f"reshape failed with trans_mat.shape={trans_mat.shape} "
            f"and probs.shape={probs.shape}"
        )
        return rho


def qpt(
    rho_out: list[np.matrix],
    rho_in: Union[list[np.matrix], Literal["0xy1"]] = "0xy1",
    basis: Union[list[np.matrix], Literal["sigma", "raise-lower"]] = "sigma",
    fit_method: Literal["cvx", "lstsq"] = "cvx",
):
    """Find the process matrix given the input and output density matrices.

    Example:
    >>> rho_out = [Ypi @ rho @ Ypi.T.conj() for rho in get_rho_in("0xy1", 1)]
    >>> chi = qpt(rho_out)
    >>> chi_ideal = np.zeros((4,4))
    >>> chi_ideal[2,2] = 1
    >>> np.allclose(chi, chi_ideal, atol=1e-5)
    True
    """
    # Infer number of qubits.
    if basis == "sigma" or basis == "raise-lower":
        n_basis = 4
    elif isinstance(basis, str):
        raise ValueError(f"Unknown basis={basis}")
    else:
        n_basis = len(basis)

    n_qbs = int(math.log(len(rho_out), n_basis))
    if len(rho_out) != n_basis**n_qbs:
        raise ValueError(
            "Mismatch between rho_out and basis."
            f"len(rho_out)={len(rho_out)} is not a power of {n_basis}."
        )

    if rho_in == "0xy1":
        rho_in = get_rho_in("0xy1", n_qbs)
    elif isinstance(rho_in, str):
        raise ValueError(f"Unknown rho_in={rho_in}")

    trans_mat = qpt_transform_matrix(basis, n_qbs)

    # Fit the chi matrix.
    if fit_method == "cvx":
        chi = qpt_cvx(rho_in, rho_out, trans_mat)
    elif fit_method == "lstsq":
        chi = qpt_lstsq(rho_in, rho_out, trans_mat)
    else:
        raise ValueError(f"Unknown fit_method: {fit_method}")

    return chi


@misc.cache_with_bypass()
def qpt_transform_matrix(
    basis: Union[list[np.matrix], Literal["sigma", "raise-lower"]] = "sigma",
    n_qbs: int = 1,
    return_basis: bool = False,
) -> np.matrix:
    """Returns the transformation matrix for process tomography:
    `rho_out @ pointer = rho_in`
    `transform @ chi.ravel() = pointer.ravel()`

    >>> qpt_transform_matrix("sigma").shape
    (16, 16)
    >>> np.allclose(qpt_transform_matrix("sigma"),
    ...             qpt_transform_matrix([np.eye(2), sigmaX, sigmaY, sigmaZ]))
    True
    """
    if basis == "sigma":
        basis = [np.eye(2), sigmaX, sigmaY, sigmaZ]
    elif basis == "raise-lower":
        basis = [np.eye(2), sigmaP, sigmaM, sigmaZ]
    elif isinstance(basis, str):
        raise ValueError(f"Unknown basis={basis}")
    else:
        pass

    mq_basis = tensor_combinations(basis, n_qbs)
    if return_basis:
        return mq_basis

    trans_mat = _qpt_transform_matrix(mq_basis)
    return trans_mat


def _qpt_transform_matrix(basis: list[np.matrix]) -> np.matrix:
    basis = np.asarray(basis)  # For array indexing.
    n_out, n_in = basis[0].shape
    chi_size = n_out * n_in
    transform_shape = (chi_size**2, chi_size**2)

    def trans_element(alpha: int, beta: int):
        L, J = divmod(alpha, chi_size)
        M, N = divmod(beta, chi_size)
        i, j = divmod(J, n_out)
        k, l = divmod(L, n_in)
        return basis[M, i, k] * basis[N, j, l].conj()

    if chi_size <= 16:  # One or two qubits.
        trans_mat = np.fromfunction(trans_element, transform_shape, dtype=int)
    else:  # three or more qubits.
        # Slower, but takes less memory, according to pyle.
        trans_mat = np.zeros(transform_shape, dtype=complex)
        for alpha in range(transform_shape[0]):
            for beta in range(transform_shape[1]):
                trans_mat[alpha, beta] = trans_element(alpha, beta)

    return trans_mat


def qpt_cvx(
    rho_in: list[np.matrix],
    rho_out: list[np.matrix],
    trans_mat: np.matrix,
) -> np.matrix:
    """Find the process matrix on basis specified by trans_mat given the input and output.

    CVXPY is used to constrain the process matrix to be hermitian, unit-trace and
    semi-positive-definite.

    It is actually solving the linear equations:
    `rho_out @ pointer = rho_in`
    `transform @ chi.ravel() = pointer.ravel()`

    Returns:
        the process matrix.

    Examples:
    >>> rho_in = get_rho_in("0xy1", 2)
    >>> rho_out = rho_in
    >>> trans_mat = qpt_transform_matrix("sigma", 2)
    >>> chi = qpt_cvx(rho_in, rho_out, trans_mat)
    >>> chi_ideal = np.zeros((16, 16))
    >>> chi_ideal[0, 0] = 1
    >>> np.allclose(chi, chi_ideal, atol=1e-5)
    True
    """
    chi_size = trans_mat.shape[1]
    n_basis = int(np.sqrt(chi_size))
    if n_basis**2 != chi_size:
        logger.warning(f"Cannot infer n_basis from trans_mat.shape={trans_mat.shape}")

    # Assumes uniform shape in rho_in and rho_out, and same length of rho_in and rho_out.
    n_inits = len(rho_in)
    rho_in = np.asarray(rho_in).reshape(n_inits, rho_in[0].size)
    rho_out = np.asarray(rho_out).reshape(n_inits, rho_out[0].size)

    import cvxpy as cp

    chi = cp.Variable((n_basis, n_basis), hermitian=True)
    pointer = trans_mat @ cp.reshape(chi, chi_size, "C")
    pointer = cp.reshape(pointer, (n_basis, n_basis), "C")
    objective = cp.Minimize(cp.norm(rho_in @ pointer - rho_out))
    constraints = [
        cp.trace(chi) == 1,
        chi >> 0,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    warn_if_not_herm_unit_or_pos(chi.value)
    return chi.value  # No transpose as in ZYP's matlab code.


def qpt_lstsq(
    rho_in: list[np.matrix],
    rho_out: list[np.matrix],
    trans_mat: np.matrix,
) -> np.matrix:
    """Find the process matrix on basis specified by trans_mat given the input and output.

    Using least squares and the return are NOT guaranteed to be hermitian, unit-trace,
    or semi-positive-definite.

    It is actually solving the linear equations:
    `rho_out @ pointer = rho_in`
    `transform @ chi.ravel() = pointer.ravel()`

    Returns:
        the process matrix.

    >>> rho_in = get_rho_in("0xy1", 2)
    >>> rho_out = rho_in
    >>> trans_mat = qpt_transform_matrix("sigma", 2)
    >>> chi = qpt_cvx(rho_in, rho_out, trans_mat)
    >>> chi_ideal = np.zeros((16, 16))
    >>> chi_ideal[0, 0] = 1
    >>> np.allclose(chi, chi_ideal, atol=1e-5)
    True
    """
    chi_size = trans_mat.shape[1]
    n_basis = int(np.sqrt(chi_size))
    if n_basis**2 != chi_size:
        logger.warning(f"Cannot infer n_basis from trans_mat.shape={trans_mat.shape}")

    # Assumes uniform shape in rho_in and rho_out, and same length of rho_in and rho_out.
    n_inits = len(rho_in)
    rho_in = np.asarray(rho_in).reshape(n_inits, rho_in[0].size)
    rho_out = np.asarray(rho_out).reshape(n_inits, rho_out[0].size)

    pointer: np.matrix = np.linalg.lstsq(rho_in, rho_out, rcond=-1)[0]
    chi: np.matrix = np.linalg.lstsq(trans_mat, pointer.ravel(), rcond=-1)[0]

    try:
        chi = chi.reshape(n_basis, -1)  # No transpose as in pyle's code.
        warn_if_not_herm_unit_or_pos(chi)
        return chi
    except:
        logger.exception(
            f"Returning a flatten chi of shape={chi.shape}, "
            f"reshape failed with trans_mat.shape={trans_mat.shape} "
            f"and rho_in.shape={rho_in.shape}, rho_out.shape={rho_out.shape}"
        )
        return chi


check_tolerance = 1e-5


def warn_if_not_herm_unit_or_pos(mat: np.matrix, atol: float = None) -> None:
    """Warn if the mat is found violating any of the conditions upto the tolerance.

    >>> warn_if_not_herm_unit_or_pos([[-1, 0], [-1, -1]])  # Trigger all warnings.
    """
    if atol is None:
        atol = check_tolerance
    mat = np.asarray(mat)
    msg = []

    trace = np.trace(mat)
    if not np.allclose(trace, 1, atol=atol):
        msg.append(f"mat is not unit trace: tr(mat) = {trace}")

    diff = mat - mat.T.conj()
    if not np.allclose(diff, 0, atol=atol):
        msg.append(
            f"mat is not hermitian: max(abs(mat - mat.dagger())) = {abs(diff).max()}"
        )

    eigvals = np.linalg.eigvalsh(mat)
    if not np.all(eigvals + atol >= 0):
        msg.append(
            f"mat is not positive: eigvals include {eigvals[eigvals + atol < 0]}"
        )

    if msg:
        logger.warning("\n".join(msg))


def random_density_matrix(n_dim: int, pure: bool = False) -> np.matrix:
    """Returns a random density matrix of dimension n_dim.

    Using idea from https://physics.stackexchange.com/a/441124

    Also you can try toqito.random.random_density_matrix(n_dim) instead. See
    https://toqito.readthedocs.io/en/latest/_autosummary/toqito.random.random_density_matrix.html

    >>> random_density_matrix(4)  # doctest: +ELLIPSIS
    array([...])
    >>> rho = random_density_matrix(4, pure=True)
    >>> np.isclose(np.trace(rho @ rho), 1)
    True
    """
    if pure:
        ket = np.random.rand(n_dim) + 1j * np.random.rand(n_dim)
        ket = ket / np.linalg.norm(ket)
        rho = np.outer(ket, ket.conj())
    else:
        diags = np.random.rand(n_dim)
        diags = diags / diags.sum()
        rho = np.diagflat(diags)

    unitary = scipy.stats.unitary_group.rvs(n_dim)  # Generate a random unitary.
    rho = unitary @ rho @ unitary.T.conj()
    warn_if_not_herm_unit_or_pos(rho)
    return rho


def get_probs(
    rho: np.matrix,
    tomo_ops: Union[list[np.matrix], Literal["ixy", "oct"]] = "ixy",
) -> np.ndarray:
    """Get the probabilities of the given density matrix under the given tomo_ops.

    >>> get_probs([[1, 0], [0, 0]])
    array([1. , 0. , 0.5, 0.5, 0.5, 0.5])
    >>> get_probs([[.5, .5], [.5, .5]])
    array([0.5, 0.5, 0.5, 0.5, 0. , 1. ])
    >>> get_probs([[.5, .5j], [-.5j, .5]])
    array([0.5, 0.5, 0. , 1. , 0.5, 0.5])
    >>> get_probs([[0,0,0,0],[0,.5,.5,0],[0,.5,.5,0],[0,0,0,0]])  # Bell state.
    array([0.  , 0.5 , 0.5 , 0.  , 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
           0.25, 0.25, 0.25, 0.25, 0.25, 0.5 , 0.  , 0.  , 0.5 , 0.25, 0.25,
           0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5 ,
           0.  , 0.  , 0.5 ])
    """
    rho = np.asarray(rho)

    if tomo_ops in ["ixy", "oct"]:
        n_qbs = int(np.log2(len(rho)))
        if 2**n_qbs != len(rho):
            raise ValueError(f"rho is not a valid for tomo_ops={tomo_ops}")
        tomo_ops = qst_transform_matrix(tomo_ops, n_qbs, return_tomo_ops=True)
    elif isinstance(tomo_ops, str):
        raise ValueError(f"Unknown tomo_ops={tomo_ops}")
    else:
        pass

    probs = [np.diag(op @ rho @ op.T.conj()) for op in tomo_ops]
    return np.ravel(probs).real


@misc.cache_with_bypass()
def get_rho_in(
    init_ops: Union[list[np.matrix], Literal["0xy1"]] = "0xy1",
    n_qbs: int = 1,
) -> list[np.matrix]:
    """Return the density matrices after applying init_ops on |000...> states.

    >>> np.array(get_rho_in("0xy1"))
    array([[[ 1. +0.j ,  0. +0.j ],
            [ 0. +0.j ,  0. +0.j ]],
    <BLANKLINE>
           [[ 0.5+0.j ,  0. +0.5j],
            [-0. -0.5j,  0.5+0.j ]],
    <BLANKLINE>
           [[ 0.5+0.j ,  0.5+0.j ],
            [ 0.5+0.j ,  0.5+0.j ]],
    <BLANKLINE>
           [[ 0. +0.j ,  0. +0.j ],
            [ 0. +0.j ,  1. +0.j ]]])
    >>> np.allclose(get_rho_in("0xy1"),
    ...             get_rho_in([np.eye(2), Xpi2, Ypi2, Xpi]))
    True
    """
    if init_ops == "0xy1":
        rho_1q = [
            [[1, 0], [0, 0]],  # 0
            [[0.5, 0.5j], [-0.5j, 0.5]],  # x
            [[0.5, 0.5], [0.5, 0.5]],  # y
            [[0, 0], [0, 1]],  # 1
        ]
        rho_1q = [np.asarray(rho) for rho in rho_1q]
    elif isinstance(init_ops, str):
        raise ValueError(f"Unknown init_ops={init_ops}")
    else:
        rho0 = np.array([[1, 0], [0, 0]])
        rho_1q = [op @ rho0 @ op.T.conj() for op in init_ops]

    rho_mq = tensor_combinations(rho_1q, n_qbs)
    return rho_mq


def get_rho_out(
    rho_in: np.matrix,
    chi: np.matrix,
    basis: Union[list[np.matrix], Literal["sigma", "raise-lower"]] = "sigma",
) -> np.matrix:
    """Return density matrix after applying process chi on rho_in.

    >>> chi = np.zeros((4,4))
    >>> chi[2,2] = 1
    >>> get_rho_out([[1,0],[0,0]], chi)
    array([[0.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j]])
    >>> get_rho_out([[.5,.5],[.5,.5]], chi)
    array([[ 0.5+0.j, -0.5+0.j],
           [-0.5+0.j,  0.5+0.j]])
    >>> get_rho_out([[.5,.5j],[-.5j,.5]], chi)
    array([[0.5+0.j , 0. +0.5j],
           [0. -0.5j, 0.5+0.j ]])
    """
    rho_in = np.asarray(rho_in)
    chi = np.asarray(chi)

    if basis in ["sigma", "raise-lower"]:
        n_qbs = int(np.log2(len(rho_in)))
        if 2**n_qbs != len(rho_in):
            raise ValueError(f"rho_in is not a valid for basis={basis}")
        basis = qpt_transform_matrix(basis, n_qbs, return_basis=True)
    elif isinstance(basis, str):
        raise ValueError(f"Unknown basis={basis}")
    else:
        pass

    comps = []
    for i in range(len(basis)):
        for j in range(len(basis)):
            comp = chi[i, j] * (basis[i] @ rho_in @ basis[j].T.conj())
            comps.append(comp)
    return np.sum(comps, axis=0)


# Helper functions
def sqrtm(mat: np.matrix) -> np.matrix:
    """Compute the matrix square root of a matrix

    >>> sqrtm(np.diag([1, 4, 9]))
    array([[1.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 2.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 3.+0.j]])
    """
    mat = np.asarray(mat)
    d, U = np.linalg.eig(mat)
    s = np.sqrt(d.astype(complex))
    return U @ np.diag(s) @ U.conj().T


def fidelity(a: np.matrix, b: np.matrix) -> float:
    """Compute the fidelity between matrices a and b. See Nielsen and Chuang, p. 409

    >>> mix = [[.5,0], [0,.5]]
    >>> rho = [[1,0], [0,0]]
    >>> np.isclose(fidelity(rho, rho), 1)
    True
    >>> np.isclose(fidelity(rho, mix) ** 2, 0.5)  # different from experimentist's convention.
    True
    """
    a, b = np.asarray(a), np.asarray(b)
    a_sqrt = sqrtm(a)
    return np.real(np.trace(sqrtm(a_sqrt @ b @ a_sqrt)))


def overlap(a: np.matrix, b: np.matrix) -> float:
    """Trace of the product of two matrices.

    >>> mix = [[.5,0], [0,.5]]
    >>> rho = [[1,0], [0,0]]
    >>> np.isclose(overlap(rho, rho), 1)
    True
    >>> np.isclose(overlap(rho, mix), 0.5)
    True
    """
    a, b = np.asarray(a), np.asarray(b)
    if not np.isclose(purity(a), 1) or not np.isclose(purity(b), 1):
        logger.warning("overlap is only correct if at least one operand is a pure")
    return np.real(np.trace(a @ b))


fid_overlap = overlap  # Alias.


def trace_distance(a: np.matrix, b: np.matrix) -> float:
    """Compute the trace distance between matrices a and b. See Nielsen and Chuang, p. 403

    >>> mix = [[.5,0], [0,.5]]
    >>> rho = [[1,0], [0,0]]
    >>> np.isclose(trace_distance(rho, rho), 0)
    True
    >>> np.isclose(trace_distance(rho, mix), 0.5)
    True
    """
    a, b = np.asarray(a), np.asarray(b)
    diff = a - b
    abs = sqrtm(diff.conj().T @ diff)
    return np.real(np.trace(abs)) / 2


def purity(rho: np.matrix) -> float:
    """Compute the purity of a density matrix rho. See
    https://en.wikipedia.org/wiki/Purity_(quantum_mechanics)

    >>> np.isclose(purity([[1,0], [0,0]]), 1)
    True
    >>> np.isclose(purity([[.5,0], [0,.5]]), 0.5)
    True
    >>> mix = np.diagflat(np.ones(16)/16)
    >>> np.isclose(purity(mix), 1/16)
    True
    """
    rho = np.asarray(rho)
    return np.real(np.trace(rho @ rho))


def test_qst(list_n_qbs=(1, 2, 3), runs=100):
    from tqdm.contrib.itertools import product
    import pandas as pd
    import matplotlib.pyplot as plt

    def qst_one(tomo_ops, fit_method, n_qbs):
        rho = random_density_matrix(2**n_qbs)
        rho_fit = qst(get_probs(rho, tomo_ops), tomo_ops, fit_method)
        return np.max(np.abs(rho - rho_fit)), fidelity(rho, rho_fit) ** 2

    records = []
    for tomo_ops, fit_method, n_qbs, run in product(
        ["ixy", "oct"],
        ["cvx", "lstsq"],
        list_n_qbs,
        range(runs),
    ):
        resi, fid = qst_one(tomo_ops, fit_method, n_qbs)
        rec = dict(
            tomo_ops=tomo_ops,
            fit_method=fit_method,
            n_qbs=n_qbs,
            run=run,
            resi=resi,
            fid=fid,
        )
        records.append(rec)

    df_qst = pd.DataFrame.from_records(records)

    fig, (ax, ax2) = plt.subplots(ncols=2, sharex=True)
    fig.suptitle("qst test")
    df_qst.pivot(
        index="run", columns=["tomo_ops", "fit_method", "n_qbs"], values="resi"
    ).plot(ax=ax)
    ax.set_yscale("log")
    for l in ax.lines:
        if "cvx" in l.get_label():
            l.set_marker("+")
    ax.legend()
    ax.set_title("resi")

    df_qst.pivot(
        index="run", columns=["tomo_ops", "fit_method", "n_qbs"], values="fid"
    ).plot(ax=ax2)
    for l in ax2.lines:
        if "cvx" in l.get_label():
            l.set_marker("+")
    ax2.legend()
    ax2.set_title("fidelity")
    ax2.axhline(y=1, color="k", linestyle="--")
    return df_qst


def test_qpt(list_n_qbs=(1, 2), runs=50):
    from tqdm.contrib.itertools import product
    import pandas as pd
    import matplotlib.pyplot as plt

    def qpt_one(basis, fit_method, n_qbs):
        rho_in = get_rho_in(init_ops="0xy1", n_qbs=n_qbs)
        chi = random_density_matrix(4**n_qbs)
        rho_out = [get_rho_out(rho, chi, basis) for rho in rho_in]
        chi_fit = qpt(rho_out, rho_in, basis, fit_method)
        return np.max(np.abs(chi - chi_fit)), fidelity(chi, chi_fit) ** 2

    records = []
    for basis, fit_method, n_qbs, run in product(
        ["sigma", "raise-lower"],
        ["cvx", "lstsq"],
        list_n_qbs,
        range(runs),
    ):
        resi, fid = qpt_one(basis, fit_method, n_qbs)
        rec = dict(
            basis=basis, fit_method=fit_method, n_qbs=n_qbs, run=run, resi=resi, fid=fid
        )
        records.append(rec)

    df_qpt = pd.DataFrame.from_records(records)

    fig, (ax, ax2) = plt.subplots(ncols=2, sharex=True)
    fig.suptitle("qpt test")
    df_qpt.pivot(
        index="run", columns=["basis", "fit_method", "n_qbs"], values="resi"
    ).plot(ax=ax)
    ax.set_yscale("log")
    for l in ax.lines:
        if "cvx" in l.get_label():
            l.set_marker("+")
    ax.legend()
    ax.set_title("resi")

    df_qpt.pivot(
        index="run", columns=["basis", "fit_method", "n_qbs"], values="fid"
    ).plot(ax=ax2)
    for l in ax2.lines:
        if "cvx" in l.get_label():
            l.set_marker("+")
    ax2.legend()
    ax2.set_title("fidelity")
    ax2.axhline(y=1, color="k", linestyle="--")
    return df_qpt


def _more_tests():
    """
    >>> import labcodes.frog.pyle_tomo as pyle  # TODO: remove this dependency.
    >>> np.allclose(qst_transform_matrix("ixy", 1), pyle._qst_transforms['tomo'][1])
    True
    >>> np.allclose(qst_transform_matrix("ixy", 2), pyle._qst_transforms['tomo2'][1])
    True
    >>> np.allclose(qst_transform_matrix("ixy", 3), pyle._qst_transforms['tomo3'][1])
    True
    >>> np.allclose(qst_transform_matrix("oct", 1), pyle._qst_transforms['octomo'][1])
    True
    >>> np.allclose(qst_transform_matrix("oct", 2), pyle._qst_transforms['octomo2'][1])
    True
    >>> np.allclose(qst_transform_matrix("oct", 3), pyle._qst_transforms['octomo3'][1])
    True

    >>> import labcodes.frog.state_list as state_list  # TODO: remove this dependency.
    >>> np.allclose(state_list.rho_in, get_rho_in("0xy1", 2))
    True

    >>> np.allclose(qpt_transform_matrix("sigma", 1), pyle._qpt_transforms['sigma'][1])
    True
    >>> np.allclose(qpt_transform_matrix("sigma", 2), pyle._qpt_transforms['sigma2'][1])
    True
    >>> np.allclose(qpt_transform_matrix("raise-lower", 1), pyle._qpt_transforms['raise-lower'][1])
    True
    >>> np.allclose(qpt_transform_matrix("raise-lower", 2), pyle._qpt_transforms['raise-lower2'][1])
    True
    """


if __name__ == "__main__":
    import doctest

    import cvxpy  # Let the import warnings emit here.

    doctest.testmod()

    test_qst()
    test_qpt()
    import matplotlib.pyplot as plt

    plt.show()
