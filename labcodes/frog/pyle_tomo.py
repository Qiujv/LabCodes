"""Tomo scripts provided by Ziyu Tao, originally from Martinis groud pyle."""

# %%
import itertools
# import pdb
import numpy as np
from scipy import optimize
from scipy.linalg import expm
from functools import reduce
import math
import numpy as np


# some helper functions for dealing with matrices and computing fidelity
def tensor(matrices):
    """Compute the tensor product of a list (or array) of matrices"""
    return reduce(np.kron, matrices)

def dots(matrices):
    """Compute the dot product of a list (or array) of matrices"""
    return reduce(np.dot, matrices)

def dot3(A, B, C):
    """Compute the dot product of three matrices"""
    return np.dot(np.dot(A, B), C)

def commutator(A, B):
    """Compute the commutator of two matrices"""
    return np.dot(A, B) - np.dot(B, A)

def lindblad(rho, L, Ld=None):
    """Compute the contribution of one Lindblad term to the master equation""" 
    if Ld is None:
        Ld = L.conj().T
    return dot3(L, rho, Ld) - 0.5*dot3(Ld, L, rho) - 0.5*dot3(rho, Ld, L)

def ket2rho(ket):
    """Convert a state (ket) to a density matrix (rho)."""
    return np.outer(ket, ket.conj())

def sqrtm(A):
    """Compute the matrix square root of a matrix"""
    d, U = np.linalg.eig(A)
    s = np.sqrt(d.astype(complex))
    return dot3(U, np.diag(s), U.conj().T)

def trace_distance(rho, sigma):
    """Compute the trace distance between matrices rho and sigma
    See Nielsen and Chuang, p. 403
    """
    A = rho - sigma
    abs = sqrtm(np.dot(A.conj().T, A))
    return np.real(np.trace(abs)) / 2.0

def fidelity(rho, sigma):
    """Compute the fidelity between matrices rho and sigma
    See Nielsen and Chuang, p. 409
    """
    rhosqrt = sqrtm(rho)
    return np.real(np.trace(sqrtm(dot3(rhosqrt, sigma, rhosqrt))))

def overlap(rho, sigma):
    """Trace of the product of two matrices."""
    # XXX this is only correct if at least one of rhos or sigma is a pure state
    return np.trace(np.dot(rho, sigma))

def concurrence(rho):
    """Concurrence of a two-qubit density matrix.
    see http://qwiki.stanford.edu/wiki/Entanglement_of_Formation
    """
    yy = np.array([[ 0, 0, 0,-1],
                   [ 0, 0, 1, 0],
                   [ 0, 1, 0, 0],
                   [-1, 0, 0, 0]], dtype=complex)
    m = dots([rho, yy, rho.conj(), yy])
    eigs = [np.abs(e) for e in np.linalg.eig(m)[0]]
    e = [np.sqrt(x) for x in sorted(eigs, reverse=True)]
    return max(0, e[0] - e[1] - e[2] - e[3])

def eof(rho):
    """Entanglement of formation of a two-qubit density matrix.
    see http://qwiki.stanford.edu/wiki/Entanglement_of_Formation
    """
    def h(x):
        if x <= 0 or x >= 1:
            return 0
        return -x*np.log2(x) - (1-x)*np.log2(1-x)
    C = concurrence(rho)
    arg = max(0, np.sqrt(1-C**2))
    return h((1 + arg)/2.0)

# define some useful matrices
def Rmat(axis, angle):
    return expm(-1j*angle/2.0*axis)

sigmaI = np.eye(2, dtype=complex)
sigmaX = np.array([[0, 1], [1, 0]], dtype=complex)
sigmaY = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigmaZ = np.array([[1, 0], [0, -1]], dtype=complex)

sigmaP = (sigmaX - 1j*sigmaY)/2
sigmaM = (sigmaX + 1j*sigmaY)/2

Xpi2 = Rmat(sigmaX, np.pi/2)
Ypi2 = Rmat(sigmaY, np.pi/2)
Zpi2 = Rmat(sigmaZ, np.pi/2)

Xpi = Rmat(sigmaX, np.pi)
Ypi = Rmat(sigmaY, np.pi)
Zpi = Rmat(sigmaZ, np.pi)

Xmpi2 = Rmat(sigmaX, -np.pi/2)
Ympi2 = Rmat(sigmaY, -np.pi/2)
Zmpi2 = Rmat(sigmaZ, -np.pi/2)

Xmpi = Rmat(sigmaX, -np.pi)
Ympi = Rmat(sigmaY, -np.pi)
Zmpi = Rmat(sigmaZ, -np.pi)


def qst(diags, U, return_all=False):
    """Convert a set of diagonal measurements into a density matrix.
    
    Args:
        diags: 2d array with shape n_ops * n_stats = [I, X/2, Y/2]**n_qs * [0, 1]**n_qs
        label order in nested-loop style, i.e. 001, 010, **011** instead of 001, 010, **100**.
        - for single qubit QST, diags = [
                [p0_I, p1_I],
                [p0_X, p1_X],
                [p0_Y, p1_Y],
            ]
        - for two qubit qst, diags = [
                [p00_II, p01_II, p10_II, p11_II],
                [p00_IX, p01_IX, p10_IX, p11_IX],
                ...
                [p00_YY, p01_YY, p10_YY, p11_YY],
            ]
        
        measured probabilities (diagonal elements) after acting on the state 
        with each of the unitaries from the qst protocol
    
        U: transformation matrix from init_qst for this protocol, or 
            key passed to init_qst under which the transformation was saved

    Returns:
        Density matrix rho, with shape: n_stats * n_stats = [0, 1]**n_qs.
    """  # TODO: Confirm if the order is in nested-loop like.
    if isinstance(U, str) and U in _qst_transforms:
        U = _qst_transforms[U][1]
    
    diags = np.asarray(diags)
    N = diags.shape[1]
    rhoFlat, resids, rank, s = np.linalg.lstsq(U, diags.flatten(), rcond=-1)
    if return_all:
        return rhoFlat.reshape((N, N)), resids, rank, s
    else:
        return rhoFlat.reshape((N, N))


def qst_mle(pxms, Us, F=None, rho0=None):
    """State tomography with maximum-likelihood estimation.
    
    Args: 
        pxms: a 2D array of measured probabilites.  The first index indicates which
               operation from Us was applied, while the second index tells which measurement
               result this was (e.g. 000, 001, etc.).
              
        Us: the unitary operations that were applied to the system before measuring.
        F: a 'fidelity' matrix, relating the actual or 'intrinsic' probabilities to the
            measured probabilites, via pms = dot(F, pis).  If no fidelity matrix is given,
            the identity will be used.
        rho0: an initial guess for the density matrix, e.g. from linear tomography.
    """
    if isinstance(Us, str) and Us in _qst_transforms:
        Us = _qst_transforms[Us][0]

    N = len(Us[0]) # size of density matrix
    
    if F is None:
        F = np.eye(N)
    
    try:
        indices_re = np.tril_indices(N)
        indices_im = np.tril_indices(N, -1)
    except AttributeError:
        # tril_indices is new in numpy 1.4.0
        indices_re = (np.hstack([[k]*(k+1) for k in range(N)]),
                      np.hstack([list(range(k+1)) for k in range(N)]))
        indices_im = (np.hstack([[k+1]*(k+1) for k in range(N-1)]),
                      np.hstack([list(range(k+1)) for k in range(N-1)]))
    n_re = len(indices_re[0])  # N*(N+1)/2
    n_im = len(indices_im[0])  # N*(N-1)/2
    
    def make_T(tis):
        T = np.zeros((N,N), dtype=complex)
        T[indices_re] = tis[:n_re]
        T[indices_im] += 1j*tis[n_re:]
        return T
    
    def unmake_T(T):
        return np.hstack((T[indices_re].real, T[indices_im].imag))
    
    def make_rho(ts):
        T = make_T(ts)
        TT = np.dot(T.conj().T, T)
        return TT / np.trace(TT)
    
    # make an initial guess using linear tomography
    if rho0 is None:
        T = init_qst(Us)
        Finv = np.linalg.inv(F)
        pis_guess = np.array([np.dot(Finv, p) for p in pxms])
        rho0 = qst(pis_guess, T)
    
    # convert the initial guess into t vector
    # to do this we use a cholesky decomposition, which
    # only works if the matrix is positive and hermitian.
    # so, we diagonalize and fix up the eigenvalues before
    # attempting the cholesky decomp.
    d, V = np.linalg.eig(rho0)
    d = d.real
    d = d*(d > 0) + 0.01
    dfix = d / sum(d)
    rho0 = dot3(V, np.diag(dfix), V.conj().T)
    T0 = np.linalg.cholesky(rho0)
    tis_guess = unmake_T(T0)
    
    # precompute conjugate transposes of matrices
    UUds = [(U, U.conj().T) for U in Us]
    
    def log(x):
        """Safe version of log that returns -Inf when x < 0, rather than NaN.
        
        This is good for our purposes since negative probabilities are infinitely unlikely.
        """
        return np.log(x.real * (x.real > 0))
    
    array = np.array
    dot = np.dot
    diag = np.diag
        
    def unlikelihood(tis): # negative of likelihood function
        rho = make_rho(tis)
        pxis = array([dot(F, diag(dot3(U, rho, Ud))) for U, Ud in UUds])
        terms = pxms * log(pxis) + (1-pxms) * log(1-pxis)
        return -sum(terms.flat)
    
    #minimize
    tis = optimize.fmin(unlikelihood, tis_guess)
    #tis = optimize.fmin_bfgs(unlikelihood, tis_guess)
    return make_rho(tis)

def rst_mle(pxms, Us, F=None, rho0=None):  # for resonator state?
    """State tomography with maximum-likelihood estimation.
    Args:
        pxms: a 2D array of measured probabilites.  The first index indicates which 
        operation from Us was applied, while the second index tells which measurement
        result this was (e.g. 000, 001, etc.).
              
        Us: the unitary operations that were applied to the system before measuring.
        F: a 'fidelity' matrix, relating the actual or 'intrinsic' probabilities to the
        measured probabilites, via pms = dot(F, pis).  If no fidelity matrix is given,
        the identity will be used.
        
        rho0: an initial guess for the density matrix, e.g. from linear tomography.
    """
    N = len(Us[0]) # size of density matrix
    
    if F is None:
        F = np.eye(N)
    
    try:
        indices_re = np.tril_indices(N)
        indices_im = np.tril_indices(N, -1)
    except AttributeError:
        # tril_indices is new in numpy 1.4.0
        indices_re = (np.hstack([[k]*(k+1) for k in range(N)]),
                      np.hstack([list(range(k+1)) for k in range(N)]))
        indices_im = (np.hstack([[k+1]*(k+1) for k in range(N-1)]),
                      np.hstack([list(range(k+1)) for k in range(N-1)]))
    n_re = len(indices_re[0]) # N*(N+1)/2
    n_im = len(indices_im[0]) # N*(N-1)/2
    
    def make_T(tis):
        T = np.zeros((N,N), dtype=complex)
        T[indices_re] = tis[:n_re]
        T[indices_im] += 1j*tis[n_re:]
        return T
    
    def unmake_T(T):
        return np.hstack((T[indices_re].real, T[indices_im].imag))
    
    def make_rho(ts):
        T = make_T(ts)
        TT = np.dot(T.conj().T, T)
        sU = np.shape(Us[0])
        resonatorLev = np.sqrt(sU[0])
        sT = np.shape(TT)
        TTL = np.zeros(sU,dtype=complex)
        resonatorLevSolv = np.sqrt(sT[0])
        for ir0 in range(resonatorLevSolv):
            for ir1 in range(resonatorLevSolv):
                for ic0 in range(resonatorLevSolv):
                    for ic1 in range(resonatorLevSolv):
                        TTL[ir0*resonatorLev+ir1,ic0*resonatorLev+ic1] = TT[ir0*resonatorLevSolv+ir1,ic0*resonatorLevSolv+ic1]

        return TTL / np.trace(TT)
    
    # make an initial guess using linear tomography
    if rho0 is None:
        T = init_qst(Us)
        Finv = np.linalg.inv(F)
        pis_guess = np.array([np.dot(Finv, p) for p in pxms])
        rho0 = qst(pis_guess, T)
    
    # convert the initial guess into t vector
    # to do this we use a cholesky decomposition, which
    # only works if the matrix is positive and hermitian.
    # so, we diagonalize and fix up the eigenvalues before
    # attempting the cholesky decomp.
    d, V = np.linalg.eig(rho0)
    d = d.real
    d = d*(d > 0) + 0.01
    dfix = d / sum(d)
    rho0 = dot3(V, np.diag(dfix), V.conj().T)
    T0 = np.linalg.cholesky(rho0)
    tis_guess = unmake_T(T0)
    
    # precompute conjugate transposes of matrices
    UUds = [(U, U.conj().T) for U in Us]
    
    def log(x):
        """Safe version of log that returns -Inf when x < 0, rather than NaN.
        
        This is good for our purposes since negative probabilities are infinitely unlikely.
        """
        return np.log(x.real * (x.real > 0))
    
    array = np.array
    dot = np.dot
    diag = np.diag
        
    def unlikelihood(tis): # negative of likelihood function
        rho = make_rho(tis)
        pxis = array([dot(F, diag(dot3(U, rho, Ud))) for U, Ud in UUds])
        terms = pxms * log(pxis) + (1-pxms) * log(1-pxis)
        return -sum(terms.flat)
    
    #minimize
    tis = optimize.fmin(unlikelihood, tis_guess)
    #tis = optimize.fmin_bfgs(unlikelihood, tis_guess)
    return make_rho(tis)

def qpt(rhos, Erhos, T, return_all=False):
    """Calculate the chi matrix of a quantum process.
    
    rhos - array of input density matrices
    Erhos - array of output density matrices
    
    T - transformation matrix from init_qpt for the desired operator
        basis, or key passed to init_qpt under which this basis was saved
    """
    chi_pointer = qpt_pointer(rhos, Erhos)
    return transform_chi_pointer(chi_pointer, T, return_all)


def transform_chi_pointer(chi_pointer, T, return_all=False):
    """Convert a chi matrix from the pointer basis into a different basis.
    
    transform_chi_pointer(chi_pointer, As) will transform the chi_pointer matrix
    from the pointer basis (as produced by qpt_pointer, for example) into the
    basis specified by operator elements in the cell array As.
    """
    if T in _qpt_transforms:
        T = _qpt_transforms[T][1]

    _Din, Dout = chi_pointer.shape
    chi_flat, resids, rank, s = np.linalg.lstsq(T, chi_pointer.flatten(), rcond=-1)
    chi = chi_flat.reshape((Dout, Dout))
    if return_all:
        return chi, resids, rank, s
    else:
        return chi 


def qpt_pointer(rhos, Erhos, return_all=False):
    """Calculates the pointer-basis chi-matrix.
    
    rhos - array of input density matrices
    Erhos - array of output density matrices.
    
    Uses linalg.lstsq to calculate the closest fit
    when the chi-matrix is overdetermined by the data.
    The return_all flag specifies whether to return
    all the parameters returned from linalg.lstsq, such
    as the residuals and the rank of the chi-matrix.  By
    default (return_all=False) only chi is returned.
    """

    # the input and output density matrices can have different
    # dimensions, although this will rarely be the case for us.
    Din = rhos[0].size
    Dout = Erhos[0].size
    n = len(rhos)

    # reshape the input and output density matrices
    # each row of the resulting matrix has a flattened
    # density matrix (in or out, respectively)
    rhos_mat = np.asarray(rhos).reshape((n, Din))
    Erhos_mat = np.asarray(Erhos).reshape((n, Dout))

    chi, resids, rank, s = np.linalg.lstsq(rhos_mat, Erhos_mat, rcond=-1)
    if return_all:
        return chi, resids, rank, s
    else:
        return chi



def init_qst(Us, key=None):
    """Initialize quantum state tomography for a set of unitaries.
    
    Us - a list of unitary operations that will be applied to the
        state before measuring the diagonal elements.  These unitaries
        should form a 'complete' set to allow the full density matrix
        to be determined, though this is not enforced.

    key - (optional) a dictionary key under which this tomography
        protocol will be stored so it can be referred to without
        recomputing the transformation matrix.
    
    Returns a transformation matrix that should be passed to qst along
    with measurement data to perform the state tomography.
    """
    
    Us = np.asarray(Us)
    
    M = len(Us) # number of different measurements
    N = len(Us[0]) # number of states (= number of diagonal elements)
    
    # we have to be a bit careful here, because things blow up
    # exponentially with the number of qubits. The first method
    # uses direct indexing to generate the entire transform matrix
    # in one shot. This is elegant and much faster than for-loop
    # iteration, but uses more memory and so only works for
    # smaller qubit numbers.
    if N <= 16:
        # 1-4 qubits
        def transform(K, L):
            i, j = divmod(K, N)
            m, n = divmod(L, N)
            return Us[i, j, m] * Us[i, j, n].conj()
        U = np.fromfunction(transform, (M*N, N**2), dtype=int)
    else:
        # 5+ qubits
        U = np.zeros((M*N, N**2), dtype=complex)
        for K in range(M*N):
            for L in range(N**2):
                i, j = divmod(K, N)
                m, n = divmod(L, N)
                U[K, L] = Us[i, j, m] * Us[i, j, n].conj()
    
    # save this transform if a key was provided
    if key is not None:
        _qst_transforms[key] = (Us, U)
    
    return U


def init_qpt(As, key=None):
    """Initialize quantum process tomography for an operator basis.
    
    As - a list of matrices giving the basis in which to compute
        the chi matrix for process tomography.  These matrices
        should form a 'complete' set to allow the full chi matrix
        to be represented, though this is not enforced.

    key - (optional) a dictionary key under which this tomography
        protocol will be stored so it can be referred to without
        recomputing the transformation matrix.
    
    Returns a transformation matrix that should be passed to qpt along
    with input and output density matrices to perform the process tomography.
    """
    
    As = np.asarray(As, dtype=complex)
    
    Dout, Din = As[0].shape
    chiSize = Dout*Din
    
    # we have to be a bit careful here, because things blow up
    # exponentially with the number of qubits.  The first method
    # uses direct indexing to generate the entire transform matrix
    # in one shot.  This is elegant and much faster than for-loop
    # iteration, but uses more memory and so only works for
    # smaller qubit numbers.
    if chiSize <= 16:
        # one or two qubits
        def transform(alpha, beta):
            L, J = divmod(alpha, chiSize)
            M, N = divmod(beta, chiSize)
            i, j = divmod(J, Dout)
            k, l = divmod(L, Din)
            return As[M, i, k] * As[N, j, l].conj()
        T = np.fromfunction(transform, (chiSize**2, chiSize**2), dtype=int)
    else:
        # three or more qubits
        T = np.zeros((chiSize**2, chiSize**2), dtype=complex)
        for alpha in range(chiSize**2):
            for beta in range(chiSize**2):
                L, J = divmod(alpha, chiSize)
                M, N = divmod(beta, chiSize)
                i, j = divmod(J, Dout)
                k, l = divmod(L, Din)
                T[alpha, beta] = As[M, i, k] * As[N, j, l].conj()
    
    if key is not None:
        _qpt_transforms[key] = (As, T)
    
    return T

def tensor_combinations(matrices, repeat):
    return [tensor(ms) for ms in itertools.product(matrices, repeat=repeat)]

# store all initialized tomography protocols
_qst_transforms = {}
_qpt_transforms = {}


# standard single-qubit QST protocols

tomo_ops = [np.eye(2), Xpi2, Ypi2]
octomo_ops = [np.eye(2), Xpi2, Ypi2, Xmpi2, Ympi2, Xpi]

init_qst(tomo_ops, 'tomo')
init_qst(octomo_ops, 'octomo')

init_qst(tensor_combinations(tomo_ops, 2), 'tomo2')
init_qst(tensor_combinations(octomo_ops, 2), 'octomo2')

init_qst(tensor_combinations(tomo_ops, 3), 'tomo3')
init_qst(tensor_combinations(octomo_ops, 3), 'octomo3')

#init_qst([tensor(ops) for ops in itertools.product(tomo_ops, repeat=4)], 'tomo4')
#init_qst([tensor(ops) for ops in itertools.product(octomo_ops, repeat=4)], 'octomo4')


# standard QPT protocols

sigma_basis = [np.eye(2), sigmaX, sigmaY, sigmaZ]
raise_lower_basis = [np.eye(2), sigmaP, sigmaM, sigmaZ]

init_qpt(sigma_basis, 'sigma')
init_qpt(raise_lower_basis, 'raise-lower')

init_qpt(tensor_combinations(sigma_basis, 2), 'sigma2')
init_qpt(tensor_combinations(raise_lower_basis, 2), 'raise-lower2')

# takes A LOT of memory!
#init_qpt(tensor_combinations(sigma_basis, 3), 'sigma3')
#init_qpt(tensor_combinations(raise_lower_basis, 3), 'raise-lower3')


## tests

def test_qst(n=100):
    """Generate a bunch of random states, and check that
    we recover them from state tomography.
    """

    def test_qst_protocol(proto):
        Us = _qst_transforms[proto][0]
        # pdb.set_trace()
        rho = (np.random.uniform(-1, 1, Us[0].shape) +
            1j*np.random.uniform(-1, 1, Us[0].shape))
        diags = np.vstack([np.diag(dot3(U, rho, U.conj().T)) for U in Us])
        # pdb.set_trace()
        rhoCalc = qst(diags, proto)
        return np.max(np.abs(rho - rhoCalc))
    
    # 1 qubit
    et1 = [test_qst_protocol('tomo') for _ in range(n)]
    eo1 = [test_qst_protocol('octomo') for _ in range(n)]
    print('1 qubit max error: tomo=%g, octomo=%g' % (max(et1), max(eo1)))
     
    # 2 qubits
    et2 = [test_qst_protocol('tomo2') for _ in range(n//2)]
    eo2 = [test_qst_protocol('octomo2') for _ in range(n//2)]
    print('2 qubits max error: tomo2=%g, octomo2=%g' % (max(et2), max(eo2)))

    # 3 qubits
    et3 = [test_qst_protocol('tomo3') for _ in range(n//10)]
    eo3 = [test_qst_protocol('octomo3') for _ in range(n//10)]
    print('3 qubits max error: tomo3=%g, octomo3=%g' % (max(et3), max(eo3)))
    
    # 4 qubits
    #et4 = [testQstProtocol('tomo4') for _ in range(2)]
    #eo4 = [testQstProtocol('octomo4') for _ in range(2)]
    #print '4 qubits max error: tomo4=%g, octomo4=%g' % (max(et4), max(eo4))
    
    
def test_qpt(n=1):
    """Generate a random chi matrix, and check that we
    recover it from process tomography.
    """
    def operate(rho, chi, As):
        return sum(chi[m, n] * dot3(As[m], rho, As[n].conj().T)
                   for m in range(len(As)) for n in range(len(As)))
    
    def test_qpt_protocol(proto):
        As = _qpt_transforms[proto][0]
        s = As.shape[1]
        N = len(As)
        chi = (np.random.uniform(-1, 1, (N, N)) +
            1j*np.random.uniform(-1, 1, (N, N)))
        
        # create input density matrices from a bunch of rotations
        ops = [np.eye(2), Xpi2, Ypi2, Xmpi2]
        Nqubits = int(math.log(s, 2))
        Us = tensor_combinations(ops, Nqubits)
        rho = np.zeros((s, s))
        rho[0, 0] = 1
        rhos = [dot3(U, rho, U.conj().T) for U in Us]
        
        # apply operation to all inputs
        Erhos = [operate(rho, chi, As) for rho in rhos]
        
        # calculate chi matrix and compare to actual
        chiCalc = qpt(rhos, Erhos, proto)
        return np.max(np.abs(chi - chiCalc))
    
    # 1 qubit
    errs = [test_qpt_protocol('sigma') for _ in range(n)]
    print('sigma max error:', max(errs))
    
    errs = [test_qpt_protocol('raise-lower') for _ in range(n)]
    print('raise-lower max error:', max(errs))
    
    # 2 qubits
    errs = [test_qpt_protocol('sigma2') for _ in range(n)]
    print('sigma2 max error:', max(errs))
    
    errs = [test_qpt_protocol('raise-lower2') for _ in range(n)]
    print('raise-lower2 max error:', max(errs))
    
    # 3 qubits
    #from datetime import datetime
    #start = datetime.now()
    #errs = [test_qpt_protocol('sigma3') for _ in range(n)]
    #print 'sigma3 max error:', max(errs)
    #print 'elapsed:', datetime.now() - start
    
    #errs = [test_qpt_protocol('raise-lower3') for _ in range(n)]
    #print 'raise-lower3 max error:', max(errs)


if __name__ == '__main__':
    print('Testing state tomography...')
    test_qst(10)
    
    print()
    print('Testing process tomography...')
    test_qpt()

