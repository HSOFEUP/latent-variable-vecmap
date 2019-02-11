import time

import torch
from torch.autograd import Variable

from sparsemap.layers_pt.matching_layer import MatchingSparseMarginals
from sparsemap._sparsemap import sparsemap_fwd
from sparsemap._factors import PFactorMatching, PFactorMatchingSparse
from smoothot import dual_solvers as ot
from scipy import sparse as sp


def sparsemap_matching_fwd(X, max_iter=10, verbose=0):
    match = PFactorMatching()
    match.initialize(*X.shape)
    # method on sparsemap branch memory_efficient_fwd; rebuild source
    u, _, status = sparsemap_fwd(match, X.ravel(), [],
                                 max_iter=max_iter,
                                 verbose=verbose)
    return u.reshape(*X.shape)


def sparsemap_sparse_matching(X, max_iter=10, verbose=0):
    Xsp = sp.csr_matrix(X)
    factor_sparse = PFactorMatchingSparse()
    factor_sparse.initialize(Xsp)
    logp = -Xsp.data
    val, u, _ = factor_sparse.solve_map(logp, [])
    U = Xsp.copy()
    U.data[:] = u
    return U.toarray()


def sparse_vecmap(xp, sims, max_iter, solver='sparsemap_fwd', gamma=1.0):
    start = time.time()
    print(f'Using solver {solver} for matching')
    if solver == 'ot':
        C = -sims  # "cost" matrix
        regul = ot.SquaredL2(gamma=gamma)
        a = b = np.ones(sims.shape[0])
        alpha, beta = ot.solve_dual(a, b, C, regul, max_iter=1000, tol=1e-8)
        matching = ot.get_plan_from_dual(alpha, beta, C, regul)
    elif solver == 'sparsemap_fwd':
        matching = sparsemap_matching_fwd(sims, max_iter=max_iter, verbose=True)
    elif solver == 'sparsemap':
        sims = torch.tensor(sims)
        scores = Variable(sims, requires_grad=True)
        matcher = MatchingSparseMarginals(max_iter=max_iter)
        matching = matcher(scores)
        matching.sum().backward()
        matching = matching.detach().numpy()
    elif solver == 'sparsemap_sparse':
        matching = sparsemap_sparse_matching(sims, max_iter=max_iter, verbose=True)
    else:
        raise ValueError(f'{solver} is not a valid solver.')
    print(f'Calling matcher took {int(time.time() - start)}s.')

    indices = xp.array(np.where(matching > 0))
    print('Indices:', indices)

    # print("dpost_dunary", scores.grad)
    src_indices, trg_indices = indices
    print('# of src indices:', len(src_indices))
    return src_indices, trg_indices


if __name__ == '__main__':
    import numpy as np

    X = np.random.randn(5, 5)
    U = sparsemap_matching_fwd(X)
    np.set_printoptions(precision=2, suppress=True)

    print(X)
    print(U)
    print()

    # only works with square matrices
    C = -X  # "cost" matrix
    regul = ot.SquaredL2(gamma=1.0)
    a = b = np.ones(5)

    alpha, beta = ot.solve_dual(a, b, C, regul, max_iter=1000, tol=1e-8)
    T = ot.get_plan_from_dual(alpha, beta, C, regul)

    print(T)
    print(T.sum(axis=0))
    print(T.sum(axis=1))
