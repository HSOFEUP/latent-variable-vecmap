import numpy as np
from lap import lapmod


def lat_var(xp, sims, n_similar, n_repeats, batch_size, asym):
    """
    Run the matching in the E-step of the latent-variable model.
    :param xp: numpy or cupy, depending whether we run on CPU or GPU.
    :param xw: the xw matrix
    :param zw: the zw matrix
    :param sims: an matrix of shape (src_size, trg_size) where the similarity values
                 between each source word and target words are stored
    :param best_sim_forward: an array of shape (src_size), which stores the best similarity
                             scores for each
    :param n_similar:
    :param n_repeats:
    :param batch_size:
    :param asym:
    :return:
    """
    src_size = sims.shape[0]
    cc = np.empty(src_size * n_similar)  # 1D array of all finite elements of the assignement cost matrix
    kk = np.empty(src_size * n_similar)  # 1D array of the column indices. Must be sorted within one row.
    ii = np.empty((src_size * n_repeats + 1,), dtype=int)   # 1D array of indices of the row starts in cc.
    ii[0] = 0
    # if each src id should be matched to trg id, then we need to double the source indices
    for i in range(1, src_size * n_repeats + 1):
        ii[i] = ii[i - 1] + n_similar
    for i in range(0, src_size, batch_size):
        # j = min(x.shape[0], i + batch_size)
        j = min(i + batch_size, src_size)
        sim = sims[i:j]

        trg_indices = xp.argpartition(sim, -n_similar)[:, -n_similar:]  # get indices of n largest elements
        if xp != np:
            trg_indices = xp.asnumpy(trg_indices)
        trg_indices.sort()  # sort the target indices

        trg_indices = trg_indices.flatten()
        row_indices = np.array([[i] * n_similar for i in range(j-i)]).flatten()
        sim_scores = sim[row_indices, trg_indices]
        costs = 1 - sim_scores
        if xp != np:
            costs = xp.asnumpy(costs)
        cc[i * n_similar:j * n_similar] = costs
        kk[i * n_similar:j * n_similar] = trg_indices
    if n_repeats > 1:
        # duplicate costs and target indices
        new_cc = cc
        new_kk = kk
        for i in range(1, n_repeats):
            new_cc = np.concatenate([new_cc, cc], axis=0)
            if asym == '1:2':
                # for 1:2, we don't duplicate the target indices
                new_kk = np.concatenate([new_kk, kk], axis=0)
            else:
                # update target indices so that they refer to new columns
                new_kk = np.concatenate([new_kk, kk + src_size * i], axis=0)
        cc = new_cc
        kk = new_kk
    # trg indices are targets assigned to each row id from 0-(n_rows-1)
    cost, trg_indices, _ = lapmod(src_size * n_repeats, cc, ii, kk)
    src_indices = np.concatenate([np.arange(src_size)] * n_repeats, 0)
    src_indices, trg_indices = xp.asarray(src_indices), xp.asarray(trg_indices)

    # remove the pairs in which a source word was connected to a target
    # which was not one of its k most similar words
    wrong_inds = []
    for i, trgind in enumerate(trg_indices):
        krow = ii[i]
        candidates = kk[krow:krow + n_similar]
        if trgind not in candidates:
            wrong_inds.append(i)
    trg_indices = np.delete(trg_indices, wrong_inds)
    src_indices = np.delete(src_indices, wrong_inds)

    for i in range(len(src_indices)):
        src_idx, trg_idx = src_indices[i], trg_indices[i]
        # we do this if args.n_repeats > 0 to assign the target
        # indices in the cost matrix to the correct idx
        while trg_idx >= src_size:
            # if we repeat, we have indices that are > n_rows
            trg_idx -= src_size
            trg_indices[i] = trg_idx
    return src_indices, trg_indices
