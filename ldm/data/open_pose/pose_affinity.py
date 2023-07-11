__all__ = ["pose_affinity"]

import numpy as np
import math

_LIMB_SEQ = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]  # type: ignore
_MAP_IDX = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]]  # type: ignore


def pose_affinity(threshold_1, height, affinity, all_peaks):
    # find connection in the specified sequence, center 29 is in the position 15

    # the middle joints heatmap correspondence

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(_MAP_IDX)):
        score_mid = affinity[:, :, [x - 19 for x in _MAP_IDX[k]]]
        candA = all_peaks[_LIMB_SEQ[k][0] - 1]
        candB = all_peaks[_LIMB_SEQ[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = _LIMB_SEQ[k]
        if nA != 0 and nB != 0:
            connection_candidate = []

            # FIXME: Parallelize over ij.
            for i in range(nA):
                for j in range(nB):

                    vec = np.asarray(candB[j][:2]) - np.asarray(candA[i][:2])  # type: ignore
                    norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2) + 1e-8
                    vec = vec / norm

                    startend_y = np.round(
                        np.linspace(candA[i][1], candB[j][1], num=mid_num)
                    ).astype(int)

                    startend_x = np.round(
                        np.linspace(candA[i][0], candB[j][0], num=mid_num)
                    ).astype(int)

                    vec_xy = score_mid[startend_y, startend_x, :2]

                    score_midpts = vec_xy[:, 0] * vec[0] + vec_xy[:, 1] * vec[1]

                    score_with_dist_prior = score_midpts.mean(0).sum(0)
                    score_with_dist_prior += min(0.5 * height / norm - 1, 0)

                    criterion1 = (
                        np.nonzero(score_midpts > threshold_1)[0].shape[0]
                        > 0.8 * score_midpts.shape[0]
                    )
                    criterion2 = score_with_dist_prior > 0

                    if criterion1 and criterion2:
                        connection_candidate.append(
                            [
                                i,
                                j,
                                score_with_dist_prior,
                                score_with_dist_prior
                                + candA[i][2]
                                + candB[j][2],
                            ]
                        )

            connection_candidate = sorted(
                connection_candidate, key=lambda x: x[2], reverse=True
            )
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if i not in connection[:, 3] and j not in connection[:, 4]:
                    connection = np.vstack(
                        [connection, [candA[i][3], candB[j][3], s, i, j]]
                    )
                    if len(connection) >= min(nA, nB):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(_MAP_IDX)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(_LIMB_SEQ[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if (
                        subset[j][indexA] == partAs[i]
                        or subset[j][indexB] == partBs[i]
                    ):
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if subset[j][indexB] != partBs[i]:
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += (
                            candidate[partBs[i].astype(int), 2]
                            + connection_all[k][i][2]
                        )
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = (
                        (subset[j1] >= 0).astype(int)
                        + (subset[j2] >= 0).astype(int)
                    )[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += subset[j2][:-2] + 1
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += (
                            candidate[partBs[i].astype(int), 2]
                            + connection_all[k][i][2]
                        )

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = (
                        sum(candidate[connection_all[k][i, :2].astype(int), 2])
                        + connection_all[k][i][2]
                    )
                    subset = np.vstack([subset, row])
    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
    # candidate: x, y, score, id
    return candidate, subset

