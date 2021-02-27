"""
Compute mutual information and other relevant information metrics for two random variables.
"""

__author__ = "Augustin Burchell, Hörmet Yiltiz"
__copyright__ = "Copyright (C) 2021 Hörmet Yiltiz"
__license__ = "GNU GPL version 3 or later"
__version__ = "0.9"

from itertools import product
import math
import numpy as np

# TODO: add `hypothesis` property tests for mutual information and entropy


def compute(X: list, Y: list, domain: list):
    """Computes the mutual information between two random variables, defined as:
    sum pXY log(pXY/(pX pY)) in bits. Returns a tuple that contains:
      - mutual information
      - mutual information normalized by the entropy of X (useful if X is
        signal)
      - a nested dictionary that contains other normalized variants of mutual
        information, as well as some commonly used information theoretic
        metrics such as entropy, conditional entropy, self-information (also
        known as information content or surprisal), relative entropy (also
        known as KL-divergence or information gain). Joint and marginal
        probability mass functions and contingency tables are also included for
        convenience.

    Copyright (C) Hörmet Yiltiz <hyiltiz@gmail.com>, 2021.

    """
    # TODO generalize for N variables; currently it assumes the X and Y are in
    # the same encoding/dictionary/domain generalize the input arguments s.t.
    # it works for ND vectors, like:
    # compute_mutual_information([X, Y, ...],
    #                            [(range(of X)),(domain of Y),...])

    observed = list(zip(X, Y))
    N = len(observed)

    # NOTE: range() upper range is larger by 1
    grid = product(range(domain[0][0], domain[0][1]+1),
                   range(domain[0][0], domain[0][1]+1))

    # compute the joint frequency table
    observed_counts = [observed.count(coord) for coord in grid]
    cXY = np.array(observed_counts).reshape(domain[0][1], domain[1][1])

    # compute the marginal counts
    [cX, cY] = np.meshgrid(cXY.sum(axis=0), cXY.sum(axis=1))

    # now normalize into probability mass functions
    [pXY, pX, pY] = [cXY/N, cX/N, cY/N]
    pXpY = pX * pY

    # get rid of the zero events (only in the joint is sufficient)
    # 0 * log(0/[possibly zero]) := 0 for information
    # this limit actually comes from the expectation definition of information
    nonzeros = pXY != 0
    nonzeros_X = pX[0, :] != 0
    nonzeros_Y = pY[:, 0] != 0

    # -inf < pmi < min(-log(pX), -log(pY))
    pointwise_mutual_information = np.log2(pXY[nonzeros]/pXpY[nonzeros])

    # the expected value of the pointwise MI
    mutual_XY = sum(pXY[nonzeros] * pointwise_mutual_information)

    # now compute other relevant information metrics
    # the self-information
    self_information_XY = -np.log2(pXY)
    self_information_X = -np.log2(pX[0, :])
    self_information_Y = -np.log2(pY[:, 0])

    # entropy := expected self information
    HX = sum(pX[0, :][nonzeros_X] * self_information_X[nonzeros_X])
    HY = sum(pY[:, 0][nonzeros_Y] * self_information_Y[nonzeros_Y])
    HXY = sum(pXY[nonzeros] * self_information_XY[nonzeros])

    # as a reference and to prevent confusion, we provide most of the relevant
    # values in probability and information theory here.
    # See for more information:
    # https://en.wikipedia.org/wiki/Mutual_information
    # https://en.wikipedia.org/wiki/Entropy_(information_theory)
    # https://en.wikipedia.org/wiki/Information_content for self-information
    out = {'N': N,
           'entropy': {  # expected self-information; non-negative
               'XY': HXY,
               'X': HX,
               'Y': HY,
               'X-given-Y': -sum(pXY[nonzeros] * np.log2(
                   pXY[nonzeros]/pX[nonzeros])),
               'Y-given-X': -sum(pXY[nonzeros] * np.log2(
                   pXY[nonzeros]/pY[nonzeros])),
               'Y-approximates-X': kl(Y, X),
               'X-approximates-Y': kl(X, Y),
               # TODO: better name; check if matrix <> vectors is correct
               'conditional-fallacy-mistook-as-given-B': EI(pXY/pY, pX/pY),
               'conditional-fallacy-mistook-as-given-B': EI(pXY/pX, pY/pX)},
           'mutual-information': {
               'standard': mutual_XY,  # non-negative
               # normalize by entropy (asymmetric), ranges from [0, 1]
               'normalized-by-HX': mutual_XY/HX,
               'normalized-by-HY': mutual_XY/HX,
               # other normalization variants (symmetric)
               'redundancy': mutual_XY/(HX+HY),
               'symmetric': 2*mutual_XY/(HX+HY),
               'generalized-total-correlation': mutual_XY/min(HX, HY),
               'generalized-pearson-correlation': mutual_XY/np.sqrt(HX*HY),
               'information-quality-ratio': mutual_XY/HXY,
               # AMI(X,Y) = [MI(X,Y)-E(MI(X,Y))]/[avg(H(X),H(Y))-E(MI(X,Y))]'
               'adjusted-mutual-information': 'TODO'},
           'self-information': {  # could be inf if p=0
               'X': self_information_X,
               'Y': self_information_Y},
               'XY': self_information_XY,
           'count': {
               'X': cX,
               'Y': cY},
               'XY': cXY,
           'p': {
               'X': pX,
               'Y': pY}
               'XY': pXY,
           }

    # print(f"====\nmutual_XY:{mutual_XY}, "
    #       f"normalied by entropy(X): {mutual_XY/HX}\n, "
    #       f"{out['mutual-information']}")
    # import ipdb; ipdb.set_trace() # BREAKPOIN for debugging

    return (mutual_XY, mutual_XY/HX, out)

def kl_div(p, q):
    "Compute KL divergence, the information gain if p was used as an
    approximation for q (also known as the relative entropy of p w.r.t. q)."

    nonzeros = p != 0
    kl = sum(p[nonzero] * (math.log2(p[nonzero]) - math.log2(q[nonzero])))
    return kl

def EI(w, p):
    "Compute the expected information of p using weights w."
    nonzeros = w != 0
    e_info = sum(w[nonzero] * math.log2(p[nonzero]))
    return e_info

def main():
    # example function call
    domain = [(1, 5), (1, 5)]
    samples = 1000
    X = np.random.choice(range(domain[0][0], domain[0][1]+1),
                         samples, replace=True)
    Y = np.random.choice(range(domain[1][0], domain[1][1]+1),
                         samples, replace=True)
    identical = X
    shuffled = np.random.choice(X, size=len(X), replace=False)

    # test for properties: symmetry, non-negativity
    mi_X_self = compute(X, identical, domain)
    mi_self_X = compute(identical, X, domain)

    mi_X_R = compute(X, shuffled, domain)
    mi_R_X = compute(shuffled, X, domain)

    mi_X_Y = compute(X, Y, domain)
    mi_Y_X = compute(Y, X, domain)

    [mi_X_Y, normalized, info] = compute(X, Y, domain)
    print(mi_X_Y)

if __name__ == '__main__':
    main()
