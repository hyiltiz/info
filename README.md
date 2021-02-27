# Quick start
Download the module `info.py` into your project directory, and do:
```
from info import compute as compute_mutual_information

X = [1, 2, 3, 4, 5]
Y = [4, 1, 3, 4, 2]
X_range = (1, 5)
Y_range = (1, 5)  # 5 did not occur in the sample Y yet

[mi_XY, mi_XY_normalized, everything] = compute_mutual_information(X, Y, [X_range, Y_range])

# `everything` is a nested dictionary that contains other normalized
# variants of mutual information, as well as some commonly used information
# theoretic metrics such as entropy, conditional entropy, self-information
# (also known as information content or surprisal). Joint and marginal
# probability mass functions and contingency tables are also included for
# convenience.

everything = {
 'N': 1000,
 'entropy': {'X':         2.3168095633619847,
             'Y':         2.3199539124456714,
             'XY':        4.625700518983682,
             'X-given-Y': 2.3088909556216985,
             'Y-given-X': 2.305746606538012},
 'mutual-information': {
             'standard':                        0.01106295682397285,
              # normalize by entropy (asymmetric), ranges from [0, 1]
             'normalized-by-HX':                0.0047750825095521,
             'normalized-by-HY':                0.0047750825095521,
              # other normalization variants (symmetric)
             'symmetric':                       0.004771844361565519,
             'redundancy':                      0.0023859221807827594,
             'generalized-pearson-correlation': 0.004771845458771204,
             'generalized-total-correlation':   0.0047750825095521,
             'information-quality-ratio':       0.0023916284200784154,
             'adjusted-mutual-information':     'TODO'},
 'self-information': {'X': array([2.35845397, 2.41119543, 2.23786383, 2.48196851, 2.14560532]),
                      'Y': array([2.29335894, 2.42662547, 2.32192809, 2.20423305, 2.37332725]),
                      'XY': array([[4.83650127, 4.50635267, 4.57346686, 4.68038207, 4.50635267],
                                   [4.53951953, 5.01158797, 4.83650127, 4.87832144, 4.53951953],
                                   [4.57346686, 4.96578428, 4.60823228, 5.15842936, 4.13289427],
                                   [4.64385619, 4.44222233, 4.21089678, 4.79585928, 4.60823228],
                                   [4.83650127, 4.83650127, 4.64385619, 4.57346686, 4.60823228]])}
 'count': {'X': array([[195, 188, 212, 179, 226],
                       [195, 188, 212, 179, 226],
                       [195, 188, 212, 179, 226],
                       [195, 188, 212, 179, 226],
                       [195, 188, 212, 179, 226]]),
           'Y': array([[204, 204, 204, 204, 204],
                       [186, 186, 186, 186, 186],
                       [200, 200, 200, 200, 200],
                       [217, 217, 217, 217, 217],
                       [193, 193, 193, 193, 193]])},
           'XY': array([[35, 44, 42, 39, 44],
                        [43, 31, 35, 34, 43],
                        [42, 32, 41, 28, 57],
                        [40, 46, 54, 36, 41],
                        [35, 35, 40, 42, 41]]),
 'p': {'X': array([[0.195, 0.188, 0.212, 0.179, 0.226],
                   [0.195, 0.188, 0.212, 0.179, 0.226],
                   [0.195, 0.188, 0.212, 0.179, 0.226],
                   [0.195, 0.188, 0.212, 0.179, 0.226],
                   [0.195, 0.188, 0.212, 0.179, 0.226]]),
       'Y': array([[0.204, 0.204, 0.204, 0.204, 0.204],
                   [0.186, 0.186, 0.186, 0.186, 0.186],
                   [0.2  , 0.2  , 0.2  , 0.2  , 0.2  ],
                   [0.217, 0.217, 0.217, 0.217, 0.217],
                   [0.193, 0.193, 0.193, 0.193, 0.193]])}
       'XY': array([[0.035, 0.044, 0.042, 0.039, 0.044],
                    [0.043, 0.031, 0.035, 0.034, 0.043],
                    [0.042, 0.032, 0.041, 0.028, 0.057],
                    [0.04 , 0.046, 0.054, 0.036, 0.041],
                    [0.035, 0.035, 0.04 , 0.042, 0.041]]),
}
```
