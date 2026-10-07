This directory contains a Multiple-Ancestral-Sequence-Alignment simulated with Indelible. 
It is used for checking the correctness of the Dynamic Programming (DP) algorithm that re-estimates 
the ancestral sequences of two adjacent nodes in a phylogenetic tree under the TKF92 indel model.
The DP is checked against a brute-force calculation of the optimal ancestral sequences. Precomputed 
results of the brute-force calculation are provided, but can be recomputed by passing the compile feature
'recompute-brute-force-asr' (and optionally 'multithread-brute-force-asr').
Note: for this particular alignment the re-estimated sequences alter the observed fragment boundaries in the alignment. 
