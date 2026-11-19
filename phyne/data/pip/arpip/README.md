This directory contains data samples from the [ARPIP paper](https://academic.oup.com/sysbio/article/72/2/307/6648472).
The data is used to check the correctness of the ARPIP reimplementation within `phyne`.

## Larger simulated dataset

Files starting with `sim-0` refer to simulated sequences using the PIP model with mu = 0.1:
- sim-0_msa_new.fasta is the input msa;
- sim-0_new.newick as the input tree (changed node ids from "x" -> "i_x");
- sim-0_inferred.fasta is the result for comparison (concatenated alignment from input and ARPIP reconstruction).

*Note*: this dataset has no insertions that are more likely to be higher than the MRCA. Such cases are covered by unit tests.

## Small example

The `tree.nwk` and `msa.fasta` files are a small example from the [ARPIP repository](https://github.com/acg-team/bpp-arpip/tree/master/test/input), used to check exact numerical values or the PIP likelihood.