# Multi-level data assimilation for ocean forecasting using the shallow-water equations
This repository contains code, scripts and notebooks related to the work presented in the paper "Multi-level data assimilation for ocean forecasting using the shallow-water equations" by Beiser, Holm, Lye and Eidsvik.
The motivation is to setup multi-level Monte Carlo ensembles of simplified ocean models and evaluate the usability of multi-level data assimilating in the presence of sparse observations of ocean currents.
We then use the posterior multi-level ensemble to forecast the ocean state and drift trajectories.

All code relies heavily on [GPU Ocean](https://github.com/gpuocean/gpuocean) v1.3.0.

## Result Reproduction
Although the exact figures from the paper will be hard to reproduce due to stochastic processes, the figures can be qualitatively reproduced as follows:

- Figure 0: tikz 

- Figure 1: tikz

- Figure 2: `scripts/DoubleJetPracticalCost_square.py`

- Figure 3: `notebooks/DoubleJet-Truth4paper.ipynb` (requires running `scripts/DoubleJetTruth.py -L 9`, but results are stochastic)

- Figure 4: `notebooks/DoubleJet-Resoltions4paper.ipynb` (results are seeded)

- Figure 5: tikz + requires running `scripts/DoubleJetVarianceLevel-DA.py` + `scripts/DoubleJetVarianceLevel-DA-PP.py`
(The first script save results in a newly created putput folder. The second script requries to set `source_path` to the output folder of the first script. Moreover, for different functionals, follow the instructions in the second script to create the a- and b-plot. Also the second script saves results in a newly created output folder.) 

- Table 1: `notebooks/DoubleJet-SpeedUp.ipynb` (requires to set the output folders of the previous cost and variance experiments are manual inputs)

> The next figures require running
> - `scripts/DoubleJetSLDA.py -L 9` 
>
> Then set the output folder of this script as `truth_path` in the following scripts
>
> - `scripts/DoubleJetSLDA.py -L 9 -Ne 50`, and 
> - e.g. `scripts/DoubleJetMLDA.py -ls 6 7 8 9 -Nes 275 134 46 13` 
>
> Again, output folders are created and those folders paths have to be manually set in the following notebooks!

- Figure 6: `notebooks/DoubleJet-DA-PostProcessing-ERR4paper.ipynb` (set folder path as explained above)

- Figure 7: `notebooks/DoubleJet-DA-PostProcessing-ERRloc4paper.ipynb` (set folder path as explained above) + requires running `scripts/DoubleJetMLDA.py -ls 6 7 8 9 -Nes 275 134 46 13` (with right `truth_path` for the output of the truth AND `min_localisation_level=1` see comment in the upper part of the script)

- Figure 8: `notebooks/DoubleJet-MLDA-PostProcessing-MLscores4paper.ipynb` (set folder path as explained above)

- Figure 9: `notebooks/DoubleJet-MLRanks-PostProcessing4paper.ipynb` (set folder path as explained above)  + requires running `scripts/DoubleJetMLRanks.py`

- Figure 10: `notebooks/DoubleJet-DA-PostProcessing-Drifter4paper.ipynb` (set folder path as explained above)

- Figure 11: `notebooks/DoubleJet-DA-PostProcessing-DrifterERR4paper.ipynb` (set folder path as explained above)

- Figure 12: `notebooks/DoubleJet-DA-PostProcessing-DrifterKDE4paper.ipynb` (set folder path as explained above) 

- Table 2: `notebooks/DoubleJet-DA-PostProcessing-DrifterKDE4paper.ipynb` (set folder path as explained above, see far down in the notebook) 

A lot of parameters are collected in `utils/DoubleJetParametersReproduction.py`. 


### Funding
This work is funded by the Norwegian Research Council under grant number 310515 (Havvarsel).
