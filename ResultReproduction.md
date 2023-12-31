# Multi-level data assimilation for simplified ocean models

The results in the manuscript „Multi-level data assimilation for simplified ocean models“ by Beiser, Holm, Lye and Eidsvik are reproducible with this repository. 
However, the results are stochastic and the exact results may differ with statistical variation. 

- Figure 0: tikz 

- Figure 1: tikz

- Figure 2: `scripts/DoubleJetPracticalCost_square.py`

- Figure 3: `notebooks/DoubleJet-Truth4paper.ipynb` (requires running `scripts/DoubleJetTruth.py -L 9`, but results are stochastic)

- Figure 4: `notebooks/DoubleJet-Resoltions4paper.ipynb` (results are seeded)

- Figure 5: tikz + requires running `scripts/DoubleJetVarianceLevel-DA.py` + `scripts/DoubleJetVarianceLevel-DA-PP.py` (with right `source_path` to the output folder of the preceding script and with different functionals in the code for a- and b-plot, see comments in the code) 

- Table 1: `notebooks/DoubleJet-SpeedUp.ipynb` (with the corresponding output folders of practical costs and variances)

> The next figures require running 
> - `scripts/DoubleJetSLDA.py -L 9 -Ne 50`, and 
> - e.g. `scripts/DoubleJetMLDA.py -ls 6 7 8 9 -Nes 275 134 46 13` 
>
> (with right `truth_path` for the output of the truth) 

- Figure 6: `notebooks/DoubleJet-DA-PostProcessing-ERR4paper.ipynb` (with right output folders as input)

- Figure 7: `notebooks/DoubleJet-DA-PostProcessing-ERRloc4paper.ipynb` (with right output folders as input) + requires running `scripts/DoubleJetMLDA.py -ls 6 7 8 9 -Nes 275 134 46 13` (with right `truth_path` for the output of the truth AND `min_localisation_level=1` see comment in the upper part of the script)

- Figure 8: `notebooks/DoubleJet-MLDA-PostProcessing-MLscores4paper.ipynb` (with right output folders as input) 

- Figure 9: `notebooks/DoubleJet-MLRanks-PostProcessing4paper.ipynb` (with right output folders as input)  + requires running `scripts/DoubleJetMLRanks.py`

- Figure 10: `notebooks/DoubleJet-DA-PostProcessing-Drifter4paper.ipynb` (with right output folders as input) 

- Figure 11: `notebooks/DoubleJet-DA-PostProcessing-DrifterERR4paper.ipynb` (with right output folders as input) 

- Figure 12: `notebooks/DoubleJet-DA-PostProcessing-DrifterKDE4paper.ipynb` (with right output folders as input) 

- Table 2: `notebooks/DoubleJet-DA-PostProcessing-DrifterKDE4paper.ipynb` (with right output folders as input, see far down in the notebook) 

A lot of parameters are collected in `utils/DoubleJetParametersReproduction.py`. 