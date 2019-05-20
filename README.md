# MagicRW
DUNE-PRISM mock data reweighting tools

## runMagicRW.py
Main script that sets everything going. By default it will:
 - Read in ROOT files and write variables of interest to pickle.
 - Train the BDT to classify Nominal vs variation (Nominal is always the first sample in the lists) in reconstructed variables.
 - Produce histograms of weights in pairs of true variables (deprecated).
 - Write the histograms in ROOT format (deprecated).
 - Train a BDT to learn the predictions of the initial BDT, but in true kinematic variables.
 - Make plots at several of the stages above.
 
### DunePRISMSamples.py
 - Samples defined here. This includes calculating variables (and mock data variations of the variables), selection, defining space of variables to reweight in and useful plotting stuff.

### MagicRWSample.py
 - Where most functionality is implemented.
 
### MagicRWSampleXGB.py
 - Inherits functinality from MagicRWSample.py, but uses XGBoost BDT framework.
 
## Random utilities:
### testTreelite.py
 - Uses Treelite to export the BDT as C code -- this is used for mock data implementation in CAFAna.

### plattScaling.py
 - Calculates Platt scaling parameters to calibrate the BDT output -- not always required.
 
### plotTrain.py
 - Plots training curves
 
### makeFriendTrees.py
 - Produce ROOT friend trees with weights from a BDT model. Probably needs updating.
