import pandas as pd
import pickle
import xgboost as xgb
import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt

dfOrigin = pickle.load(open("/dune/data/users/cvilela/MagicRW/mcc11_v4_RHC_RecoLepE_RecoY_200Trees_HackDays/Nominal//Nominal_test.p", "r"))
dfTarget = pickle.load(open("/dune/data/users/cvilela/MagicRW/mcc11_v4_RHC_RecoLepE_RecoY_200Trees_HackDays/ProtonEdepm20pc//ProtonEdepm20pc_test.p", "r"))
bst = xgb.Booster({'nthread':1})
bst.load_model("/dune/data/users/cvilela/MagicRW/mcc11_v4_RHC_RecoLepE_RecoY_200Trees_HackDays/ProtonEdepm20pc/ProtonEdepm20pc.xgb")
                
originWeights = dfOrigin['preweight']
dfOrigin.drop(columns = ['preweight'], inplace = True)

targetWeights = dfTarget['preweight']
dfTarget.drop(columns = ['preweight'], inplace = True)

rw_vars = ["Erec", "Elep_true", "Eproton_dep", "EpiC_dep", "Epi0_dep", "Reco_y"]

dfOrigin.drop(dfOrigin.columns.difference(rw_vars), 1, inplace=True)
dfTarget.drop(dfTarget.columns.difference(rw_vars), 1, inplace=True)

#print dfOrigin
#print dfTarget

dmOrigin = xgb.DMatrix(dfOrigin)
dmTarget = xgb.DMatrix(dfTarget)

originOut = pd.Series(bst.predict(dmOrigin))
targetOut = pd.Series(bst.predict(dmTarget))

originMargins = pd.Series(bst.predict(dmOrigin, output_margin = True))
targetMargins = pd.Series(bst.predict(dmTarget, output_margin = True))

originPredWeights = pd.Series(bst.predict(dmOrigin))
targetPredWeights = pd.Series(bst.predict(dmTarget))

originLabels = pd.Series([1]*len(originMargins))
targetLabels = pd.Series([0]*len(targetMargins))

margins = pd.concat([originMargins, targetMargins], ignore_index = True)
predWeights = pd.concat([originPredWeights, targetPredWeights], ignore_index = True)
labels = pd.concat([originLabels, targetLabels], ignore_index = True)
weights = pd.concat([originWeights, targetWeights], ignore_index = True)

bins = 50

dfEverything = pd.DataFrame()
dfEverything['margin'] = margins
dfEverything['weight'] = weights
dfEverything['label'] = (1-labels)
dfEverything['prob'] = predWeights


(valsPos, binslist , patch) = plt.hist(dfEverything['prob'], weights = dfEverything['label']*dfEverything['weight'], bins = bins, range = (0, 1.), histtype = 'step')
plt.hist(dfEverything['prob'], weights = (1-dfEverything['label'])*dfEverything['weight'], bins = bins, range = (0, 1.), histtype = 'step')
(valsAll, binslist , patch) = plt.hist(dfEverything['prob'], weights = dfEverything['weight'], bins = bins, range = (0., 1.), histtype = 'step')

plt.show()

valsReliabilityRaw = valsPos/valsAll

plt.plot([0,1], [0,1], color = 'k', ls = 'dashed')
plt.plot(binslist[:-1],valsReliabilityRaw, label = 'Raw')


#plt.hist(originMargins, bins = 100, range = (-5, 5) , histtype = 'step')
#plt.hist(targetMargins, bins = 100, range = (-5, 5) , histtype = 'step')

plt.show()

plt.hist(dfEverything['prob'])

plt.show()

#exit()


def platt(x, margins, labels, weights) :
    A = x[0]
    B = x[1]

    p = 1/(1 + np.exp( A * margins + B) )
    
#    print p
#    print weights
#    print labels

    summand = labels*weights*np.log(p) + (1-labels)*weights*np.log(1-p)

#    print summand

    ret = -1*np.sum(summand)
#    print A,B,ret                                                                                                                                                                                                 
    return ret

x0 = np.array([-1., 0.])
res = minimize(platt, args = (margins, (1-labels), weights), x0 = x0)

print res.x

raw_input()
