from MagicRWSample import Sample

import os

import numpy as np

import pandas as pd
import pickle

import xgboost as xgb

class SampleXGB(Sample) :
    
    def __init__(self, name, outFilePath, inFilePath, trainFrac = 0.0, numTrees = 20):
        super(SampleXGB, self).__init__(name = name, outFilePath = outFilePath, inFilePath = inFilePath, trainFrac = trainFrac, numTrees = numTrees)
        
    def trainBDT(self, targetSample) :

        originDF, originDFtest = self.getDataFrames()
        targetDF, targetDFtest = targetSample.getDataFrames()

        if self.observables.keys() != targetSample.observables.keys() :
            print 'Error observables for target and origin data sets do not match. Exiting...'
            print 'Origin:', self.observables.keys()
            print 'Target:', targetSample.observables.keys()
            exit(-1)

        originPreWeights = originDF["preweight"]
        targetPreWeights = targetDF["preweight"]

        originPreWeightsTest = originDFtest["preweight"]
        targetPreWeightsTest = targetDFtest["preweight"]
        
        originDF = originDF[self.observables.keys()]
        targetDF = targetDF[targetSample.observables.keys()]

        labelOrigin = [1]*len(originDF)
        labelTarget = [0]*len(targetDF)

        originDFtest = originDFtest[self.observables.keys()]
        targetDFtest = targetDFtest[targetSample.observables.keys()]

        labelOrigintest = [1]*len(originDFtest)
        labelTargettest = [0]*len(targetDFtest)
        
        allDF = pd.concat([originDF, targetDF])
        label = labelOrigin + labelTarget
        weight = pd.concat([originPreWeights,targetPreWeights])

        allDFtest = pd.concat([originDFtest, targetDFtest])
        labeltest = labelOrigintest + labelTargettest
        weighttest = pd.concat([originPreWeightsTest, targetPreWeightsTest])

        data = xgb.DMatrix(data = allDF, label = label, weight = weight)
        datatest = xgb.DMatrix(data = allDFtest, label = labeltest, weight = weighttest)

        params = {}
        params['nthread'] = 1
        params["tree_method"] = "auto" 
        params["objective"] = 'reg:logistic'
        params["eta"] = 0.3
        params["max_depth"] = 3
        params["min_child_weight"] = 100
        params["subsample"] = 0.5
        params["lambda"] = 1
        params["alpha"] = 0
        params["feature_selector"] = "greedy"

        eval_result = {}
        evals = [(data, "train"), (datatest, "test")]

        bst = xgb.train(params = params, dtrain = data,  num_boost_round = self.numTrees, evals = evals, evals_result = eval_result , verbose_eval = 10)

        for f in [ self.xgbPath()+'.raw.txt', self.xgbPath()+'.eval.p', self.xgbPath() ] :
            try:
                os.remove(f)
            except OSError:
                pass

        # dump model
        bst.dump_model(self.xgbPath()+'.raw.txt')
        with open(self.xgbPath()+'.eval.p', "w") as f :
            pickle.dump(eval_result, f)
        
        bst.save_model(self.xgbPath())
        print "Saved XGBoost model"
        
    def trainTrueKinBDT(self) :

        originDF, originDFtest = self.getDataFrames()
        
        originPreWeights = originDF["preweight"]
        originTestPreWeights = originDFtest["preweight"]

        bst = xgb.Booster({'nthread' : 1})
        bst.load_model(self.xgbPath())
            
        data = xgb.DMatrix(originDF[self.observables.keys()], weight=originPreWeights)
        datatest = xgb.DMatrix(originDFtest[self.observables.keys()], weight=originTestPreWeights)

# FHC        
#        plattA = -1.01280336
#        plattB = 0.01425506
# RHC        
        plattA = -1.06557091
        plattB = -0.01089998

        weights_to_predict = bst.predict(data, output_margin = True)
        weights_to_predict = np.exp(plattA*weights_to_predict + plattB)

        weights_to_predictTest = bst.predict(datatest, output_margin = True)
        weights_to_predictTest = np.exp(plattA*weights_to_predictTest+plattB)


        dataKin = xgb.DMatrix(data = self.trueKinDF(originDF), label = weights_to_predict, weight=originPreWeights)
        dataKinTest = xgb.DMatrix(data = self.trueKinDF(originDFtest), label = weights_to_predictTest, weight=originTestPreWeights)

        params = {}
        params['nthread'] = 1
        params["tree_method"] = "auto" 
        params["objective"] = 'reg:linear'

        eval_result = {}
        evals = [(dataKin, "train"), (dataKinTest, "test")]

        bstKin = xgb.train(params = params, dtrain = dataKin,  num_boost_round=20, evals = evals, evals_result = eval_result)
        
        for f in [ self.trueKinBDTPath()+'.raw.txt', self.trueKinBDTPath(), self.trueKinBDTPath()+'.eval.p' ] :
            try:
                os.remove(f)
            except OSError:
                pass

        # dump model
        bstKin.dump_model(self.trueKinBDTPath()+'.raw.txt')

        bstKin.save_model(self.trueKinBDTPath())

        pickle.dump(eval_result, open(self.trueKinBDTPath()+'.eval.p', 'w'))
        print "Saved XGBoost model"
