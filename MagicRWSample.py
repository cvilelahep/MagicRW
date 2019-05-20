from random import random
from abc import ABCMeta, abstractmethod, abstractproperty
import pandas as pd
from ROOT import TFile, TChain, TH2F, TTree
import cPickle as pickle
import os
from hep_ml.reweight import GBReweighter
from corner import corner
import matplotlib.pyplot as plt
import numpy as np
from root_numpy import array2root
from array import array
import xgboost as xgb

overallWeight = 1.0 # HARDCODED! BAD! FIX!!

# Is this the most appropriate place for this? Maybe not...
# GENIE MODE ENUM, from:
# https://github.com/GENIE-MC/Generator/blob/R-2_12_10/src/Interaction/ScatteringType.h
# retrieved Jan 17

#  kScNull = 0,
#  kScQuasiElastic,
#  kScSingleKaon,
#  kScDeepInelastic,
#  kScResonant,
#  kScCoherent,
#  kScDiffractive,
#  kScNuElectronElastic,
#  kScInverseMuDecay,
#  kScAMNuGamma,
#  kScMEC,
#  kScCoherentElas,
#  kScInverseBetaDecay,
#  kScGlashowResonance,
#  kScIMDAnnihilation

GenieCodeDict = { 
    0 : "Unknown",
    1 : "QE",
    2 : "SingleKaon",
    3 : "DIS",
    4 : "RES",
    5 : "COH",
    6 : "Diffractive",
    7 : "NuElectronElastic",
    8 : "InverseMuonDecay",
    9 : "AMNuGamma",
    10 : "MEC",
    11 : "COHElastic",
    12 : "IBD",
    13 : "GlashowRES",
    14 : "IMDAnnihilation"}

SimbToGenieDict = {
    -1 : 0,   # Unknown
     0 : 1,   # QE
     1 : 4,   # RES
     2 : 3,   # DIS
     3 : 5,   # COH
     4 : 11,  # COHElastic
     5 : 7,   # Nu+e
     6 : 14,  # IMDAnnihilation
     7 : 12,  # IBD
     8 : 13,  # GlashowRES
     9 : 9,   # AMNuGamma
     10 : 10, # MEC
     11 : 6,  # Diffractive
     12 : 0,  # "kEM"
     13 : 0}  # "kWeakMix"

class Sample(object) :
    
    __metaclass__ = ABCMeta

    def __init__(self, name, outFilePath, inFilePath, trainFrac = 0.0, numTrees = 20):
        self.name = name
        self.outFilePath = outFilePath
        self.inFilePath = inFilePath
        self.trainFrac = trainFrac
        self.numTrees = numTrees
        super(Sample, self).__init__()
        
    @abstractmethod
    def selection(self, event) :
        pass
        
    @abstractmethod
    def variables(self, event) :
        pass

    @abstractproperty
    def observables() :
        pass

    @abstractproperty
    def trueVarPairs() :
        pass
    
    def dataframe(self) :
        EDeps = TChain("caf")
        EDeps.Add(self.inFilePath)

        df = pd.DataFrame( [ self.variables(event)
                             for event in EDeps
                             if self.selection(event) ])
        del EDeps
        
        if (self.trainFrac == 0.0) :
            return df.iloc[:1], df 
        elif (self.trainFrac == 1.0) :
            return df, df.iloc[:1]
        else :
            splitIndex = int( len(df) * self.trainFrac)
            print splitIndex, len(df), self.trainFrac
            #      Training set             Testing set
            return df.iloc[:splitIndex], df.iloc[splitIndex:]

    def baseDir(self) :
        return self.outFilePath+"/"+self.name+"/"

    def plotsDir(self) :
        return self.outFilePath+"/"+self.name+"/Plots/"

    def testDFpath(self) :
        return self.baseDir()+self.name+"_test.p"
    
    def trainDFpath(self) :
        return self.baseDir()+self.name+"_train.p"

    def gbrwPath(self) :
        return self.baseDir()+self.name+"_gbrw.p"

    def xgbPath(self) :
        return self.baseDir()+self.name+".xgb"

    def binnedWeightsPath(self) :
        return self.baseDir()+self.name+"_binnedWeights.p"

    def binnedWeightsPathROOT(self) :
        return self.baseDir()+self.name+"_binnedWeights.root"

    def trueKinBDTPath(self) :
        return self.baseDir()+self.name+"_trueKinBDT.xgb"
        
    def pickleData(self) :
        trainSet, testSet = self.dataframe()

        if not os.path.isdir(self.baseDir()) :
            os.makedirs(self.baseDir())
        
        with open(self.trainDFpath(), "wb")  as f :
            pickle.dump(trainSet, f, protocol = pickle.HIGHEST_PROTOCOL)
        
        with open(self.testDFpath(), "wb")  as f :
            pickle.dump(testSet, f, protocol = pickle.HIGHEST_PROTOCOL)

        return trainSet, testSet

    def getDataFrames(self, train = True, test = True) :

        try :
            if train :
                fTrain = open(self.trainDFpath(), 'r')
                trainDF = pickle.load(fTrain)
            if test :
                fTest = open(self.testDFpath(), 'r')
                testDF = pickle.load(fTest)
        except IOError as e :
            print 'MagicRWsample getTrainDF I/O error({0}): {1}'.format(e.errno, e.strerror), 'Attempting to produce dataframes...'
            trainDF, testDF = self.pickleData()
            print 'New dataframes produced'
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise
        finally :
            if train and test :
                return trainDF, testDF
            elif train :
                return trainDF
            elif test :
                return testDF

    def getTrainDataFrame(self) :
        return self.getDataFrames(train = True, test = False)

    def getTestDataFrame(self) :
        return self.getDataFrames(train = False, test = True)
        
    def trainBDT(self, targetSample) :

        originDF = self.getTrainDataFrame()
        targetDF = targetSample.getTrainDataFrame()

        if self.observables.keys() != targetSample.observables.keys() :
            print 'Error observables for target and origin data sets do not match. Exiting...'
            print 'Origin:', self.observables.keys()
            print 'Target:', targetSample.observables.keys()
            exit(-1)
        
        originPreWeights = originDF["preweight"]
        targetPreWeights = targetDF["preweight"]

        originDF = originDF[self.observables.keys()]
        targetDF = targetDF[targetSample.observables.keys()]

        reweighter = GBReweighter(n_estimators=200,
                                  learning_rate=.1,
                                  max_depth=3,
                                  min_samples_leaf=1000,
                                  loss_regularization=1.0)

        reweighter.fit(original = originDF, target = targetDF, original_weight = originPreWeights, target_weight = targetPreWeights)

        with open(self.gbrwPath(), "wb") as f :
            pickle.dump(reweighter, f)

    def plotDiagnostics(self, targetSample, weightScheme = "gbrw", useROOThistos = False) :
        
        if not os.path.isdir(self.plotsDir()) :
            os.makedirs(self.plotsDir())

        targetDFtrain, targetDFtest = targetSample.getDataFrames()
        originDFtrain, originDFtest = self.getDataFrames()
        
        print self.observables.keys()
        print targetDFtest

        targetDFtestObs = targetDFtest[self.observables.keys()]
        originDFtestObs = originDFtest[self.observables.keys()]

        if weightScheme == "gbrw" :
            with open(self.gbrwPath(), 'r') as f :
                reweighter = pickle.load(f)
                weightsTest  = reweighter.predict_weights(originDFtestObs)

        elif weightScheme == "xgbTrueKin" :
            bst = xgb.Booster({'nthread': 1})
            bst.load_model(self.trueKinBDTPath())
            
            data = xgb.DMatrix(self.trueKinDF(originDFtest))

            weightsTest = bst.predict(data)

        elif weightScheme == "xgb" :
            bst = xgb.Booster({'nthread': 1})
            bst.load_model(self.xgbPath())
            
            data = xgb.DMatrix(originDFtestObs)

            weightsTest = bst.predict(data)
            weightsTest = 1./weightsTest - 1

        elif not useROOThistos :
            originDFtestVarPairs = originDFtest[self.trueVarPairs[weightScheme]["vars"] + ["GENIEIntMode"]]
            weightsTest = self.predictBinnedWeights(originDFtestVarPairs,  weightScheme)
        elif useROOThistos :
            originDFtestVarPairs = originDFtest[self.trueVarPairs[weightScheme]["vars"] + ["GENIEIntMode"]]
            weightsTest = self.predictBinnedWeightsROOT(originDFtestVarPairs,  weightScheme)
            
        if useROOThistos and weightScheme != "gbrw" :
            weightScheme += "_ROOT"
            

        figTarget = self.getCornerPlot(dataFrame = targetDFtestObs, color = 'r', label = 'Nominal' )
        figTarget.savefig(self.plotsDir()+"corner_"+self.name+"_targetOnly.png", transparent = True, figsize = {50, 50}, dpi = 240)

        figOrigin = self.getCornerPlot(dataFrame = originDFtestObs, color = 'b', label = self.name, fig = figTarget )
        figTarget.savefig(self.plotsDir()+"corner_"+self.name+"_target_originNotRW.png", transparent = True, figsize = {50, 50}, dpi = 240)

        figOriginRW = self.getCornerPlot(dataFrame = originDFtestObs, color = 'g', label = self.name , weights = weightsTest, fig = figTarget)
        figTarget.savefig(self.plotsDir()+"corner_"+self.name+"_"+weightScheme+".png", transparent = True, figsize = {50, 50}, dpi = 240)

        figErec = plt.figure()

        self.getErecResponse(dataFrame = targetDFtest, color = 'r', label = 'Nominal' )
        figErec.savefig(self.plotsDir()+"Erec_"+self.name+"_targetOnly.png", transparent = True, figsize = {5, 5}, dpi = 240)
        
        self.getErecResponse(dataFrame = originDFtest, color = 'b', label = self.name, fig = figErec )
        figErec.savefig(self.plotsDir()+"Erec_"+self.name+"_target_originNotRW.png", transparent = True, figsize = {5, 5}, dpi = 240)

        self.getErecResponse(dataFrame = originDFtest, color = 'g', label = self.name, weights = weightsTest, fig = figErec )
        figErec.savefig(self.plotsDir()+"Erec_"+self.name+"_"+weightScheme+".png", transparent = True, figsize = {5, 5}, dpi = 240)

        
            
    def getCornerPlot(self, dataFrame, color, label, weights = None, fig = None) :

        labels    = [ observable["label"]    for key, observable in self.observables.items() ]
        ranges    = [ observable["range"]    for key, observable in self.observables.items() ]
        logScales = [ observable["logScale"] for key, observable in self.observables.items() ]

        histArgs1d = {"normed" : False , "linewidth" : 1.5}
        contourArgs = { "linewidths" : 1.5}

        bins = 24

        fig =  corner(dataFrame, labels = labels, plot_contours = True, plot_datapoints = False, plot_density = False, color = color, bins = bins, max_n_ticks = 6, weights = weights, range = ranges, no_fill_contours = True, fig = fig, hist_kwargs = dict(histArgs1d.items() + {"color" : color}.items()+{"label" : label}.items()), contour_kwargs = dict(contourArgs.items() + {"color" : color}.items()), smooth = True, smooth1d = None, label = label )

        fig.axes[len(labels)**2-1].legend()

        for i in range(0, len(logScales)) :
            if logScales[i] :
                fig.axes[i*len(logScales)+i].set_yscale("log")
                fig.axes[i*len(logScales)+i].set_ylim(ymin = 1)
                fig.axes[i*len(logScales)+i].yaxis.set_label_position("right")

        return fig

    def getErecResponse(self, dataFrame, color, label, weights = None, fig = None )  :
        return plt.hist( ( dataFrame["Erec"] - dataFrame["Etrue"] ) / dataFrame["Etrue"], label = label, color = color, weights = weights, bins = 100, range = [-0.3, 0.2] , histtype = 'step')
#        return plt.hist( ( dataFrame["Erec"] - dataFrame["Etrue"] ) / dataFrame["Etrue"], label = label, color = color, fig = fig, weights = weights, bins = 100, range = [-0.3, 0.2] )
    
    def makeBinnedWeights(self) :
        if not os.path.isdir(self.plotsDir()) :
            os.makedirs(self.plotsDir())
            
        originDFtrain, originDFtest = self.getDataFrames()

        originDF = pd.concat([originDFtrain, originDFtest])

        with open(self.gbrwPath(), 'r') as f :
            reweighter = pickle.load(f) # Wrap this?

        weights = reweighter.predict_weights(originDF[self.observables.keys()])

        originDF = pd.DataFrame.join(originDF, pd.DataFrame([{"weights" : weight} for weight in weights ]))

        binnedWeights = {}
        
        for schemeName, schemeVars in self.trueVarPairs.iteritems() :
            binnedWeights[schemeName] = {}
            for iMode, modeName in GenieCodeDict.iteritems() :
                thisModeDF = originDF[originDF['GENIEIntMode'] == iMode]
                h1, xedges, yedges = np.histogram2d(x = thisModeDF[schemeVars['vars'][0]], y = thisModeDF[schemeVars['vars'][1]], bins = schemeVars['bins'], range = schemeVars['range'])
                h2, xedges, yedges = np.histogram2d(x = thisModeDF[schemeVars['vars'][0]], y = thisModeDF[schemeVars['vars'][1]], bins = schemeVars['bins'], range = schemeVars['range'], weights = thisModeDF['weights'].apply(lambda x: x*overallWeight))

                h = h2/h1
                binnedWeights[schemeName][iMode] = {"histogram" : h, "xedges" : xedges, "yedges" : yedges }

        with open(self.binnedWeightsPath(), "wb") as f :
            pickle.dump(binnedWeights, f)

    def makeROOTBinnedWeights(self) :
        if not os.path.isfile(self.binnedWeightsPath()) :
            print "makeROOTBinnedWeights WARNING: binned weights do not exist. Calling makeBinnedWeights ..."
            self.makeBinnedWeights()
        with open(self.binnedWeightsPath(), "r") as f :
            binnedWeights = pickle.load(f)
        
        rootHistos = []
        
        for schemeName, binnedWeightScheme in binnedWeights.iteritems() :
            for modeNum, binnedWeightSchemeMode in binnedWeightScheme.iteritems() :

                rootHistos.append(TH2F(schemeName+"_"+str(modeNum), schemeName+" "+GenieCodeDict[modeNum], self.trueVarPairs[schemeName]["bins"], self.trueVarPairs[schemeName]["range"][0][0], self.trueVarPairs[schemeName]["range"][0][1], self.trueVarPairs[schemeName]["bins"], self.trueVarPairs[schemeName]["range"][1][0], self.trueVarPairs[schemeName]["range"][1][1]))
                for x in range(1, self.trueVarPairs[schemeName]["bins"]+1) :
                    for y in range(1, self.trueVarPairs[schemeName]["bins"]+1) :
                        if np.isnan(binnedWeightSchemeMode["histogram"][x-1][y-1]) :
                            rootHistos[-1].SetBinContent(x,y,1.)
                        else :
                            rootHistos[-1].SetBinContent(x,y,binnedWeightSchemeMode["histogram"][x-1][y-1])
        fOut = TFile(self.binnedWeightsPathROOT(), "RECREATE")
        for histo in rootHistos :
            histo.Write()
        fOut.Close()
            

    def trainTrueKinBDT(self) :

        originDF = self.getTrainDataFrame()
        
        originPreWeights = originDF["preweight"]

        with open(self.gbrwPath(), 'r') as f :
            reweighter = pickle.load(f) # Wrap this?

        weights_to_predict = reweighter.predict_weights(originDF[self.observables.keys()])

        data = xgb.DMatrix(data = self.trueKinDF(originDF), label = weights_to_predict, weight = originPreWeights)

        params = {}
        params['nthread'] = 1
        params["tree_method"] = "auto" 
        params["objective"] = 'reg:linear'

        bst = xgb.train(params = params, dtrain = data,  num_boost_round=1000)
        
        for f in [ self.trueKinBDTPath()+'.raw.txt', self.trueKinBDTPath() ] :
            try:
                os.remove(f)
            except OSError:
                pass

        # dump model
        bst.dump_model(self.trueKinBDTPath()+'.raw.txt')

        bst.save_model(self.trueKinBDTPath())
        print "Saved XGBoost model"

    def predictBinnedWeights(self, df, varPair) :
        with open(self.binnedWeightsPath(), "r") as f :
            binnedWeights = pickle.load(f)
            
        weights = []
        for index, row in df.iterrows() :

            if row["GENIEIntMode"] in binnedWeights[varPair].keys() :

                xedges = binnedWeights[varPair][row["GENIEIntMode"]]["xedges"]
                yedges = binnedWeights[varPair][row["GENIEIntMode"]]["yedges"]

                xBin = np.digitize(row[self.trueVarPairs[varPair]["vars"][0]], xedges)
                yBin = np.digitize(row[self.trueVarPairs[varPair]["vars"][1]], yedges)

                weights += [ binnedWeights[varPair][row["GENIEIntMode"]]["histogram"][xBin-1][yBin-1] if xBin < len(xedges) and yBin < len(yedges) else 1. ]
            else :
                weights += [ 1.0 ]

        return weights

    def predictBinnedWeightsROOT(self, df, varPair) :
        
        fIn = TFile(self.binnedWeightsPathROOT())
        rootHistos = {}
        for modeNum in GenieCodeDict :
            rootHistos[modeNum] = fIn.Get(varPair+"_"+str(modeNum))
            
        weights = []
        for index, row in df.iterrows() :
            if (row[ self.trueVarPairs[varPair]["vars"][0] ] > self.trueVarPairs[varPair]["range"][0][0] and 
                row[ self.trueVarPairs[varPair]["vars"][0] ] < self.trueVarPairs[varPair]["range"][0][1] and
                row[ self.trueVarPairs[varPair]["vars"][1] ] > self.trueVarPairs[varPair]["range"][1][0] and 
                row[ self.trueVarPairs[varPair]["vars"][1] ] < self.trueVarPairs[varPair]["range"][1][1] and 
                row["GENIEIntMode"] in rootHistos.keys() ) : 
                weights += [ rootHistos[row["GENIEIntMode"]].Interpolate(row[ self.trueVarPairs[varPair]["vars"][0] ], row[ self.trueVarPairs[varPair]["vars"][1] ] ) ]
            else :
                weights += [ 1.0 ]

        fIn.Close()
        return weights

    @staticmethod
    def produceFriendTrees(filePath, nuModeSample, antinuModeSample, isNominal = False, isFD = False) :
        
        with open(nuModeSample.binnedWeightsPath(), "r") as f :
            print nuModeSample.binnedWeightsPath()
            binnedWeightsNu = pickle.load(f)
        with open(antinuModeSample.binnedWeightsPath(), "r") as f :
            binnedWeightsAntiNu = pickle.load(f)

        binnedWeightsNuROOT = TFile(nuModeSample.binnedWeightsPathROOT())
        binnedWeightsAntiNuROOT = TFile(antinuModeSample.binnedWeightsPathROOT())
        
        EDeps = TChain("caf")
        EDeps.Add(filePath)

#        df = pd.DataFrame()
        df = []
        eventCounter = 0

        # Create ROOT file to store the weights
        outFname = os.path.splitext(filePath)[0]+'_'+nuModeSample.name+'_weights.root'
        outFile = TFile(outFname, "RECREATE")

        # Declare TTree to store weights
        outTree = TTree('rw_'+nuModeSample.name, "Fake data weights")
        treeVars = {}
        for schemeName, schemeVars in nuModeSample.trueVarPairs.iteritems() :
            treeVars[schemeName] = array( 'f', [ 0. ] )
            treeVars[schemeName+"_ROOT"] = array( 'f', [ 0. ] )

            outTree.Branch(schemeName, treeVars[schemeName], schemeName+"/F")
            outTree.Branch(schemeName+"_ROOT", treeVars[schemeName+"_ROOT"], schemeName+"_ROOT/F")
        treeVars["Erec"] = array( 'f', [ 0. ] )
        treeVars["IsSelected"] = array( 'i', [0] )
        outTree.Branch("Erec", treeVars["Erec"], "Erec/F")
        outTree.Branch("IsSelected", treeVars["IsSelected"], "IsSelected/I")

        for event in EDeps :
            
            thisEvent = nuModeSample.variables(event)

#            weights = {}
            
            binnedWeights = None
            binnedWeightsROOT = None

            thisSample = nuModeSample
#            if nuModeSample.selection(event) :
            if event.isCC and (np.sign(event.LepPDG) > 0) :
#                print "Got nu event"
                # It's a nu event!
                if not isNominal :
                    binnedWeights = binnedWeightsNu
                    binnedWeightsROOT = binnedWeightsNuROOT
                thisSample = nuModeSample
#            elif antinuModeSample.selection(event) :
            elif event.isCC and (np.sign(event.LepPDG) < 0) :
#                print "Got antinu event"
                # It's an antinu event:
                if not isNominal :
                    binnedWeights = binnedWeightsAntiNu
                    binnedWeightsROOT = binnedWeightsAntiNuROOT
                thisSample = antinuModeSample
            else : 
                pass

            eventDF = thisSample.variables(event)

            mode = event.mode
            if isFD :
                mode = SimbToGenieDict[mode]


            for schemeName, schemeVars in nuModeSample.trueVarPairs.iteritems() :
                if binnedWeights != None and mode in binnedWeights[schemeName].keys() :
                    xedges = binnedWeights[schemeName][mode]["xedges"]
                    yedges = binnedWeights[schemeName][mode]["yedges"]
                    
                    xBin = np.digitize(eventDF[thisSample.trueVarPairs[schemeName]["vars"][0]], xedges)
                    yBin = np.digitize(eventDF[thisSample.trueVarPairs[schemeName]["vars"][1]], yedges)
                    
#                    weights[schemeName] = [binnedWeights[schemeName][mode]["histogram"][xBin-1][yBin-1] if xBin < len(xedges) and yBin < len(yedges) else 1.]
                    treeVars[schemeName][0] = binnedWeights[schemeName][mode]["histogram"][xBin-1][yBin-1] if xBin < len(xedges) and yBin < len(yedges) else 1.
                else :
#                    weights[schemeName] = [1.]
                    treeVars[schemeName][0] = 1.
                # ROOT histogram weights
                histName = schemeName+"_"+str(mode)
                if (binnedWeightsROOT != None and binnedWeightsROOT.GetListOfKeys().Contains(histName) and 
                    eventDF[ thisSample.trueVarPairs[schemeName]["vars"][0] ] > thisSample.trueVarPairs[schemeName]["range"][0][0] and 
                    eventDF[ thisSample.trueVarPairs[schemeName]["vars"][0] ] < thisSample.trueVarPairs[schemeName]["range"][0][1] and
                    eventDF[ thisSample.trueVarPairs[schemeName]["vars"][1] ] > thisSample.trueVarPairs[schemeName]["range"][1][0] and 
                    eventDF[ thisSample.trueVarPairs[schemeName]["vars"][1] ] < thisSample.trueVarPairs[schemeName]["range"][1][1] ) :
#                    weights[schemeName+"_ROOT"] = [ binnedWeightsROOT.Get(histName).Interpolate(eventDF[thisSample.trueVarPairs[schemeName]["vars"][0]], eventDF[thisSample.trueVarPairs[schemeName]["vars"][1]]) ]
                    treeVars[schemeName+"_ROOT"][0] =  binnedWeightsROOT.Get(histName).Interpolate(eventDF[thisSample.trueVarPairs[schemeName]["vars"][0]], eventDF[thisSample.trueVarPairs[schemeName]["vars"][1]]) 
                else : 
#                    weights[schemeName+"_ROOT"] = [1.]
                    treeVars[schemeName+"_ROOT"][0] = 1.

            # Not really a weight, but will be useful

#            weights["Erec"] = [eventDF["Erec"]]
            treeVars["Erec"][0] = eventDF["Erec"]
            
            if binnedWeights == None : 
                treeVars["IsSelected"][0] = 0
            else :
                treeVars["IsSelected"][0] = 1 
#            weights["IsSelected"] = [ not (binnedWeights == None) ]

#            df.append(pd.DataFrame(weights))
                
            # Remove pesky nans
            for key, value in treeVars.iteritems() :
                if np.isnan(value[0]) :
                    treeVars[key][0] = 1.

            outTree.Fill()

            if not eventCounter%1000 :
                print eventCounter
            eventCounter += 1

        outTree.Write()
        outFile.Close()
#        df = pd.concat(df)

#        df.fillna(1.)

#        outFname = os.path.splitext(filePath)[0]+'_'+nuModeSample.name+'_weights.root'

#        array2root(arr = df.to_records(index = False), filename = outFname, treename = 'rw_'+nuModeSample.name, mode = 'recreate')
