from random import random
from abc import ABCMeta, abstractmethod, abstractproperty
import pandas as pd
from ROOT import TFile, TChain
import cPickle as pickle
import os
from hep_ml.reweight import GBReweighter
from corner import corner

class Sample(object) :
    
    __metaclass__ = ABCMeta
#   
    def __init__(self, name, outFilePath, inFilePath, trainFrac = 0.0):
        self.name = name
        self.outFilePath = outFilePath
        self.inFilePath = inFilePath
        self.trainFrac = trainFrac
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
        EDeps = TChain("EDeps")
        EDeps.Add(self.inFilePath)

        df = pd.DataFrame( [ self.variables(event)
                             for event in EDeps
                             if self.selection(event) ])
        del EDeps
        
        if (self.trainFrac == 0.0) or (self.trainFrac == 1.0) :
            return df
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
        
        originDF = originDF[self.observables.keys()]
        targetDF = targetDF[targetSample.observables.keys()]

        reweighter = GBReweighter(n_estimators=200,
                                  learning_rate=.1,
                                  max_depth=3,
                                  min_samples_leaf=1000,
                                  loss_regularization=1.0)

        reweighter.fit(original = originDF, target = targetDF)

        with open(self.gbrwPath(), "wb") as f :
            pickle.dump(reweighter, f)

    def plotDiagnostics(self, targetSample) :
        
        if not os.path.isdir(self.plotsDir()) :
            os.makedirs(self.plotsDir())

        targetDFtrain, targetDFtest = targetSample.getDataFrames()
        originDFtrain, originDFtest = self.getDataFrames()

        targetDFtrain = targetDFtrain[self.observables.keys()]
        originDFtrain = originDFtrain[self.observables.keys()]
        targetDFtest = targetDFtest[self.observables.keys()]
        originDFtest = originDFtest[self.observables.keys()]
        
        with open(self.gbrwPath(), 'r') as f :
            reweighter = pickle.load(f)

        weightsTest  = reweighter.predict_weights(originDFtest)
        weightsTrain = reweighter.predict_weights(originDFtrain)

        figTarget = self.getCornerPlot(dataFrame = targetDFtest, color = 'r', label = 'Nominal' )
        figTarget.savefig(self.plotsDir()+"corner_"+self.name+"_targetOnly.png", transparent = True, figsize = {50, 50}, dpi = 240)

        figOrigin = self.getCornerPlot(dataFrame = originDFtest, color = 'b', label = self.name, fig = figTarget )
        figTarget.savefig(self.plotsDir()+"corner_"+self.name+"_target_originNotRW.png", transparent = True, figsize = {50, 50}, dpi = 240)

        figOriginRW = self.getCornerPlot(dataFrame = originDFtest, color = 'g', label = self.name , weights = weightsTest, fig = figTarget)
        figTarget.savefig(self.plotsDir()+"corner_"+self.name+".png", transparent = True, figsize = {50, 50}, dpi = 240)
        
            
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
                fig.axes[i*5+i].set_yscale("log")
                fig.axes[i*5+i].set_ylim(ymin = 1)
                fig.axes[i*5+i].yaxis.set_label_position("right")

        return fig
    
    def produceFriendTrees(self, filePaths) :
        pass
    
