import ROOT

from multiprocessing import Process

import DunePRISMSamples

import MagicRWSample

import os

FHC_FilePath = "/dune/data/users/cvilela/CAFs/mcc11_v3/ND_FHC_CAF.root"
RHC_FilePath = "/dune/data/users/cvilela/CAFs/mcc11_v3/ND_RHC_CAF.root" 

FD_FHC_FilePath = "/dune/data/users/cvilela/CAFs/mcc11_v3/FD_FHC_nonswap.root"

samplesOA_FHC = [ DunePRISMSamples.ProtonEdepm20pc( inFilePath = FHC_FilePath,   outFilePath = "/dune/data/users/cvilela/MagicRW/FHC/", chargeSel=+1),
                  DunePRISMSamples.PionEdepm20pc(   inFilePath = FHC_FilePath,   outFilePath = "/dune/data/users/cvilela/MagicRW/FHC/", chargeSel=+1),
                  DunePRISMSamples.ProtonEdepm20pcA(inFilePath = FHC_FilePath,   outFilePath = "/dune/data/users/cvilela/MagicRW/FHC/", chargeSel=+1) ]

FD_samplesOA_FHC = [ DunePRISMSamples.ProtonEdepm20pc_FD( inFilePath = FD_FHC_FilePath,   outFilePath = "/dune/data/users/cvilela/MagicRW/FD_FHC/", chargeSel=0) ]


samplesOA_RHC = [DunePRISMSamples.ProtonEdepm20pc( inFilePath = RHC_FilePath,   outFilePath = "/dune/data/users/cvilela/MagicRW/RHC/", chargeSel=-1),
                 DunePRISMSamples.PionEdepm20pc(   inFilePath = RHC_FilePath,    outFilePath = "/dune/data/users/cvilela/MagicRW/RHC/", chargeSel=-1),
                 DunePRISMSamples.ProtonEdepm20pcA(inFilePath = RHC_FilePath,    outFilePath = "/dune/data/users/cvilela/MagicRW/RHC/", chargeSel=-1)]


fakeDataSet = 0


#MagicRWSample.Sample.produceFriendTrees(filePath = "/dune/data/users/cvilela/CAFs/mcc11_v3/ND_FHC_CAF.root", nuModeSample = samplesOA_FHC[fakeDataSet], antinuModeSample = samplesOA_RHC[fakeDataSet])


#produceFriendTrees(filePath, nuModeSample, antinuModeSample, isNominal = False) 
processesPickle = []

for fakeDataSet in range(0, len(samplesOA_FHC) ) :
    processesPickle.append( Process( target = MagicRWSample.Sample.produceFriendTrees, args = ("/dune/data/users/cvilela/CAFs/mcc11_v3/ND_FHC_CAF.root", samplesOA_FHC[fakeDataSet], samplesOA_RHC[fakeDataSet],) ) )
    processesPickle.append( Process( target = MagicRWSample.Sample.produceFriendTrees, args = ("/dune/data/users/cvilela/CAFs/mcc11_v3/ND_RHC_CAF.root", samplesOA_FHC[fakeDataSet], samplesOA_RHC[fakeDataSet],) ) )


processesPickle.append( Process( target = MagicRWSample.Sample.produceFriendTrees, args = (FD_FHC_FilePath, samplesOA_FHC[0], samplesOA_RHC[0], False, True) ) )


for p in processesPickle :
    p.start()
for p in processesPickle :
    p.join()
