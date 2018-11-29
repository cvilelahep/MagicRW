import ROOT

from multiprocessing import Process

from DunePRISMSamples import *

pickle = True
train = True
produceBinned = True
produceROOT = True

plot = True
plotBinned = True
plotBinnedROOT = True

FHC_nominalFilePath = "/gpfs/scratch/crfernandesv/DunePrism/FHC/4855489.*[0-4].Processed_mergedWeights.root"
FHC_fakeFilePath = "/gpfs/scratch/crfernandesv/DunePrism/FHC/4855489.*[5-9].Processed_mergedWeights.root"
FHC_outFilePath = "/gpfs/scratch/crfernandesv/MagicRW/FHC_Samples"
FHC_outFilePathTV = "/gpfs/scratch/crfernandesv/MagicRW/FHC_SamplesTV"
FHC_outFilePathTV_Neutron = "/gpfs/scratch/crfernandesv/MagicRW/FHC_SamplesTV_Neutron"

RHC_nominalFilePath = "/gpfs/scratch/crfernandesv/DunePrism/RHC/4855497.*[0-4].Processed_mergedWeights.root"
RHC_fakeFilePath = "/gpfs/scratch/crfernandesv/DunePrism/RHC/4855497.*[5-9].Processed_mergedWeights.root"
RHC_outFilePath = "/gpfs/scratch/crfernandesv/MagicRW/RHC_Samples"
RHC_outFilePathTV = "/gpfs/scratch/crfernandesv/MagicRW/RHC_SamplesTV"
RHC_outFilePathTV_Neutron = "/gpfs/scratch/crfernandesv/MagicRW/RHC_SamplesTV_Neutron"

# BELOW "chargeSel" means sign of the PDG code, so + is particle, and - antiparticle - selecting the outgoing muon.
samplesOA_FHC = [ Nominal(         inFilePath = FHC_nominalFilePath, outFilePath = FHC_outFilePath, chargeSel=+1),
                  ProtonEdepm20pc( inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePath, chargeSel=+1),
                  PionEdepm20pc(   inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePath, chargeSel=+1),
                  ProtonEdepm20pcA(inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePath, chargeSel=+1) ]

samplesOA_TV_FHC = [NominalTV(         inFilePath = FHC_nominalFilePath, outFilePath = FHC_outFilePathTV, chargeSel=+1), 
                    ProtonEdepm20pcTV( inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV, chargeSel=+1),
                    PionEdepm20pcTV(   inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV, chargeSel=+1),
                    ProtonEdepm20pcATV(inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV, chargeSel=+1)]

samplesNeutron_TV_FHC = [ NominalTV_Neutron(         inFilePath = FHC_nominalFilePath, outFilePath = FHC_outFilePathTV_Neutron, chargeSel=+1), 
                        ProtonEdepm20pcTV_Neutron( inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV_Neutron, chargeSel=+1),
                        PionEdepm20pcTV_Neutron(   inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV_Neutron, chargeSel=+1),
                        ProtonEdepm20pcATV_Neutron(inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV_Neutron, chargeSel=+1)]

samplesOA_RHC = [Nominal(         inFilePath = RHC_nominalFilePath, outFilePath = RHC_outFilePath, chargeSel=-1), 
                 ProtonEdepm20pc( inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePath, chargeSel=-1),
                 PionEdepm20pc(   inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePath, chargeSel=-1),
                 ProtonEdepm20pcA(inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePath, chargeSel=-1)]

samplesOA_TV_RHC = [ NominalTV(         inFilePath = RHC_nominalFilePath, outFilePath = RHC_outFilePathTV, chargeSel=-1), 
                     ProtonEdepm20pcTV( inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV, chargeSel=-1),
                     PionEdepm20pcTV(   inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV, chargeSel=-1),
                     ProtonEdepm20pcATV(inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV, chargeSel=-1)]

samplesNeutron_TV_RHC = [ NominalTV_Neutron(         inFilePath = RHC_nominalFilePath, outFilePath = RHC_outFilePathTV_Neutron, chargeSel=-1), 
                        ProtonEdepm20pcTV_Neutron( inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV_Neutron, chargeSel=-1),
                        PionEdepm20pcTV_Neutron(   inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV_Neutron, chargeSel=-1),
                        ProtonEdepm20pcATV_Neutron(inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV_Neutron, chargeSel=-1)]

#samples = [  samplesOA_FHC, samplesOA_TV_FHC, samplesNeutron_TV_FHC, samplesOA_RHC, samplesOA_TV_RHC, samplesNeutron_TV_RHC ]
#samples = [ samplesNeutron_TV_FHC, samplesNeutron_TV_FHC ]

samples = [  samplesOA_FHC ]


if pickle :
    processesPickle = []
    for sample in samples :
        for s in sample : 
            processesPickle.append( Process( target = s.pickleData ) )

    for p in processesPickle :
        p.start()
    for p in processesPickle :
        p.join()

if train :
    processesPickle = []

    for sample in samples :
        sNom = sample[0]
        for s in sample[1:] :
            processesPickle.append( Process( target = s.trainBDT, args=(sNom,) ) )
        
    for p in processesPickle :
        p.start()
    for p in processesPickle :
        p.join()

if plot :
    processesPickle = []

    for sample in samples :
        sNom = sample[0]
        for s in sample[1:] :
            processesPickle.append( Process( target = s.plotDiagnostics, args=(sNom,) ) )

    for p in processesPickle :
        p.start()
    for p in processesPickle :
        p.join()

if produceBinned :
    processesPickle = []

    for sample in samples :
        sNom = sample[0]
        for s in sample[1:] :
            processesPickle.append( Process( target = s.makeBinnedWeights ) )

    for p in processesPickle :
        p.start()
    for p in processesPickle :
        p.join()

if produceROOT :
    processesPickle = []

    for sample in samples :
        sNom = sample[0]
        for s in sample[1:] :
            processesPickle.append( Process( target = s.makeROOTBinnedWeights ) )

    for p in processesPickle :
        p.start()
    for p in processesPickle :
        p.join()

if plotBinned :
    processesPickle = []

    for sample in samples :
        sNom = sample[0]
        for s in sample[1:] :
            processesPickle.append( Process( target = s.plotDiagnostics, args=(sNom, "EnuTp",) ) )
            processesPickle.append( Process( target = s.plotDiagnostics, args=(sNom, "ElTp",) ) )
            processesPickle.append( Process( target = s.plotDiagnostics, args=(sNom, "q0q3",) ) )
            processesPickle.append( Process( target = s.plotDiagnostics, args=(sNom, "EnuW",) ) )
            processesPickle.append( Process( target = s.plotDiagnostics, args=(sNom, "EnuQ2",) ) )

    for p in processesPickle :
        p.start()
    for p in processesPickle :
        p.join()

if plotBinnedROOT :
    processesPickle = []

    for sample in samples :
        sNom = sample[0]
        for s in sample[1:] :
            processesPickle.append( Process( target = s.plotDiagnostics, args=(sNom, "EnuTp",True,) ) )
            processesPickle.append( Process( target = s.plotDiagnostics, args=(sNom, "ElTp",True,) ) )
            processesPickle.append( Process( target = s.plotDiagnostics, args=(sNom, "q0q3",True,) ) )
            processesPickle.append( Process( target = s.plotDiagnostics, args=(sNom, "EnuW",True,) ) )
            processesPickle.append( Process( target = s.plotDiagnostics, args=(sNom, "EnuQ2",True,) ) )

    for p in processesPickle :
        p.start()
    for p in processesPickle :
        p.join()


