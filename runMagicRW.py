import ROOT

from multiprocessing import Process

from DunePRISMSamples import *

pickle = True
train = True
plot = True

FHC_nominalFilePath = "/gpfs/scratch/crfernandesv/DunePrism/FHC/4855489.*[0-4].Processed.root"
FHC_fakeFilePath = "/gpfs/scratch/crfernandesv/DunePrism/FHC/4855489.*[5-9].Processed.root"
FHC_outFilePath = "/gpfs/home/crfernandesv/MagicRW/FHC_Samples"
FHC_outFilePathTV = "/gpfs/home/crfernandesv/MagicRW/FHC_SamplesTV"
FHC_outFilePathTV_PRISM = "/gpfs/home/crfernandesv/MagicRW/FHC_SamplesTV_PRISM"

RHC_nominalFilePath = "/gpfs/scratch/crfernandesv/DunePrism/RHC/4855497.*[0-4].Processed.root"
RHC_fakeFilePath = "/gpfs/scratch/crfernandesv/DunePrism/RHC/4855497.*[5-9].Processed.root"
RHC_outFilePath = "/gpfs/home/crfernandesv/MagicRW/RHC_Samples"
RHC_outFilePathTV = "/gpfs/home/crfernandesv/MagicRW/RHC_SamplesTV"
RHC_outFilePathTV_PRISM = "/gpfs/home/crfernandesv/MagicRW/RHC_SamplesTV_PRISM"

# BELOW "chargeSel" means sign of the PDG code, so + is particle, and - antiparticle - selecting the outgoing muon.
samples = [ Nominal(         inFilePath = FHC_nominalFilePath, outFilePath = FHC_outFilePath, chargeSel=+1),
            ProtonEdepm20pc( inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePath, chargeSel=+1),
            PionEdepm20pc(   inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePath, chargeSel=+1),
            ProtonEdepm20pcA(inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePath, chargeSel=+1),
            NominalTV(         inFilePath = FHC_nominalFilePath, outFilePath = FHC_outFilePathTV, chargeSel=+1), 
            ProtonEdepm20pcTV( inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV, chargeSel=+1),
            PionEdepm20pcTV(   inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV, chargeSel=+1),
            ProtonEdepm20pcATV(inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV, chargeSel=+1),
            NominalTV_PRISM(         inFilePath = FHC_nominalFilePath, outFilePath = FHC_outFilePathTV_PRISM, chargeSel=+1), 
            ProtonEdepm20pcTV_PRISM( inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV_PRISM, chargeSel=+1),
            PionEdepm20pcTV_PRISM(   inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV_PRISM, chargeSel=+1),
            ProtonEdepm20pcATV_PRISM(inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV_PRISM, chargeSel=+1),
            Nominal(         inFilePath = RHC_nominalFilePath, outFilePath = RHC_outFilePath, chargeSel=-1), 
            ProtonEdepm20pc( inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePath, chargeSel=-1),
            PionEdepm20pc(   inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePath, chargeSel=-1),
            ProtonEdepm20pcA(inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePath, chargeSel=-1),
            NominalTV(         inFilePath = RHC_nominalFilePath, outFilePath = RHC_outFilePathTV, chargeSel=-1), 
            ProtonEdepm20pcTV( inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV, chargeSel=-1),
            PionEdepm20pcTV(   inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV, chargeSel=-1),
            ProtonEdepm20pcATV(inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV, chargeSel=-1),
            NominalTV_PRISM(         inFilePath = RHC_nominalFilePath, outFilePath = RHC_outFilePathTV_PRISM, chargeSel=-1), 
            ProtonEdepm20pcTV_PRISM( inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV_PRISM, chargeSel=-1),
            PionEdepm20pcTV_PRISM(   inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV_PRISM, chargeSel=-1),
            ProtonEdepm20pcATV_PRISM(inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV_PRISM, chargeSel=-1)]

if pickle :
    processesPickle = [ Process( target = s.pickleData ) for s in samples ] 

    for p in processesPickle :
        p.start()
    for p in processesPickle :
        p.join()

if train :
    processesPickle = [ Process( target = s.trainBDT, args=(samples[0],) ) for s in samples[1:] ] 

    for p in processesPickle :
        p.start()
    for p in processesPickle :
        p.join()

if plot :
    processesPickle = [ Process( target = s.plotDiagnostics, args=(samples[0],) ) for s in samples[1:] ] 

    for p in processesPickle :
        p.start()
    for p in processesPickle :
        p.join()

