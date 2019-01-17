import ROOT
import datetime

from multiprocessing import Process

from DunePRISMSamples import *

pickle = True
train = True
produceBinned = True
produceROOT = True

plot = True
plotBinned = True
plotBinnedROOT = True

pickle = False
train = False
#produceBinned = False
#produceROOT = False

#plot = False
#plotBinned = False
#plotBinnedROOT = False

FHC_nominalFilePath = "/dune/data/users/cvilela/CAFs/mcc11_v3/ND_FHC_CAF_nom.root"
FHC_fakeFilePath = "/dune/data/users/cvilela/CAFs/mcc11_v3/ND_FHC_CAF_fake.root"
FHC_outFilePath = "/dune/data/users/cvilela/MagicRW/FHC/"

FD_FHC_nominalFilePath = "/dune/data/users/marshalc/CAFs/mcc11_v3/FD_FHC_nonswap.root"
FD_FHC_fakeFilePath = "/dune/data/users/marshalc/CAFs/mcc11_v3/FD_FHC_nonswap.root"
FD_FHC_outFilePath = "/dune/data/users/cvilela/MagicRW/FHC_FD/"
#FHC_outFilePathTV = "/gpfs/scratch/crfernandesv/MagicRW/FHC_SamplesTV"
#FHC_outFilePathTV_Neutron = "/gpfs/scratch/crfernandesv/MagicRW/FHC_SamplesTV_Neutron"

RHC_nominalFilePath = "/dune/data/users/cvilela/CAFs/mcc11_v3/ND_RHC_CAF_nom.root"
RHC_fakeFilePath = "/dune/data/users/cvilela/CAFs/mcc11_v3/ND_RHC_CAF_fake.root"
RHC_outFilePath = "/dune/data/users/cvilela/MagicRW/RHC/"
#RHC_outFilePathTV = "/gpfs/scratch/crfernandesv/MagicRW/RHC_SamplesTV"
#RHC_outFilePathTV_Neutron = "/gpfs/scratch/crfernandesv/MagicRW/RHC_SamplesTV_Neutron"

# BELOW "chargeSel" means sign of the PDG code, so + is particle, and - antiparticle - selecting the outgoing muon.
samplesOA_FHC = [ Nominal(         inFilePath = FHC_nominalFilePath, outFilePath = FHC_outFilePath, chargeSel=-1),
                  ProtonEdepm20pc( inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePath, chargeSel=-1),
                  PionEdepm20pc(   inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePath, chargeSel=-1),
                  ProtonEdepm20pcA(inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePath, chargeSel=-1) ]

#samplesOA_TV_FHC = [NominalTV(         inFilePath = FHC_nominalFilePath, outFilePath = FHC_outFilePathTV, chargeSel=+1), 
#                    ProtonEdepm20pcTV( inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV, chargeSel=+1),
#                    PionEdepm20pcTV(   inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV, chargeSel=+1),
#                    ProtonEdepm20pcATV(inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV, chargeSel=+1)]
#
#samplesNeutron_TV_FHC = [ NominalTV_Neutron(         inFilePath = FHC_nominalFilePath, outFilePath = FHC_outFilePathTV_Neutron, chargeSel=+1), 
#                        ProtonEdepm20pcTV_Neutron( inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV_Neutron, chargeSel=+1),
#                        PionEdepm20pcTV_Neutron(   inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV_Neutron, chargeSel=+1),
#                        ProtonEdepm20pcATV_Neutron(inFilePath = FHC_fakeFilePath,    outFilePath = FHC_outFilePathTV_Neutron, chargeSel=+1)]

samplesOA_RHC = [Nominal(         inFilePath = RHC_nominalFilePath, outFilePath = RHC_outFilePath, chargeSel=+1), 
                 ProtonEdepm20pc( inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePath, chargeSel=+1),
                 PionEdepm20pc(   inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePath, chargeSel=+1),
                 ProtonEdepm20pcA(inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePath, chargeSel=+1)]

#samplesOA_TV_RHC = [ NominalTV(         inFilePath = RHC_nominalFilePath, outFilePath = RHC_outFilePathTV, chargeSel=-1), 
#                     ProtonEdepm20pcTV( inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV, chargeSel=-1),
#                     PionEdepm20pcTV(   inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV, chargeSel=-1),
#                     ProtonEdepm20pcATV(inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV, chargeSel=-1)]
#
#samplesNeutron_TV_RHC = [ NominalTV_Neutron(         inFilePath = RHC_nominalFilePath, outFilePath = RHC_outFilePathTV_Neutron, chargeSel=-1), 
#                        ProtonEdepm20pcTV_Neutron( inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV_Neutron, chargeSel=-1),
#                        PionEdepm20pcTV_Neutron(   inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV_Neutron, chargeSel=-1),
#                        ProtonEdepm20pcATV_Neutron(inFilePath = RHC_fakeFilePath,    outFilePath = RHC_outFilePathTV_Neutron, chargeSel=-1)]
#
#samples = [  samplesOA_FHC, samplesOA_TV_FHC, samplesNeutron_TV_FHC, samplesOA_RHC, samplesOA_TV_RHC, samplesNeutron_TV_RHC ]
#samples = [ samplesNeutron_TV_FHC, samplesNeutron_TV_FHC ]

samplesFD_FHC = [ Nominal_FD(         inFilePath = FD_FHC_nominalFilePath, outFilePath = FD_FHC_outFilePath, chargeSel = 0), 
                  ProtonEdepm20pc_FD( inFilePath = FD_FHC_fakeFilePath,    outFilePath = FD_FHC_outFilePath, chargeSel = 0) ]

samples = [  samplesOA_FHC ]
#samples = [  samplesOA_RHC ]

#samples = [ samplesFD_FHC ]

#samples = [  samplesOA_FHC, samplesOA_RHC ] # Doesn't seem to be working. Run one at a time

if pickle :
    print "Starting Pickle"
    print(datetime.datetime.now())
    processesPickle = []
    for sample in samples :
        for s in sample : 
            processesPickle.append( Process( target = s.pickleData ) )

    for p in processesPickle :
        p.start()
    for p in processesPickle :
        p.join()

    print "Pickle Ended"
    print(datetime.datetime.now())



if train :
    print "Starting training"
    print(datetime.datetime.now())
    processesPickle = []

    for sample in samples :
        sNom = sample[0]
        for s in sample[1:] :
            processesPickle.append( Process( target = s.trainBDT, args=(sNom,) ) )
        
    for p in processesPickle :
        p.start()
    for p in processesPickle :
        p.join()
    print "Training ended"
    print(datetime.datetime.now())

if plot :
    print "Plotting diagnostics"
    print(datetime.datetime.now())
    processesPickle = []

    for sample in samples :
        sNom = sample[0]
        for s in sample[1:] :
            processesPickle.append( Process( target = s.plotDiagnostics, args=(sNom,) ) )

    for p in processesPickle :
        p.start()
    for p in processesPickle :
        p.join()
    print "Diagnostics plotting ended"
    print(datetime.datetime.now())

if produceBinned :
    print "Producing binned weighting scheme"
    print(datetime.datetime.now())
    processesPickle = []

    for sample in samples :
        sNom = sample[0]
        for s in sample[1:] :
            processesPickle.append( Process( target = s.makeBinnedWeights ) )

    for p in processesPickle :
        p.start()
    for p in processesPickle :
        p.join()
    print "Binned weighting scheme production ended"
    print(datetime.datetime.now())

if produceROOT :
    print "Producing binned weighting scheme (ROOT)"
    print(datetime.datetime.now())
    processesPickle = []

    for sample in samples :
        sNom = sample[0]
        for s in sample[1:] :
            processesPickle.append( Process( target = s.makeROOTBinnedWeights ) )

    for p in processesPickle :
        p.start()
    for p in processesPickle :
        p.join()
    print "Binned weighting scheme production ended (ROOT)"
    print(datetime.datetime.now())

if plotBinned :
    print "Plotting binned weights diagnostics"
    print(datetime.datetime.now())
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
    print "Binned weights diagnostics plotting ended"
    print(datetime.datetime.now())

if plotBinnedROOT :
    print "Plotting binned weights diagnostics (ROOT)"
    print(datetime.datetime.now())
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
    print "Binned weights diagnostics plotting ended (ROOT)"
    print(datetime.datetime.now())
