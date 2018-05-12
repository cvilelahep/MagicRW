import ROOT

from multiprocessing import Process

from DunePRISMSamples import Nominal, ProtonEdepm20pc, PionEdepm20pc, ProtonEdepm20pcA

pickle = False
train = False
plot = True

nominalFilePath = "/gpfs/scratch/crfernandesv/DunePrism/FHC/4855489.*[0-4].Processed.root"
fakeFilePath = "/gpfs/scratch/crfernandesv/DunePrism/FHC/4855489.*[5-9].Processed.root"
outFilePath = "/gpfs/home/crfernandesv/MagicRW/Samples"

samples = [ Nominal(         inFilePath = nominalFilePath, outFilePath = outFilePath), 
            ProtonEdepm20pc( inFilePath = fakeFilePath,    outFilePath = outFilePath),
            PionEdepm20pc(   inFilePath = fakeFilePath,    outFilePath = outFilePath),
            ProtonEdepm20pcA(inFilePath = fakeFilePath,    outFilePath = outFilePath) ]


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

