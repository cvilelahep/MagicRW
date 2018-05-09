import ROOT

from DunePRISMSamples import Nominal, ProtonEdepm20pc

nom = Nominal(inFilePath = "/home/cvilela/DunePRISM/SomePlots/4855489.*.Processed.root", outFilePath = "/home/cvilela/MagicRWFramework")
ProtonEdepm20pc = ProtonEdepm20pc(inFilePath = "/home/cvilela/DunePRISM/SomePlots/4855489.*.Processed.root", outFilePath = "/home/cvilela/MagicRWFramework")

#nom.pickleData()
#ProtonEdepm20pc.pickleData()

ProtonEdepm20pc.trainBDT(nom)
ProtonEdepm20pc.plotDiagnostics(nom)
