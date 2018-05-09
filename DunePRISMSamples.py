from MagicRWSample import Sample

import pandas as pd

m_proton = 0.93827208

class Nominal(Sample) :
    
    def __init__(self, outFilePath, inFilePath) :
        super(Nominal, self).__init__(name = "Nominal_ND_stop0_FHC", outFilePath = outFilePath, inFilePath = inFilePath, trainFrac = 0.75)

    def leptonEnergy(self, event) :
        return event.PrimaryLep_4mom[3]

    def protonEdepFV(self, event) :
        return event.ProtonDep_FV

    def protonEdepVeto(self, event) :
        return event.ProtonDep_veto

    def protonEdep(self, event) :
        return self.protonEdepFV(event) + self.protonEdepVeto(event)

    def neutronEdepFV(self, event) :
        return event.NeutronDep_FV

    def neutronEdepVeto(self, event) :
        return event.NeutronDep_veto

    def neutronEdep(self, event) :
        return self.neutronEdepFV(event) + self.neutronEdepVeto(event)

    def piCEdepFV(self, event) :
        return event.PiCDep_FV

    def piCEdepVeto(self, event) :
        return event.PiCDep_veto

    def piCEdep(self, event) :
        return self.piCEdepFV(event) + self.piCEdepVeto(event)

    def pi0EdepFV(self, event) :
        return event.Pi0Dep_FV

    def pi0EdepVeto(self, event) :
        return event.Pi0Dep_veto

    def pi0Edep(self, event) :
        return self.pi0EdepFV(event) + self.pi0EdepVeto(event)

    def otherEdepVeto(self, event) :
        return event.OtherDep_veto

    def otherEdepFV(self, event) :
        return event.OtherDep_FV
    
    def otherEdep(self, event) :
        return self.otherEdepFV(event) + self.otherEdepVeto(event)
    
    def nonLepDepVeto(self, event) :
        return self.protonEdepVeto(event) + self.neutronEdepVeto(event) + self.piCEdepVeto(event) + self.pi0EdepVeto(event) + self.otherEdepVeto(event)

    def nonLepDepFV(self, event) :
        return self.protonEdepFV(event) + self.neutronEdepFV(event) + self.piCEdepFV(event) + self.pi0EdepFV(event) + self.otherEdepFV(event)
    
    def nonLepDep(self, event) :
        return self.nonLepDepVeto(event) + self.nonLepDepFV(event)
    
    def bindingEnergy(self) :
        return 0.0323
    
    def Erec(self, event) :
        return self.nonLepDep(event) + self.leptonEnergy(event) + self.bindingEnergy()

    def q0(self, event) :
        return event.FourMomTransfer_True[3]

    def q3(self, event) :
        return sum ( [ event.FourMomTransfer_True[i]**2 for i in range(0,3) ] )**0.5

    def w(self, event) :
        w2 = - event.Q2_True + 2*self.q0(event)*m_proton + m_proton**2

        if w2 > 0 :
            return w2**0.5
        else :
            return 0 # Sometimes (rarely) w^2 is negative (?!), return 0 in those cases

    def Q2(self, event) :
        return event.Q2_True

    def GENIEIntMode(self, event) :
        return event.GENIEInteractionTopology

    def protonEKinTrue(self, event) :
        return event.EKinProton_True

    def Etrue(self, event) :
        return event.nu_4mom[3]

    def selection(self, event) :
        isSelected = True

        if not event.IsCC :
            isSelected = False
            return isSelected
        if self.nonLepDepVeto(event) > 0.05 :
            isSelected = False
            return isSelected
        if event.stop != 0 :
            isSelected = False
            return isSelected
        if event.PrimaryLepPDG != 13 :
            isSelected = False
            return isSelected
            
        return isSelected

    # Variables to be used in training
    observables = { "Erec"        : { "label" : r'E$_{\mathrm{rec}}$ [GeV]',               "range" : [0., 6.] , "logScale" : False },
                    "Elep_true"   : { "label" : r'E${_{\ell}}^{\mathrm{true}}$ [GeV]}',    "range" : [0., 5.] , "logScale" : False },
                    "Eproton_dep" : { "label" : r'E${_{p}}^{\mathrm{dep}}$ [GeV]',         "range" : [0., 2.] , "logScale" : True },
                    "EpiC_dep"    : { "label" : r'E${_{\pi^{\pm}}}^{\mathrm{dep}}$ [GeV]', "range" : [0., 2.] , "logScale" : True },
                    "Epi0_dep"    : { "label" : r'E${_{\pi^{0}}}^{\mathrm{dep}}$ [GeV]',   "range" : [0., 2.] , "logScale" : True } }
                    
    
    # Pairs of true variables for binned reweighting
    trueVarPairs = [ ["q0", "q3"], ["EKproton_True", "Etrue"], ["w", "Etrue"] , ["Q2", "Etrue"] ]
    
    def variables(self, event) :

        variables = { "Erec" :           self.Erec(event),
                      "Elep_true" :      self.leptonEnergy(event),
                      "Eproton_dep" :    self.protonEdep(event), 
                      "EpiC_dep" :       self.piCEdep(event),
                      "Epi0_dep" :       self.pi0Edep(event),
                      "Etrue" :          self.Etrue(event),
                      "q0" :             self.q0(event),
                      "q3" :             self.q3(event),
                      "w" :              self.w(event),
                      "Q2" :             self.Q2(event),
                      "GENIEIntMode" :   self.GENIEIntMode(event),
                      "EKproton_True" :  self.protonEKinTrue(event),
        }

        return variables

class ProtonEdepm20pc(Nominal) :
    
    def __init__(self, outFilePath, inFilePath) :
        super(Nominal, self).__init__(name = "ProtonEdepm20pc_ND_stop0_FHC", outFilePath = outFilePath, inFilePath = inFilePath, trainFrac = 0.75)

    def protonEdepFV(self, event) :
        return 0.8*event.ProtonDep_FV

    def protonEdepVeto(self, event) :
        return 0.8*event.ProtonDep_veto
