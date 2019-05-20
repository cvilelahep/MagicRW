import treelite
import pandas as pd


annotate = False

modelPath = "/dune/data/users/cvilela/MagicRW/mcc11_v4_FHC_RecoLepE_RecoY_200Trees_HackDays/ProtonEdepm20pc/ProtonEdepm20pc_trueKinBDT.xgb"
outPath = "/dune/data/users/cvilela/MagicRW/mcc11_v4_FHC_RecoLepE_RecoY_200Trees_HackDays/ProtonEdepm20pc/"

model = treelite.Model.load(modelPath, model_format='xgboost')

model.export_srcpkg(platform='unix', toolchain='gcc',
                    pkgpath=outPath+'ProtonEdepm20pc_trueKinBDT.zip', libname='ProtonEdepm20pc_trueKinBDT.so',
                    verbose=True) #, params = {"native_lib_name" : "aname", "annotate_in" : "mymodel-annotation.json"} )
