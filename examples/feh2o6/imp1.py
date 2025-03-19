import sys
from pyscf import gto, scf, symm
from pyscf import mcscf
import numpy as np
import localintegrals, dmet, qcdmet_paths
from pyscf.tools import molden, cubegen
from functools import reduce
#############
#   Input   #
#############              
localization = 'meta_lowdin'                # 'iao' or 'meta_lowdin' or 'boys'

#np.set_printoptions(threshold=sys.maxsize)

one_bath_orb_per_bond = False        # Sun & Chan, JCTC 10, 3784 (2014) [ http://dx.doi.org/10.1021/ct500512f ]
casci_energy_formula = True         # CASCI or DMET energy formula
DMguess = None
mol = gto.Mole()
mol.atom = "/project/lgagliardi/shreyav/moldmet/feh2o6/feh2o62plus-hs.xyz"
mol.basis = {'O':'cc-pvdz', 'Fe':'cc-pvtz', 'H':'cc-pvdz'}
mol.charge = 2
mol.build(verbose=4)
mf = mol.RHF().run()
mf.verbose = 4
mf.max_cycle = 1000
mf.scf(DMguess)
DMguess = mf.make_rdm1()
basis = '6-31g'

if ( True ):
    myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), localization )
    unit_sizes = None
    frag = 68
    unit_sizes = np.array([frag,mol.nao_nr()-frag]) # HNN, 4(CH2), CH3 # SV cGTOs in respective fragments
    assert( np.sum( unit_sizes ) == mol.nao_nr() )

    impurityClusters = []
    if ( casci_energy_formula ): # Do only 1 impurity at the edge
        num_orb_in_imp = np.sum( unit_sizes[ 0 : 1 ] ) # getting total cGTOs in 1st fragment
        impurity_orbitals = np.zeros( [ mol.nao_nr() ], dtype=int ) # creating an array of size total cGTOs
        impurity_orbitals[ 0 : num_orb_in_imp ] = 1
        impurityClusters.append( impurity_orbitals ) # impurityClusters is now [1,1,...,0,...,0] with 1s for first 20 indices
    else: #doesn't go in this block
        jump = 0
        for fragment in range(unit_sizes.size):
            impurity_orbitals = np.zeros( [ mol.nao_nr() ], dtype=int )
            num_orb_in_imp = unit_sizes[fragment]
            if (fragment > 0):
                impurity_orbitals[ jump : jump + num_orb_in_imp ] = -1
            else:
                impurity_orbitals[ jump : jump + num_orb_in_imp ] = 1
            impurityClusters.append( impurity_orbitals )
            jump += num_orb_in_imp

    isTranslationInvariant = False
    method = 'SA-CASPDFT'
    SCmethod = 'NONE'
    theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant, method, SCmethod, doDET=True )
    theDMET.impCAS = (6,10)
    theDMET.doSCF = True
    theDMET.state_average_ = [1/3]*3
    theDMET.fix_spin_val = 6
    theDMET.nroots = 10
    theDMET.nevpt2_roots = np.arange(0,10,1)
    theDMET.nevpt2_nroots = 10
#    theDMET.CASlist = [40,41,42,43,44]
    theDMET.CC_E_TYPE  = 'CASCI'
    the_energy = theDMET.doselfconsistent()
    print ("----Energy:", the_energy)
    print ("Occupation number:",theDMET.OccNum)
        #print and plot orbitals

    X1 = theDMET.ao2loc
    X2 = theDMET.loc2dmet
    nemb = theDMET.numBathOrbs + frag
    for mo in range(0,nemb):
        mo_coeff = reduce(np.dot,(X1,X2[:,:nemb], theDMET.MO))[:,mo].reshape(mol.nao_nr())
        name = 'MO_' + str(mo) + '.cube'
        cubegen.orbital(mol, name, mo_coeff, nx=60, ny=60, nz=60)
    for mo in range(0,nemb):
        mo_coeff = reduce(np.dot,(X1,X2[:,:nemb], theDMET.MOnat))[:,mo].reshape(mol.nao_nr())
        name = 'MO_nat_' + str(mo) + '.cube'
        cubegen.orbital(mol, name, mo_coeff, nx=60, ny=60, nz=60)
