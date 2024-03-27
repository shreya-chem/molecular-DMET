import localintegrals, dmet, qcdmet_paths
from pyscf import gto, scf, symm
from pyscf import mcscf
import numpy as np

localization = 'meta_lowdin'                # 'iao' or 'meta_lowdin' or 'boys'

bondlengths = [1.0] # np.linspace(0.2, 2.0, num=15) 
energies = []

print(bondlengths)

mol = gto.Mole()
#mol.atom = (['H', [0, 0, 0]], ('H', [0, 0, bondlengths[0]]))
mol.basis = '6-31g'

for bondlength in bondlengths:

    mol.atom = (['H', [0, 0, 0]], ('H', [0, 0, bondlength]))
    mol.build(verbose = 4)
    mf = scf.RHF(mol)
    mf.max_cycle = 1000
    mf.scf()

    if ( True ):
        myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), localization )
        myInts.molden( 'hydrogen-loc.molden' )
        myInts.TI_OK = True # Only s functions

        unit_sizes = np.array([2,2])
        assert( np.sum( unit_sizes ) == mol.nao_nr() )

        impurityClusters = [[1,1,0,0]]
        isTranslationInvariant = False
        method = 'CASPDFT'
        SCmethod = 'NONE'
        theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant, method, SCmethod, doDET=True )
        theDMET.impCAS = (2,4)
        the_energy = theDMET.doselfconsistent()
        energies.append(the_energy)
        print ("----Energy at ", bondlength," angstrom:", the_energy)

print ("Bondlengths =", bondlengths)
print ("Energies =", energies)
