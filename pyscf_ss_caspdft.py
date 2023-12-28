'''
pyscf-CASPDFT SOLVER for DMET

To use this solver, one need to modify the dmet module to recognize the pyscf_casscf.
Specifically:
line 33: assert (( method == 'ED' ) or ( method == 'CC' ) or ( method == 'MP2' ) or ( method == 'CASSCF' ))
line 257-261:
elif ( self.metihod == 'CASPDFT' ):
    import pyscf_caspdft
    assert( Nelec_in_imp % 2 == 0 )
    DMguessRHF = self.ints.dmet_init_guess_rhf( loc2dmet, Norb_in_imp, Nelec_in_imp//2, numImpOrbs, chempot_imp )
    IMP_energy, IMP_1RDM = pyscf_casscf.solve( 0.0, dmetOEI, dmetFOCK, dmetTEI, Norb_in_imp, Nelec_in_imp, numImpOrbs, DMguessRHF, chempot_imp )

History: 

- the solver is tested under FCI limit. The energy agrees with the FCI energy by chemps2 solver.
However, the energy is explosive when the active space decreasing. VERY negative! => SOLVED

- Need to improve the efficiency => SOLVED

author: Hung Pham (email: phamx494@umn.edu)
'''

import numpy as np
import localintegrals
import os, time
from pyscf.lib import logger
import sys
import qcdmet_paths
from pyscf import gto, scf, ao2mo, mcscf, lib, mrpt
from pyscf import mcpdft
from pyscf.mcpdft.otpd import get_ontop_pair_density
from pyscf.mcpdft.otfnal import otfnal, transfnal, ftransfnal
from pyscf.mcpdft import _dms
#np.set_printoptions(threshold=np.nan)

def solve(CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, impCAS, fix_spin_val, nalpha, nbeta, cas_list, DMguessRHF, loc2dmet, ao2eo, ao2eo_act, OneRDM_full_loc, mol_from_localizer, energytype='CASCI', chempot_imp=0.0, state_specific_=None, state_average_=None, state_average_mix_=None, nevpt2_roots=None, nevpt2_nroots=None, nevpt2_spin=0, printoutput=True ):
    '''
        CASPDFT with FCI solver:
            - Ground state
            - State-specfic
            - State-average
        state_specific_ is used to pass the state_id to the solver
        state_average_ is used to pass the weights to the solver
    '''    

    #SV
    '''
        CAS-PDFT Solver is implemented in this module.
    '''
    mc_dup = None
    emb_orbs=None
    OEH_type=None 
    emb_core_orbs = None
    core_orbs=None
    
    ao2dmet = ao2eo
    ao2dmet_active = ao2eo_act
    # Augment the FOCK operator with the chemical potential
    FOCKcopy = FOCK.copy()
    if (chempot_imp != 0.0):
        for orb in range(Nimp):
            FOCKcopy[ orb, orb ] -= chempot_imp
    
    # Get the RHF solution
    mol = gto.Mole()
    mol.build(verbose=0)
    mol.atom.append(('S', (0, 0, 0)))
    mol.nelectron = Nel
    mf = scf.RHF(mol)
    mol.incore_anyway = True #SV
    mol.max_memory = 4000
    mf.max_cycle = 5000
    mf.get_hcore = lambda *args: FOCKcopy
    mf.get_ovlp = lambda *args: np.eye(Norb)
    mf._eri = ao2mo.restore(8, TEI, Norb)
    mf.scf()#DMguessRHF)
    MOmf = mf.mo_coeff
    DMloc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
    if (mf.converged == False):
        mf.newton().kernel(dm0=DMloc)
    #print(mf.mo_occ)	
    '''	
    idx = mf.mo_energy.argsort()
    mf.mo_energy = mf.mo_energy[idx]
    mf.mo_coeff = mf.mo_coeff[:,idx]'''
    
    # Get the CASSCF solution
    #num_roots = nroots
    CASe = impCAS[0]
    CASorb = impCAS[1]
    nelecb = (CASe - mol.spin)//2
    neleca = CASe - nelecb
    print ("CAS Selection: (",CASe,",",CASorb,")")
    print (nalpha,",",nbeta)
    checkCAS =  (CASe <= Nel) and (CASorb <= Norb)
    if (checkCAS == False):
        CASe = Nel
        CASorb = Norb
    
    #mc = mcscf.CASSCF(mf, CASorb, CASe)
    mc = mcscf.CASSCF(mf, CASorb, (nalpha,nbeta))
    mc.fix_spin_(ss=fix_spin_val)
    mc.conv_tol = 1e-10
    mc.mol = mol #SV
    mc._scf = mf
    mc.mo_coeff = mf.mo_coeff
    mc.mo_energy = mf.mo_energy
    mc.max_cycle = 5000
    mc.chkfile = '/project/lgagliardi/shreyav/moldmet/fep-im/basis-ccpvqz-ab/q/1/fep-im-qforq-1012.chk'
    #mc.natorb = True

    #state-specific or state-average details
    
    if state_specific_ is not None:
        print ("Entering state specific")
        state_id = state_specific_
        #if not 'FakeCISolver' in str(mc.fcisolver):
            #mc = mc.state_specific_(state_id)
    elif state_specific_ is None and state_average_mix_ is None and state_average_ is not None:
        weights = state_average_
        if not 'FakeCISolver' in str(mc.fcisolver):
            mc = mc.state_average_(weights)
    elif state_average_mix_ is not None:
        solver1, solver2, weights = state_average_mix_
        mc = mcscf.state_average_mix_(self.mc, [solver1, solver2], weights)
    else:
        #nroots = 1
        #mc.fcisolver.nroots = nroots 
        print ("HERE simple casscf")
    #finished adding the said details
    
    mc.nelecas = (nalpha, nbeta)  #line 1021
    print ("SV mc.nelecas: ", mc.nelecas)
    ncorelec = mol.nelectron - (mc.nelecas[0] + mc.nelecas[1])
    #mc.ncore = ncorelec //2
    print ("SV ncorelec: ", mc.ncore)

    if cas_list is not None: 
        print('Impurity active space selection:', cas_list)
        mo = mc.mo_coeff
        #mo = lib.chkfile.load('/project/lgagliardi/shreyav/moldmet/fep-im/trunc-bath/singlet/10-12/caslist/fep-im-qfors-1012.chk', 'mcscf/mo_coeff')
        mo = mcscf.sort_mo(mc,mo,cas_list)
        E_CASSCF, E_CAS, fcivec = mc.kernel(mo)[:3]
    else:
        print ("HERE kernel")
        #mo = lib.chkfile.load('/project/lgagliardi/shreyav/moldmet/fep-im/trunc-bath/singlet/10-12/caslist/fep-im-qfors-1012.chk', 'mcscf/mo_coeff')
        mc.max_cycle = 2000
        E_CASSCF, E_CAS, fcivec = mc.kernel()[:3]
        print ("SV E_CASSCF: ", E_CASSCF)

    MO = mc.mo_coeff # save the MO coefficient and this corresponds to line 1053
    MOnat = mc.cas_natorb()[0] 
    OccNum = mc.cas_natorb()[2]	
    print('Dimension:', MO.shape[0] )	
    print('Impurity active space: ', CASe, 'electrons in ', CASorb, ' orbitals')	
    print('Impurity CASSCF energy: ', E_CASSCF)	
    #print('CASSCF orbital:', mc.mo_energy)	
    #print('NATURAL ORBITAL:')	
    #mc.analyze()
    
    #if state_specific_ is None and state_average_ is not None:
    #    E_CASSCF = np.asarray(mc.e_states)
    print ("SV mc.converged: ", mc.converged)
    if not mc.converged: print ('********  WARNING: The solver is not converged.  ********')
    if mc.converged: 
        print ('********  The solver is converged.  ********')
        SS, spin_mult = mc.fcisolver.spin_square(fcivec, CASorb, mc.nelecas)
        print ("SS: ", SS)
        #print('State %d : E(Solver) = %12.8f  <S^2> = %8.6f' % (state_id, E_CASSCF, SS)) 

    # Get 1RDM + 2RDM
    Norbcas = mc.ncas   #Number of active orbitals
    Norbcore = mc.ncore #Number of doubly occupied core orbitals
    Nelcas = mc.nelecas	
    print("SV Norbcas: ", Norbcas)
    mocore = mc.mo_coeff[:,:Norbcore]   # line 1064 
    mocas = mc.mo_coeff[:,Norbcore:Norbcore+Norbcas]    #line 1065: active_MO
            
    casdm1 = mc.fcisolver.make_rdm1(fcivec, Norbcas, Nelcas) # in CAS space
    #print ("SV casdm1: ", casdm1)
    #print ("SV casdm1 multiply: ", casdm1[0,0]*100000000)
    casdm1s = mc.fcisolver.make_rdm1s(fcivec, Norbcas, Nelcas) #
    casdm1sa, casdm1sb = mc.fcisolver.make_rdm1s(fcivec, Norbcas, Nelcas)

        # Transform the casdm1 (in CAS space) to casdm1lo (localized space).     
        # Dumb and lazy way: casdm1lo = np.einsum('pq,ap,bq->ab', casdm1, mocas, mocas) #in localized space
        
    casdm1lo = lib.einsum('ap,pq->aq', mocas, casdm1)
    casdm1lo = lib.einsum('bq,aq->ab', mocas, casdm1lo) # is in dmet basis bcoz mocas is in dmet basis and casdm1 is in cas basis
    coredm1 = np.dot(mocore, mocore.T) * 2 #in local basis
    
    casdm1losa = lib.einsum('ap,pq->aq', mocas, casdm1sa)
    casdm1losa = lib.einsum('bq,aq->ab', mocas, casdm1losa)

    casdm1losb = lib.einsum('ap,pq->aq', mocas, casdm1sb)
    casdm1losb = lib.einsum('bq,aq->ab', mocas, casdm1losb)

    OneRDM = coredm1 + casdm1lo	
    RDM1sa = coredm1/2 + casdm1losa
    RDM1sb = coredm1/2 + casdm1losb
    RDM1s = [RDM1sa,RDM1sb]
    casdm2 = mc.fcisolver.make_rdm2(mc.ci,Norbcas,Nelcas) #in CAS space
        
        # Transform the casdm2 (in CAS space) to casdm2lo (localized space). 
        # Dumb and lazy way: casdm2ao = np.einsum('pqrs,ap,bq,cr,ds->abcd',casdm2,mocas,mocas,mocas,mocas)
        
    casdm2lo = np.einsum('ap,pqrs->aqrs', mocas, casdm2)
    casdm2lo = np.einsum('bq,aqrs->abrs', mocas, casdm2lo)
    casdm2lo = np.einsum('cr,abrs->abcs', mocas, casdm2lo)
    casdm2lo = np.einsum('ds,abcs->abcd', mocas, casdm2lo)	
            
    coredm2 = np.zeros([Norb, Norb, Norb, Norb]) #in AO
    coredm2 += np.einsum('pq,rs-> pqrs',coredm1,coredm1)
    coredm2 -= 0.5*np.einsum('ps,rq-> pqrs',coredm1,coredm1)
            
    effdm2 = np.zeros([Norb, Norb, Norb, Norb]) #in AO
    effdm2 += 2*np.einsum('pq,rs-> pqrs',casdm1lo,coredm1)
    effdm2 -= np.einsum('ps,rq-> pqrs',casdm1lo,coredm1)				
            				
    TwoRDM = coredm2 + casdm2lo + effdm2	

    # Need to transform OneRDM_full_loc to OneRDM_full_dmet in order to be able to obtain the new correlated 1-RDM
    #print ("SV OneRDM_full_loc from dmet.py. This OneRDM is in the local basis: ", OneRDM_full_loc, "and its shape: ", OneRDM_full_loc.shape)
    #print ("SV OneRDM in the CAS/ MO space through CASSCF solver (Norbcas,Norbcas). This OneRDM is in the dmet basis? ", OneRDM, "and its shape: ", RDM1sa.shape)
    #OneRDM_full_dmet = lib.einsum('ji,jk,kl->il', loc2dmet, OneRDM_full_loc, loc2dmet)
    OneRDM_full_dmet = np.dot(np.dot(loc2dmet.conj().T, OneRDM_full_loc), loc2dmet)
    RDM1_dmet = OneRDM_full_dmet

    # checking for basis transformations
    #RDM1_ao_check = lib.einsum('ij,jk,lk->il', ao2dmet, RDM1_dmet, ao2dmet)
    RDM1_ao_check = np.dot(np.dot(ao2dmet, RDM1_dmet), ao2dmet.conj().T)
    print ("SV basis transformation check: ", RDM1_ao_check)
    # check over

    RDM1sa_full_dmet = OneRDM_full_dmet/2
    RDM1sb_full_dmet = OneRDM_full_dmet/2
    #print("SV OneRDM_full_dmet: ", OneRDM_full_dmet, "and its shape: ", OneRDM_full_dmet.shape)

    # Replace the (imp x imp) block in the OneRDM_full_dmet
    num_act_orbs = OneRDM.shape[0] # number of impurity orbitals will be same as the order of OneRDM (which has been brought back to the DMET space)
    OneRDM_full_dmet [:num_act_orbs,:num_act_orbs] = OneRDM [:,:]
    RDM1sa_full_dmet [:num_act_orbs,:num_act_orbs] = RDM1sa [:,:]
    RDM1sb_full_dmet [:num_act_orbs,:num_act_orbs] = RDM1sb [:,:]

    #print ("SV correlated OneRDM_full_dmet: ", OneRDM_full_dmet, "and its shape: ", OneRDM_full_dmet.shape)
    
    '''OneRDM_full_ao = lib.einsum('ij,jk,lk->il', ao2dmet, OneRDM_full_dmet, ao2dmet)
    RDM1sa_full_ao = lib.einsum('ij,jk,lk->il', ao2dmet, RDM1sa_full_dmet, ao2dmet)
    RDM1sb_full_ao = lib.einsum('ij,jk,lk->il', ao2dmet, RDM1sb_full_dmet, ao2dmet)

    
    '''
    #OneRDM_full_ao_check = np.dot(np.dot(ao2dmet, OneRDM), ao2dmet.conj().T)
    OneRDM_full_ao = np.dot(np.dot(ao2dmet, OneRDM_full_dmet), ao2dmet.conj().T)
    RDM1sa_full_ao = np.dot(np.dot(ao2dmet, RDM1sa_full_dmet), ao2dmet.conj().T)
    RDM1sb_full_ao = np.dot(np.dot(ao2dmet, RDM1sb_full_dmet), ao2dmet.conj().T)
    RDM1s_full_ao = [RDM1sa_full_ao, RDM1sb_full_ao]
    #print ("SV new OneRDM_full_ao: ", OneRDM_full_ao, "and its shape: ", OneRDM_full_ao.shape)
    #print ("SV OneRDM_full_ao_check: ", OneRDM_full_ao_check)
    #print ("SV casdm1: ", casdm1)
    
    '''RDM1sa_ao = np.einsum('ij,jk,lk->il', ao2dmet_active, RDM1sa, ao2dmet_active.T)
    RDM1sb_ao = np.einsum('ij,jk,lk->il', ao2dmet_active, RDM1sb, ao2dmet_active.T)
    '''

    RDM1sa_ao = np.dot(np.dot(ao2dmet_active, RDM1sa), ao2dmet_active.T)
    RDM1sb_ao = np.dot(np.dot(ao2dmet_active, RDM1sb), ao2dmet_active.T)
    RDM1s_ao = [RDM1sa_ao, RDM1sb_ao]

    # Transforming mo_coeff needs ao2loc and loc2dmet but only its active part
    mo_coeff_dmet = np.dot(ao2dmet_active, mc.mo_coeff [:,mc.ncore:mc.ncore+mc.ncas])
    print ("SV mc.ncore, mc.ncas: ", mc.ncore, mc.ncas)

    # Implementing NEVPT2 solver with CASSCF

    if nevpt2_roots is not None:
        print ("Entering NEVPT2")
        mf.spin = nevpt2_spin
        nelecb = (CASe - mf.spin)//2
        neleca = CASe - nelecb
        nelecas = (neleca, nelecb)
        mc_CASCI = mcscf.CASCI(mf, CASorb, (neleca, nelecb))
        mc_CASCI.fcisolver.nroots = nevpt2_nroots
        fcivec = mc_CASCI.kernel(mc.mo_coeff)[2]
        ground_state = fcivec[0]

        # Run NEVPT2
        e_casci_nevpt = []
        
        from pyscf.fci import cistring
        print("=====================================")
        if len(nevpt2_roots) > len(fcivec): nevpt2_roots = np.arange(len(fcivec))
        for root in nevpt2_roots:
            ci = fcivec[root]
            SS = mc_CASCI.fcisolver.spin_square(ci, CASorb, CASe)[0]
            e_corr = mrpt.NEVPT(mc_CASCI, root).kernel()
            if not isinstance(mc_CASCI.e_tot, np.ndarray):
                e_CASCI = mc_CASCI.e_tot
                e_nevpt = e_CASCI + e_corr
            else:
                e_CASCI = mc_CASCI.e_tot[root]
                e_nevpt = e_CASCI + e_corr
            e_casci_nevpt.append([SS, e_CASCI, e_nevpt])

            rdm1 = mc_CASCI.fcisolver.make_rdm12(ci, CASorb, CASe)[0]
            e, v = np.linalg.eig(rdm1)
            # Find the two SDs with most contribution
            strsa = np.asarray(cistring.make_strings(range(CASorb), neleca))
            strsb = np.asarray(cistring.make_strings(range(CASorb), nelecb))
            na = len(strsa)
            nb = len(strsb)

            idx_1st_max = abs(ci).argmax()
            c1 = ci.flatten()[idx_1st_max]
            stra_1st = strsa[idx_1st_max // nb]
            strb_1st = strsb[idx_1st_max % nb ]
                
            abs_fcivec = abs(ci).flatten()
            abs_fcivec[idx_1st_max] = 0.0
            idx_2nd_max = abs_fcivec.argmax()
            c2 = ci.flatten()[idx_2nd_max]
            stra_2nd = strsa[idx_2nd_max // nb]
            strb_2nd = strsb[idx_2nd_max % nb ]
            
            abs_fcivec[idx_2nd_max] = 0.0
            idx_3rd_max = abs_fcivec.argmax()
            c3 = ci.flatten()[idx_3rd_max]
            stra_3rd = strsa[idx_3rd_max // nb]
            strb_3rd = strsb[idx_3rd_max % nb ]

            abs_fcivec[idx_3rd_max] = 0.0
            idx_4th_max = abs_fcivec.argmax()
            c4 = ci.flatten()[idx_4th_max]
            stra_4th = strsa[idx_4th_max // nb]
            strb_4th = strsb[idx_4th_max % nb ]
            
            print("== State {0:d}: {1:2.4f}|{2:s},{3:s}> + {4:2.4f}|{5:s},{6:s}> + {7:2.4f}|{8:s},{9:s}> + {10:2.4f}|{11:s},{12:s}>".format(root, c1, bin(stra_1st)[2:], bin(strb_1st)[2:], c2, bin(stra_2nd)[2:], bin(strb_2nd)[2:], c3, bin(stra_3rd)[2:], bin(strb_3rd)[2:], c4, bin(stra_4th)[2:], bin(strb_4th)[2:]))
            print("   Occupancy:", e)

        e_casci_nevpt = np.asarray(e_casci_nevpt)
        print ("SV e_casci_nevpt: ", e_casci_nevpt)
        E_CASSCF = (E_CASSCF, e_casci_nevpt)

    print ("=====================================")
    print ("E_CASSCF: ", E_CASSCF)
        # To calculate the impurity energy, rescale the JK matrix with a factor 0.5 to avoid double counting: 0.5 * ( OEI + FOCK ) = OEI + 0.5 * JK
    '''
        This is the equation taken from the chemps2 solver:
        ImpurityEnergy = CONST
        ImpurityEnergy += 0.5 * np.einsum( 'ij,ij->', OneRDM[:Nimp,:], OEI[:Nimp,:] + FOCK[:Nimp,:] )
        ImpurityEnergy += 0.5 * np.einsum( 'ijkl,ijkl->', TwoRDM[:Nimp,:,:,:], TEI[:Nimp,:,:,:] )'''
        #This is the equation taken from the cc solver:
    if ( energytype == 'CASCI' ):
        ImpurityEnergy = E_CASSCF
    else:		
        ImpurityEnergy = CONST \
                           + 0.50  * np.einsum('ij,ij->',     OneRDM[:Nimp,:],     FOCK[:Nimp,:] + OEI[:Nimp,:]) \
                           + 0.125 * np.einsum('ijkl,ijkl->', TwoRDM[:Nimp,:,:,:], TEI[:Nimp,:,:,:]) \
                           + 0.125 * np.einsum('ijkl,ijkl->', TwoRDM[:,:Nimp,:,:], TEI[:,:Nimp,:,:]) \
                           + 0.125 * np.einsum('ijkl,ijkl->', TwoRDM[:,:,:Nimp,:], TEI[:,:,:Nimp,:]) \
                           + 0.125 * np.einsum('ijkl,ijkl->', TwoRDM[:,:,:,:Nimp], TEI[:,:,:,:Nimp])
    # *****  MCPDFT Energy calculation starts  *****
#    from pyscf.mcpdft.mcpdft import energy_mcwfn
 #   e_mcwfn = energy_mcwfn (mc, mo_coeff = , ot = 'tPBE', casdm1s = OneRDM_full_ao, casdm2 = casdm2)
    e_pdft = get_dmet_pdft (mc, OneRDM, 'tPBE', casdm1, casdm2, casdm1s, RDM1s_full_ao, mol_from_localizer, OneRDM_full_ao, ao2dmet, ao2dmet_active, mo_coeff_dmet, scf.RHF(mol_from_localizer))
    print ("The MC-PDFT energy is: ", e_pdft)
    print ("================================")

    return ( ImpurityEnergy, E_CASSCF, e_pdft, OneRDM, OneRDM_full_dmet, MOmf, MO, MOnat, OccNum)

def get_dmet_pdft (dmetmc, OneRDM, my_ot, casdm1, casdm2, casdm1s, RDM1s_ao, mol_full, OneRDM_fao, ao2dmet, ao2dmet_active, mo_coeff_dmet, my_mf):

    print ("***** Starting to calculate MCPDFT Energy *****")

    from copy import deepcopy
    from scipy import linalg
    from pyscf import dft, fci, lib, __config__
    from pyscf.lib import temporary_env
    from pyscf.fci import cistring
    from pyscf.dft import gen_grid
    from pyscf.mcpdft import pdft_veff
    from pyscf.mcpdft.otpd import get_ontop_pair_density
    from pyscf.mcpdft.otfnal import otfnal, transfnal, get_transfnal
    from pyscf.mcpdft import _dms

    ot_func = 'tPBE'

    ks = dft.RKS(mol_full)
    ks.xc = ot_func[1:]
    otfnal = transfnal (ks)

    grids = dft.gen_grid.Grids(mol_full)
    grids.level = 3
    otfnal.grids = grids
    otfnal.verbose = 3
    
    E_tot, E_ot = MCPDFT (dmetmc, OneRDM, otfnal, casdm1, casdm2, casdm1s, RDM1s_ao, mol_full, OneRDM_fao, ao2dmet, ao2dmet_active, mo_coeff_dmet, my_mf)
    print ("The DMET-PDFT energy is: ", E_tot)
    print ("The on-top energy is: ", E_ot)

    return (E_tot)

def MCPDFT (mc, rdm1, ot, casdm1, casdm2, casdm1s, RDM1s_ao, mol_full, OneRDM_fao, ao2dmet, ao2dmet_active, mo_coeff_dmet, my_mf):
    ''' Calculate MC-PDFT total energy
        Args:
            mc : an instance of CASSCF or CASCI class
                Note: this function does not currently run the CASSCF or CASCI calculation itself
                prior to calculating the MC-PDFT energy. Call mc.kernel () before passing to this function!
            ot : an instance of on-top density functional class - see otfnal.py
        Kwargs:
            root : int
                If mc describes a state-averaged calculation, select the root (0-indexed)
                Negative number requests state-averaged MC-PDFT results (i.e., using state-averaged density matrices)
        Returns:
            e_tot : float
            Total MC-PDFT energy including nuclear repulsion energy.
            E_ot : float
            On-top (cf. exchange-correlation) energy
    '''
    
    t0 = (logger.process_clock (), time.time ())
    dm1s = np.asarray ( mc.make_rdm1s() )
    spin = abs(mc.nelecas[0] - mc.nelecas[1])
    t0 = logger.timer (ot, 'rdms', *t0)
    omega, alpha, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
    print ("SV hyb: ", hyb)
    #print ("SV mol_full check: ", mol_full.atom)
    Vnn = mol_full.energy_nuc()
    hyb_x, hyb_c = hyb
    
    h = my_mf.get_hcore()
    dm1 = dm1s[0] + dm1s[1]
    RDM1s_aos = RDM1s_ao[0]+RDM1s_ao[1]

    if ot.verbose >= logger.DEBUG or abs (hyb_x) > 1e-10:
        vj, vk = my_mf.get_jk (dm=RDM1s_ao)
        vj = vj[0] + vj[1]
    else:
        vj = my_mf.get_j (dm=RDM1s_aos)
    print ("SV ot.verbose= ", ot.verbose)
    Te_Vne = np.tensordot (h, RDM1s_aos)
    E_j = np.tensordot (vj, RDM1s_aos) / 2

    if (ot.verbose >= logger.DEBUG)  or (abs (hyb_x) > 1e-10) :
        E_x = -(np.tensordot (vk[0], RDM1s_ao[0]) + np.tensordot (vk[1], RDM1s_ao[1])) / 2
    else:
        E_x = 0
    logger.debug (ot, 'CAS energy decomposition:')
    logger.debug (ot, 'Vnn = %s', Vnn)
    logger.debug (ot, 'Te + Vne = %s', Te_Vne)
    logger.debug (ot, 'E_j = %s', E_j)
    logger.debug (ot, 'E_x = %s', E_x)
    
    E_c = 0

    if abs (hyb_x) > 1e-10 or abs (hyb_c) > 1e-10 : 
        logger.debug (ot, 'Adding %s * %s CAS exchange to E_ot', hyb, E_x)
    t0 = logger.timer (ot, 'Vnn, Te, Vne, E_j, E_x', *t0)
    E_ot = get_E_ot(mc, rdm1, ot, casdm1, casdm2, casdm1s, RDM1s_ao, mol_full, OneRDM_fao, ao2dmet, ao2dmet_active, mo_coeff_dmet, my_mf)
    t0 = logger.timer (ot, 'E_ot', *t0)
    E_tot = Vnn + Te_Vne + E_j + (hyb_x * E_x) + (hyb_c * E_c) + E_ot 
    print ("**** This is the breakdown ****")
    print ("Vnn: ",Vnn)
    print ("Te_Vne: ",Te_Vne)
    print ("E_j: ",E_j)
    print ("E_x: ",E_x)
    print ("E_ot: ",E_ot)
    print ("E_tot: ", E_tot)
    
    logger.info (ot, 'MC-PDFT E = %s, Eot(%s) = %s', E_tot, ot.otxc, E_ot)

    return (E_tot, E_ot)

def get_E_ot(mc, rdm1, ot, casdm1, casdm2, casdm1s, RDM1s_ao, mol_full, OneRDM_fao, ao2dmet, ao2dmet_active, mo_coeff_dmet, my_mf, hermi =1, root=-1, mo_coeff=None, ci= None):
    
    from pyscf import lib
    print ('**** Starting on-top energy calculation ****')
    max_memory = 2000
    if ci is None: ci=mc.ci
    if ot is None: ot = mc.otfnal
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
#    print ("SV ncore, ncas, nelecas: ", ncore, ncas, nelecas)
    mo_cas = mo_coeff [:, ncore:(ncore+ncas)] # mo_cas = mo_coeff[:,:ncore:][:,:ncas]
#    print ("SV mo_coeff details: ", mo_coeff, mo_cas, ao2dmet_active)
    mo_cas_dmet = np.dot (ao2dmet_active, mo_cas)

    E_ot= 0.0
    ni, xctype = ot._numint, ot.xctype

    if xctype=='HF': return E_ot
    dens_deriv = ot.dens_deriv
    norbs_ao = ao2dmet.shape[1]
    OneCDMs = np.asarray(RDM1s_ao)
    cascm2 = _dms.dm2_cumulant (casdm2, casdm1s)
    #dm1s = _dms.casdm1s_to_dm1s (ot, casdm1s_ao, mo_coeff=mo_coeff, ncore=ncore, ncas=ncas)
    
    t0 = (logger.process_clock (), logger.perf_counter ())
    make_rho = tuple (ni._gen_rho_evaluator (ot.mol, OneCDMs[i,:,:], hermi) for i in range(2))
    for ao, mask, weight, coords in ni.block_loop (ot.mol, ot.grids, norbs_ao, dens_deriv, max_memory):
        rho = np.asarray ([m[0] (0, ao, mask, xctype) for m in make_rho])
        print ("SV rho: ", rho)
        t0 = logger.timer (ot, 'untransformed density', *t0)
        Pi = get_ontop_pair_density (ot, rho, ao, cascm2, mo_cas_dmet, dens_deriv, mask)
        t0 = logger.timer (ot, 'on-top pair density calculation', *t0)
        if rho.ndim == 2:
            rho = np.expand_dims (rho, 1)
            Pi = np.expand_dims (Pi, 0)
        E_ot += ot.eval_ot (rho, Pi, dderiv=0, weights=weight)[0].dot (weight)
        #E_ot += ot.get_E_ot (rho, Pi, weight)
        t0 = logger.timer (ot, 'on-top energy calculation', *t0) 
    return E_ot

    def get_energy_decomposition_mcpdft (mc, ot, mo_coeff=None, ci=None):
        ''' Compute a decomposition of the MC-PDFT energy into nuclear potential, core, Coulomb, exchange,
    and correlation terms. The exchange-correlation energy at the MC-SCF level is also returned.
    This is not meant to work with MC-PDFT methods that are already hybrids. Most return arguments
    are lists if mc is a state average instance. '''
    
        e_tot, e_ot, e_mcscf, e_cas, ci, mo_coeff = mc.kernel (mo=mo_coeff, ci=ci)[:6]
        e_nuc = mc._scf.energy_nuc ()
        h = mc.get_hcore ()
        xfnal, cfnal = ot.split_x_c ()

        e_core, e_coul, e_otx, e_otc, e_wfnxc = _get_e_decomp (mc, ot, mo_coeff, ci, e_mcscf, e_nuc, h, xfnal, cfnal, mc.nelecas)

        print ("e_nuc",e_nuc)
        print ("e_core",e_core)
        print ("e_coul",e_coul)
        print ("e_otx",e_otx)
        print ("e_otc",e_otc)
        print ("e_wfnxc",e_wfnxc)

        return e_nuc, e_core, e_coul, e_otx, e_otc, e_wfnxc


