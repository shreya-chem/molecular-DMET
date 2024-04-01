# molecular-DMET for DME-PDFT

Users must install QC-DMET (https://github.com/SebWouters/QC-DMET) before using this.

This module is for multireference solvers within DMET.
The original dmet.py has been modified to include SA-CASSCF,SS-CASSCF solvers, NEVPT2 solver in DMET. Mix-spin state average is also available.

**DME-PDFT** solver is a new method added here: takes 1-RDM, 2-RDM from DMET to compute MC-PDFT energy of the full system.

Extra files for these solvers are also included.
