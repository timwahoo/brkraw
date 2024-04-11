"""
 DEVELOPED FOR BRUKER PARAVISION 7 datasets
 Below code will work for cartesian sequence
 GRE, MSME, RARE that were acquired with linear
 phase encoding

@author: Tim Ho (UVA) 
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt 

import brkraw as br
from brkraw.lib.parser import Parameter
from brkraw.lib.pvobj  import PvDatasetDir
from brkraw.lib.utils  import get_value, mkdir
import brkraw as br
from brkraw.lib.recon import *

import nibabel as nib
import sigpy as sp

PV_zipfile = '/home/jac/data/external_data/20231128_132106_hluna_piloFe_irm4_rata26_hluna_piloFe_irm4__1_1'
data_loader = br.load(PV_zipfile)

ExpNum = 6 
for ExpNum in data_loader._avail.keys():
    # Raw data processing for single job
    fid_binary = data_loader.get_fid(ExpNum)
    acqp = data_loader.get_acqp(ExpNum)
    meth = data_loader.get_method(ExpNum)
    reco = data_loader._pvobj.get_reco(ExpNum, 1)
    print(get_value(acqp, 'ACQ_sw_version'))
    print(get_value(acqp, 'ACQ_protocol_name' ), ExpNum)
    print(get_value(acqp, 'ACQ_size' ))

    # -----------------------------------------------------------------
    data = recon(fid_binary, acqp, meth, reco, recoparams='default')
    print(data.shape)
    
    if len(data.shape) == 6:
        output = '{}_{}'.format(data_loader._pvobj.subj_id,data_loader._pvobj.study_id)
        mkdir(output)
        output_fname =f"{acqp._parameters['ACQ_scan_name'].strip().replace(' ','_')}"
        for c in range(data.shape[3]):
            ni_img  = nib.Nifti1Image(np.angle(np.squeeze(data[:,:,:,c,:,:])), affine=np.eye(4))
            nib.save(ni_img, os.path.join(output,f"{acqp._parameters['ACQ_scan_name'].strip().replace(' ','_')}_C{c}.nii.gz"))
        print('NifTi file is generated... [{}]'.format(output_fname))
           