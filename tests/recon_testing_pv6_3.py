"""
 DEVELOPED FOR BRUKER PARAVISION 6 datasets
 Below code will work for cartesian sequence
 GRE, MSME, RARE that were acquired with linear
 phase encoding
 EALEXWater dataset

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

PV_zipfile = '/home/jac/data/external_data/Test_Wat202402_15141272_1_Default_0206_Tomato_15141275_6.0.1.PvDatasets'
data_loader = br.load(PV_zipfile)

for ExpNum in list(data_loader._avail.keys()):
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
        #plt.figure()
        for c in range(data.shape[3]):
            #plt.subplot(1,4,c+1)
            #plt.imshow(np.abs(np.squeeze(data[:,:,7,:,c,0,:])))
            ni_img  = nib.Nifti1Image(np.abs(np.squeeze(data[:,:,:,c,:,:])), affine=np.eye(4))
            nib.save(ni_img, os.path.join(output,f"{acqp._parameters['ACQ_scan_name'].strip().replace(' ','_')}_C{c}.nii.gz"))
        #plt.show()
        print('NifTi file is generated... [{}]'.format(output_fname))
           