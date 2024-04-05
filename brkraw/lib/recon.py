# -*- coding: utf-8 -*-
"""
Created on Sat Jan  20 10:06:38 2024
 DEVELOPED FOR BRUKER PARAVISION 360 datasets
 Below code will work for cartesian sequence
 GRE, MSME, RARE that were acquired with linear
 phase encoding

"""

from .utils import get_value
from .recoFunctions import *
import numpy as np
from copy import deepcopy


def recon(fid_binary, acqp, meth, reco, process = 'image', recoparams = 'default'):
    """ Process FID -> Channel Sorted -> Frame-Sorted -> K-Sorted -> Image
    
    Parameters
    ----------
    fid_binary : bytestring

    acqp : dict (brkraw Parameter structure)

    meth : dict (brkraw Parameter structure)

    reco : dict (brkraw Parameter structure)

    process: 'raw','kdata','image'

    recoparams: 'default' or list()
        
        'default':
            recoparams = ['quadrature', 'phase_rotate', 'zero_filling']

    Returns
    -------
    output : np.array 

    """
    output = readBrukerRaw(fid_binary, acqp, meth)
    if process == 'raw':
        return output
    
    # IF CORRECT SEQUENCES
    if  'rare'      in get_value(acqp, 'ACQ_protocol_name').lower() or \
        'msme'      in get_value(acqp, 'ACQ_protocol_name').lower() or \
        'localizer' in get_value(acqp, 'ACQ_protocol_name').lower() or \
        'gre'       in get_value(acqp, 'ACQ_protocol_name').lower() or \
        'mge'       in get_value(acqp, 'ACQ_protocol_name').lower() or \
        'FLASH.ppg' == get_value(acqp, 'PULPROG'):

        # Compressed Sensing
        if get_value(meth,'PVM_EncCS') == 'Yes':
            try:
                import sigpy as sp 
                from . import recoSigpy
            except ImportError:
                raise ImportError('Sigpy Module Not Installed')
            
            print(get_value(acqp, 'ACQ_scan_name' ))
            print("Warning: Compressed Sensing is not fully supported ...")
            output = recoSigpy.compressed_sensing_recon(output, acqp, meth, reco)
            return output

        # Full Cartesian Pipeline
        output = convertRawToKdata(output, acqp, meth)
        if process == 'kdata':
            return output

        output = brkrawReco(output, reco, meth, recoparams = recoparams)
    
    else:
        print("Warning: SEQUENCE PROTOCOL {} NOT SUPPORTED...".format(get_value(acqp, 'ACQ_scan_name' )))
        print("returning 'raw' sorting")

    return output


def readBrukerRaw(fid_binary, acqp, meth):
    """ Sorts FID into a 3D np matrix [num_readouts, channel, scan_size]

    Parameters
    ----------
    fid_binary : bytestring

    acqp : dict (brkraw Parameter structure)

    meth : dict (brkraw Parameter structure)
        

    Returns
    -------
    X : np.array [num_lines, channel, scan_size]
    """
    # META DATA
    NI = get_value(acqp, 'NI')
    NR = get_value(acqp, 'NR')

    dt_code = 'int32' 
    if get_value(acqp, 'ACQ_ScanPipeJobSettings') != None:
        if get_value(acqp, 'ACQ_ScanPipeJobSettings')[0][1] == 'STORE_64bit_float':
            dt_code = 'float64' 
    
    if '32' in dt_code:
        bits = 32 # Need to add a condition here
    elif '64' in dt_code:
        bits = 64
    DT_CODE = np.dtype(dt_code)

    BYTORDA = get_value(acqp, 'BYTORDA') 
    if BYTORDA == 'little':
        DT_CODE = DT_CODE.newbyteorder('<')
    elif BYTORDA == 'big':
        DT_CODE = DT_CODE.newbyteorder('>')

    # Get FID FROM buffer
    fid = np.frombuffer(fid_binary, DT_CODE)
    
    # Sort raw data
    if '360' in get_value(acqp,'ACQ_sw_version'):
        # METAdata for 360
        ACQ_size = get_value(acqp, 'ACQ_jobs')[0][0]
        nRecs = get_value(acqp, 'ACQ_ReceiverSelectPerChan').count('Yes')
        scanSize = get_value(acqp, 'ACQ_jobs')[0][0]
        
        # Assume data is complex
        X = fid[::2] + 1j*fid[1::2] 
        
        # [num_lines, channel, scan_size]
        X = np.reshape(X, [-1, nRecs, int(scanSize/2)])

    else:
        # METAdata Versions Before 360        
        # PV7 only save 1 channel
        nRecs = get_value(acqp, 'ACQ_ReceiverSelect').count('Yes') if get_value(acqp, 'ACQ_ReceiverSelect') != None else 1
        ACQ_size = [get_value(acqp, 'ACQ_size')] if isinstance(get_value(acqp, 'ACQ_size'),int) else get_value(acqp, 'ACQ_size')

        if get_value(acqp, 'GO_block_size') == 'Standard_KBlock_Format':
            blocksize = int(np.ceil(ACQ_size[0]*nRecs*(bits/8)/1024)*1024/(bits/8))
        else:
            blocksize = int(ACQ_size[0]*nRecs)

        # CHECK SIZE
        if fid.size != blocksize*np.prod(ACQ_size[1:])*NI*NR:
            raise Exception('readBrukerRaw 158: Error Size dont match')

        # Convert to Complex
        fid = fid[::2] + 1j*fid[1::2] 
        fid = fid.reshape([-1,blocksize//2])
 
        # Reshape Matrix [num_lines, channel, scan_size]
        if blocksize != ACQ_size[0]*nRecs:
            fid = fid[:,:ACQ_size[0]//2*nRecs]
            fid = fid.reshape((-1,nRecs,ACQ_size[0]//2))
            X = fid.transpose(0,1,2)
            
        else:
            X = fid.reshape((-1, nRecs, ACQ_size[0]//2))
         
    return X

def convertRawToKdata(raw, acqp, meth):
    _, NC, Nreadout = raw.shape

    # Meta data
    ACQ_dim = get_value(acqp, 'ACQ_dim')
    NI = get_value(acqp, 'NI')
    NR = get_value(acqp, 'NR')
    ACQ_obj_order = [get_value(acqp, 'ACQ_obj_order')] if isinstance(get_value(acqp, 'ACQ_obj_order'), int) else get_value(acqp, 'ACQ_obj_order')
    ACQ_phase_factor = get_value(meth,'PVM_RareFactor') if 'rare' in get_value(meth,'Method').lower() else get_value(acqp,'ACQ_phase_factor')

    # DATA SIZES
    PVM_Matrix = get_value(meth, 'PVM_Matrix')
    kSize = np.round(np.array(get_value(meth, 'PVM_AntiAlias'))*np.array(PVM_Matrix))
    reduceZf = 2*np.floor( (kSize - kSize/np.array(get_value(meth, 'PVM_EncZf')[0]))/2 )
    kSize = kSize - reduceZf
    readStart = int(kSize[0]-Nreadout)
    # Phase Encoding
    PVM_EncMatrix = get_value(meth,'PVM_EncMatrix')
    NPE = np.prod(PVM_EncMatrix[1:])
    PVM_EncSteps2 = [0]
    if ACQ_dim == 3:
        PVM_EncSteps2 = get_value(meth, 'PVM_EncSteps2')
        PVM_EncSteps2 = (PVM_EncSteps2 - np.min(PVM_EncSteps2)).astype(int)
    PVM_EncSteps1 = get_value(meth,'PVM_EncSteps1')
    PVM_EncSteps1 = (PVM_EncSteps1 - np.min(PVM_EncSteps1)).astype(int)
     
    # Resorting
    raw = raw.reshape((NR,int(NPE/ACQ_phase_factor),NI,ACQ_phase_factor,NC,Nreadout)).transpose(0,2,4,1,3,5)
    raw = raw.reshape((NR,NI,NC,NPE,Nreadout)).transpose((4,3,2,1,0))
    raw = raw.reshape(Nreadout, int(PVM_EncMatrix[1]), int(PVM_EncMatrix[2]) if ACQ_dim == 3 else 1, NC, NI, NR, order = 'F')
 
    kdata = np.zeros([int(kSize[0]), int(kSize[1]),int(kSize[2]) if ACQ_dim == 3 else 1, NC, NI, NR], dtype=complex)
    kdata[readStart:,PVM_EncSteps1,:,:,:,:] = raw[:,:,PVM_EncSteps2,:,:,:]
    kdata = kdata[:,:,:,:,ACQ_obj_order,:]
    if get_value(meth, 'EchoAcqMode') != None and get_value(meth,'EchoAcqMode') == 'allEchoes':
        kdata[:,:,:,:,1::2,:] = raw[::-1,:,:,:,1::2,:]
    print(kdata.shape)
    return kdata

def brkrawReco(kdata, reco, meth, recoparams = 'default'):
    reco_result = kdata.copy()
        
    if recoparams == 'default':
        recoparams = ['phase_rotate', 'zero_filling']    
    # DIMS
    _, _, _, N4, N5, N6 = kdata.shape

    for i in range(4):
        if kdata.shape[0:4][i]>1:
            dimnumber = (i+1)
    
    signal_position=np.ones(shape=(dimnumber,1))*0.5
    
    # --- START RECONSTRUCTION ---
    if 'phase_rotate' in recoparams:
        map_index= np.reshape( np.arange(0,kdata.shape[4]*kdata.shape[5]), (kdata.shape[5], kdata.shape[4]) ).flatten()
        for NR in range(N6):
            for NI in range(N5):
                for channel in range(N4):
                    reco_result[:,:,:,channel,NI,NR] = phase_rotate(kdata[:,:,:,channel,NI,NR], reco, map_index[(NI+1)*(NR+1)-1])
        
    if 'zero_filling' in recoparams:
        newdata_dims=[1, 1, 1]
        RECO_ft_size = get_value(reco,'RECO_ft_size')
        newdata_dims[0:len(RECO_ft_size)] = RECO_ft_size
        newdata = np.zeros(shape=newdata_dims+[N4, N5, N6], dtype=np.complex128)

        for NR in range(N6):
            for NI in range(N5):
                for chan in range(N4):
                    newdata[:,:,:,chan,NI,NR] = zero_filling(reco_result[:,:,:,chan,NI,NR], reco, signal_position).reshape(newdata[:,:,:,chan,NI,NR].shape)
        reco_result=newdata    

    # Always FT and Phase correct
    reco_result = np.fft.ifftn(reco_result, axes=(0,1,2))
    reco_result *= np.tile(phase_corr(reco_result)[:,:,:,np.newaxis,np.newaxis,np.newaxis],
                                                  [1,1,1,N4,N5,N6])
    # --- End of RECONSTRUCTION --- 
                            
    return reco_result