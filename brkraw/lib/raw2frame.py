# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:06:38 2023

@author: 
"""

from .utils import get_value, set_value
import .recoFunctions
import numpy as np

def readBrukerRaw(fid_binary, acqp, meth):
    dt_code = np.dtype('float64') if get_value(acqp, 'ACQ_ScanPipeJobSettings')[0][1] == 'STORE_64bit_float' else np.dtype('int32') 
    fid = np.frombuffer(fid_binary, dt_code)
    
    NI = get_value(acqp, 'NI')
    NR = get_value(acqp, 'NR')
    
    ACQ_size = get_value(acqp, 'ACQ_jobs')[0][0]
    numDataHighDim = np.prod(ACQ_size)
    numSelectedRecievers = get_value(acqp, 'ACQ_ReceiverSelectPerChan').count('Yes')
    nRecs = numSelectedRecievers
    
    jobScanSize = get_value(acqp, 'ACQ_jobs')[0][0]
    dim1 = int(len(fid)/(jobScanSize*nRecs))
    
    X = fid[::2] + 1j*fid[1::2] # Assume data is complex
    
    # [num_lines, channel, scan_size]
    X = np.reshape(X, [dim1, nRecs, int(jobScanSize/2)])
    #X = np.transpose(X, (1, 2, 0)).astype(np.complex64)
    return X


def convertRawToFrame(data, acqp, meth):
    results = data.copy()
     
    # Turn Raw into frame
    NI = get_value(acqp, 'NI')
    NR = get_value(acqp, 'NR')
    
    if 'rare' in get_value(meth,'Method').lower():
        ACQ_phase_factor    = get_value(meth,'PVM_RareFactor')    
    else:
        ACQ_phase_factor    = get_value(acqp,'ACQ_phase_factor')
    
    ACQ_obj_order       = get_value(acqp, 'ACQ_obj_order')
    if isinstance(ACQ_obj_order, int):
        ACQ_obj_order = [ACQ_obj_order]
    
    ACQ_dim             = get_value(acqp, 'ACQ_dim')
    numSelectedRecievers= results.shape[-2]
    
    acqSizes = np.zeros(ACQ_dim)
    #disp(get_value(acqp, 'ACQ_jobs'))
    scanSize = get_value(acqp, 'ACQ_jobs')[0][0]
    
    isSpatialDim = [i == 'Spatial' for i in get_value(acqp, 'ACQ_dim_desc')]
    spatialDims = sum(isSpatialDim)
    
    if isSpatialDim[0]:
        if spatialDims == 3:
            acqSizes[1:] = get_value(meth, 'PVM_EncMatrix')[1:]
        elif spatialDims == 2:
            acqSizes[1]  = get_value(meth, 'PVM_EncMatrix')[1]
    
    numresultsHighDim=np.prod(acqSizes[1:])
    acqSizes[0] = scanSize   
    
    if np.iscomplexobj(results):
        scanSize = int(acqSizes[0]/2)
    else:  
        scanSize = acqSizes[0]
    
    
    # print("NR\tRem\t\tACQ_Phase\tNI\tReceiver\tScanSize\n"+
    #     f'{int(NR)}\t{int(numresultsHighDim/ACQ_phase_factor)}\t\t\t{int(ACQ_phase_factor)}\t{int(NI)}\t{int(numSelectedRecievers)}\t\t\t{int(scanSize)}'
    #     )
    
    # print(ACQ_obj_order)
    
    # Resort
    if ACQ_dim>1:
        # [..., num_lines, channel, scan_size]
        # Order is a guess, will need to adjust where int(ACQ_phase_factor) and NR go 
        results = results.transpose((1,2,0))
        
        results1 = results.reshape(
            int(numSelectedRecievers), 
            int(scanSize), 
            int(ACQ_phase_factor), 
            int(NI), 
            int(numresultsHighDim/ACQ_phase_factor), 
            int(NR), order='F').copy()
        
        
        results2 = np.transpose(results1, (1, 2, 4, 0, 3, 5)) # => scansize, ACQ_phase_factor, numDataHighDim/ACQ_phase_factor, numSelectedReceivers, NI, NR
    
        results3 =  results2.reshape( 
            int(scanSize), 
            int(numresultsHighDim), 
            int(numSelectedRecievers), 
            int(NI), 
            int(NR), order='F') 
        
        frame = np.zeros_like(results3)
        frame[:,:,:,ACQ_obj_order,:] = results3 
        
        # Commented out (IDK if theres any use)
        #if NI != len(ACQ_obj_order):
        #    print('Size of ACQ_obj_order is not equal to NI: Data from unsupported method?')
        
        # We have a bug, Fortran vs C indexing
        #if not np.log2(results.shape[0]).is_integer() or not np.log2(results.shape[1]).is_integer():
        #    print('Current bug with any matrix that is not 2^n ')
        
        
    else:
        # Havent encountered this situation yet
        raise 'Bug here 120'
        results = np.reshape(results,(numSelectedRecievers, scanSize,1,NI,NR), order='F')
        return_out = np.zeros_like(results)
        return_out = np.transpose(results, (1, 2, 0, 3, 4))
        frame = None
    
    return frame


def convertFrameToCKData(frame, acqp, meth):
    NI = get_value(acqp, 'NI')
    NR = get_value(acqp, 'NR')
    
    ACQ_phase_factor    = get_value(acqp,'ACQ_phase_factor')
    ACQ_obj_order       = get_value(acqp, 'ACQ_obj_order')
    ACQ_dim             = get_value(acqp, 'ACQ_dim')
    numSelectedReceivers= frame.shape[2]
    
    acqSizes = np.zeros(ACQ_dim)
    
    scanSize = get_value(acqp, 'ACQ_jobs')[0][0]
    acqSizes[0] = scanSize
    ACQ_size = acqSizes
    
    isSpatialDim = [i == 'Spatial' for i in get_value(acqp, 'ACQ_dim_desc')]
    spatialDims = sum(isSpatialDim)
    
    if isSpatialDim[0]:
        if spatialDims == 3:
            acqSizes[1:] = get_value(meth, 'PVM_EncMatrix')[1:]
        elif spatialDims == 2:
            acqSizes[1]  = get_value(meth, 'PVM_EncMatrix')[1]
    numDataHighDim=np.prod(acqSizes[1:])
    
    if np.iscomplexobj(frame):
        scanSize = int(acqSizes[0]/2)
    else:
        scanSize = acqSizes[0]
    
    if ACQ_dim==3:
       
        PVM_EncSteps2=get_value(meth, 'PVM_EncSteps2')
        assert PVM_EncSteps2 != None
           
    PVM_Matrix = get_value(meth, 'PVM_Matrix')
    PVM_EncSteps1 = get_value(meth,'PVM_EncSteps1')
    
    PVM_AntiAlias = get_value(meth, 'PVM_AntiAlias')
    if PVM_AntiAlias == None:
        # No anti-aliasing available.
        PVM_AntiAlias = np.ones((ACQ_dim))
     
    PVM_EncZf=get_value(meth, 'PVM_EncZf')
    
    if PVM_EncZf == None:
        # No zero-filling/interpolation available.
        PVM_EncZf = np.ones((ACQ_dim))
    
    # Resort
    # use also method-parameters (encoding group)
    
    frameData = frame.copy()
    
    # MGE with alternating k-space readout: Reverse every second
    # scan. 
    
    if get_value(meth, 'EchoAcqMode') != None and get_value(meth,'EchoAcqMode') == 'allEchoes':
        print('Okay, Ill figure this out later')
        #frameData(:,:,:,2:2:end,:)=flipdim(data(:,:,:,2:2:end,:),1)
    
    
    # Calculate size of Cartesian k-space
    # Step 1: Anti-Aliasing
    ckSize = np.round(np.array(PVM_AntiAlias)*np.array(PVM_Matrix))
    # Step 2: Interpolation 

    reduceZf = 2*np.floor( (ckSize - ckSize/np.array(PVM_EncZf))/2 )
    ckSize = ckSize - reduceZf
    
    # # index of central k-space point (+1 for 1-based indexing in MATLAB) 
    ckCenterIndex = np.floor(ckSize/2 + 0.25) + 1

    readStartIndex = int(ckSize[0]-scanSize + 1)
    

    # # Reshape & store
    # switch ACQ_dim
    if ACQ_dim == 1:
        
        frameData = np.reshape(frameData,(scanSize, 1, 1, 1, numSelectedReceivers, NI, NR) , order='F')
        data = np.zeros((ckSize[0], 1, 1, 1, numSelectedReceivers, NI, NR), dtype=complex)
        data[readStartIndex-1:,0,0,0,:,:,:] = frameData
    elif ACQ_dim == 2: 
        frameData=np.reshape(frameData,(scanSize, int(ACQ_size[1]), 1, 1, numSelectedReceivers, NI, NR) , order='F')
        data=np.zeros([int(ckSize[0]), int(ckSize[1]), 1, 1, numSelectedReceivers, NI, NR], dtype=complex)
        encSteps1indices = (PVM_EncSteps1 + ckCenterIndex[1] - 1).astype(int)
        data[readStartIndex-1:,encSteps1indices,0,0,:,:,:] = frameData.reshape(data[readStartIndex-1:,encSteps1indices,0,0,:,:,:].shape, order='F')               
    elif ACQ_dim == 3:
    
        #if NR == 1:
        #    frameData = np.reshape(frameData,(scanSize, int(ACQ_size[1]), int(ACQ_size[2]), 1, numSelectedReceivers, NI) , order='F')
        #else:
        frameData = np.reshape(frameData,(scanSize, int(ACQ_size[1]), int(ACQ_size[2]), 1, numSelectedReceivers, NI, NR) , order='F')
        
        data=np.zeros([int(ckSize[0]), int(ckSize[1]), int(ckSize[2]), 1, numSelectedReceivers, NI, NR], dtype=complex)
        encSteps1indices = (PVM_EncSteps1 + ckCenterIndex[1] - 1).astype(int)
        encSteps2indices = (PVM_EncSteps2 + ckCenterIndex[2] - 1).astype(int)
        
        data[readStartIndex-1:,list(encSteps1indices),:,:,:,:,:] = frameData[:,:,list(encSteps2indices),:,:,:,:]
    else:
        raise 'Unknown ACQ_dim with useMethod'
    
    return data


def brkraw_Reco(kdata, reco, meth, recoparts = 'all'):
    reco_result = kdata.copy()
        
    if recoparts == 'all':
        recoparts = ['quadrature', 'phase_rotate', 'zero_filling', 'FT', 'phase_corr_pi', 
                     'cutoff',  'scale_phase_channels', 'sumOfSquares', 'transposition']
    
    # Other stuff
    RECO_ft_mode = get_value(reco, 'RECO_ft_mode')    
    if '360' in meth.headers['title'.upper()]:
        reco_ft_mode_new = []
        
        for i in RECO_ft_mode:
            if i == 'COMPLEX_FT' or i == 'COMPLEX_FFT':
                reco_ft_mode_new.append('COMPLEX_IFT')
            else:
                reco_ft_mode_new.append('COMPLEX_FT')
                
        reco = set_value(reco, 'RECO_ft_mode', reco_ft_mode_new)
        RECO_ft_mode = get_value(reco, 'RECO_ft_mode')
        
    # Adapt FT convention to acquisition version.
    N1, N2, N3, N4, N5, N6, N7 = kdata.shape

    dims = kdata.shape[0:4];
    for i in range(4):
        if dims[i]>1:
            dimnumber = (i+1);
        
    NINR=kdata.shape[5]*kdata.shape[6]
    signal_position=np.ones(shape=(dimnumber,1))*0.5;
    
    # all transposition the same?
    same_transposition = True
    RECO_transposition = get_value(reco, 'RECO_transposition')

    if not isinstance(RECO_transposition, list):
        RECO_transposition = [RECO_transposition]
    for i in RECO_transposition:
        if i != RECO_transposition[0]:
            same_transposition = False
    
    map_index= np.reshape( np.arange(0,kdata.shape[5]*kdata.shape[6]), (kdata.shape[6], kdata.shape[5]) ).flatten()
    
    # - START RECONSTRUCTION --------------------------------------------------
    for recopart in recoparts:
        if 'quadrature' in recopart:
            for NR in range(N7):
                for NI in range(N6):
                    for channel in range(N5):
                        reco_result[:,:,:,:,channel,NI,NR] = recoFunctions.reco_qopts(kdata[:,:,:,:,channel,NI,NR], reco, map_index[(NI+1)*(NR+1)-1])
                        
         
        if 'phase_rotate' in recopart:
            for NR in range(N7):
                for NI in range(N6):
                    for channel in range(N5):
                        #print(map_index[(NR+1)*(NI+1)-1])
                        reco_result[:,:,:,:,channel,NI,NR] = recoFunctions.reco_phase_rotate(kdata[:,:,:,:,channel,NI,NR], reco, map_index[(NI+1)*(NR+1)-1])
                       
        """ Need to look into if this case ever occurs"""
        if 'zero_filling' in recopart:
            RECO_ft_size = get_value(reco,'RECO_ft_size')
            
            newdata_dims=[1, 1, 1, 1]
            reco_size = get_value(reco, 'RECO_size')
            newdata_dims[0:len(reco_size)] = reco_size
            newdata = np.zeros(shape=newdata_dims+[N5, N6, N7], dtype=np.complex128)
            
            for NR in range(N7):
                for NI in range(N6):
                    for chan in range(N5):
                        newdata[:,:,:,:,chan,NI,NR] = recoFunctions.reco_zero_filling(reco_result[:,:,:,:,chan,NI,NR], reco, map_index[(NI+1)*(NR+1)-1], signal_position).reshape(newdata[:,:,:,:,chan,NI,NR].shape)
        
            reco_result=newdata        
        
        if 'FT' in recopart:
            for NR in range(N7):
                for NI in range(N6):
                    for chan in range(N5):
                        reco_result[:,:,:,:,chan,NI,NR] = recoFunctions.reco_FT(reco_result[:,:,:,:,chan,NI,NR], reco, map_index[(NI+1)*(NR+1)-1])
        
        if 'image_rotate' in recopart:
            for NR in range(N7):
                for NI in range(N6):
                    for chan in range(N5):
                        reco_result[:,:,:,:,chan,NI,NR] = recoFunctions.reco_image_rotate(reco_result[:,:,:,:,chan,NI,NR], reco, map_index[(NI+1)*(NR+1)-1])
        
        
        if 'phase_corr_pi' in recopart:
            for NR in range(N7):
                for NI in range(N6):
                    for chan in range(N5):
                        reco_result[:,:,:,:,chan,NI,NR] = recoFunctions.reco_phase_corr_pi(reco_result[:,:,:,:,chan,NI,NR], reco, map_index[(NI+1)*(NR+1)-1])
        
        if 'cutoff' in recopart: 
            newdata_dims=[1, 1, 1, 1]
            reco_size = get_value(reco, 'RECO_size')
            newdata_dims[0:len(reco_size)] = reco_size
            #print(newdata_dims)
            newdata = np.zeros(shape=newdata_dims+[N5, N6, N7], dtype=np.complex128)
            
            for NR in range(N7):
                for NI in range(N6):
                    for chan in range(N5):
                        newdata[:,:,:,:,chan,NI,NR] = recoFunctions.reco_cutoff(reco_result[:,:,:,:,chan,NI,NR], reco, map_index[(NI+1)*(NR+1)-1])
        
            reco_result=newdata
        
        
        if 'scale_phase_channels' in recopart: 
            for NR in range(N7):
                for NI in range(N6):
                    for chan in range(N5):
                        reco_result[:,:,:,:,chan,NI,NR] = recoFunctions.reco_scale_phase_channels(reco_result[:,:,:,:,chan,NI,NR], reco, chan)
         
        if 'sumOfSquares' in recopart:
            for NR in range(N7):
                for NI in range(N6):
                    reco_result[:,:,:,:,:1,NI,NR] = recoFunctions.reco_sumofsquares(reco_result[:,:,:,:,:,NI,NR], reco)
            
            reco_result = reco_result[:,:,:,:,:1,:,:]
            reco_result = np.real(reco_result)
            
            N5 = 1 # Update N5
        
    
        if 'transposition' in recopart:
            if same_transposition:
                # import variables:
                RECO_transposition = RECO_transposition[0]
                # calculate additional variables:
            
                # start process
                if RECO_transposition > 0:
                    ch_dim1 = (RECO_transposition % len(kdata.shape)) 
                    ch_dim2 = RECO_transposition - 1 
                    new_order = [0, 1, 2, 3]
                    new_order[int(ch_dim1)] = int(ch_dim2)
                    new_order[int(ch_dim2)] = int(ch_dim1)
                    #print(new_order)
                    reco_result = reco_result.transpose(new_order + [4, 5, 6])
            else:
                for NR in range(N7):
                    for NI in range(N6):
                        for chan in range(N5):
                            reco_result[:,:,:,:,chan,NI,NR] = recoFunctions.reco_transposition(reco_result[:,:,:,:,chan,NI,NR], reco, map_index[(NI+1)*(NR+1)-1])

    return reco_result