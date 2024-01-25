# DEVELOPED FOR BRUKER PARAVISION 360 datasets

from .utils import get_value
import numpy as np

def readBrukerRaw(fid_binary, acqp, meth):   
    # Obtain raw FID
    dt_code = np.dtype('float64') if get_value(acqp, 'ACQ_ScanPipeJobSettings')[0][1] == 'STORE_64bit_float' else np.dtype('int32') 
    fid = np.frombuffer(fid_binary, dt_code)
    
    NI = get_value(acqp, 'NI')
    NR = get_value(acqp, 'NR')
    
    # Obtain Data Dimesions
    ACQ_size = get_value(acqp, 'ACQ_jobs')[0][0]
    numDataHighDim = np.prod(ACQ_size)
    numSelectedRecievers = get_value(acqp, 'ACQ_ReceiverSelectPerChan').count('Yes')
    nRecs = numSelectedRecievers
    
    jobScanSize = get_value(acqp, 'ACQ_jobs')[0][0]
    dim1 = int(len(fid)/(jobScanSize*nRecs))
    
    # Assume data is complex
    X = fid[::2] + 1j*fid[1::2] 
    
    # [num_readouts, channel, scan_size]
    X = np.reshape(X, [dim1, nRecs, int(jobScanSize/2)])
    
    return X
