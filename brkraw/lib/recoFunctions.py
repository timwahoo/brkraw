# -*- coding: utf-8 -*-
"""
 DEVELOPED FOR BRUKER PARAVISION 360 datasets
 Functions below are made to support functions
 in recon.py
"""

from .utils import get_value
import numpy as np

def reco_qopts(frame, Reco):

    # import variables
    RECO_qopts = get_value(Reco, 'RECO_qopts') 

    # claculate additional parameters
    dims = [frame.shape[0], frame.shape[1], frame.shape[2], frame.shape[3]]

    # check if the qneg-Matrix is necessary:
    use_qneg = False
    if (RECO_qopts.count('QUAD_NEGATION') + RECO_qopts.count('CONJ_AND_QNEG')) >= 1:
        use_qneg = True
        qneg = np.ones(frame.shape)  # Matrix containing QUAD_NEGATION multiplication matrix

    # start process
    for i in range(len(RECO_qopts)):
        if RECO_qopts[i] == 'COMPLEX_CONJUGATE':
            frame = np.conj(frame)
        elif RECO_qopts[i] == 'QUAD_NEGATION':
            if i == 0:
                qneg = qneg * np.tile([[1, -1]], [np.ceil(dims[0]/2), dims[1], dims[2], dims[3]])
            elif i == 1:
                qneg = qneg * np.tile([[1], [-1]], [dims[0], np.ceil(dims[1]/2), dims[2], dims[3]])
            elif i == 2:
                tmp = np.zeros([1, 1, dims[2], 2])
                tmp[0, 0, :, :] = [[1, -1]]
                qneg = qneg * np.tile(tmp, [dims[0], dims[1], np.ceil(dims[2]/2), dims[3]])
            elif i == 3:
                tmp = np.zeros([1, 1, 1, dims[3], 2])
                tmp[0, 0, 0, :, :] = [[1, -1]]
                qneg = qneg * np.tile(tmp, [dims[0], dims[1], dims[2], np.ceil(dims[3]/2)])
        elif RECO_qopts[i] == 'CONJ_AND_QNEG':
            frame = np.conj(frame)
            if i == 0:
                qneg = qneg * np.tile([[1, -1]], [np.ceil(dims[0]/2), dims[1], dims[2], dims[3]])
            elif i == 1:
                qneg = qneg * np.tile([[1], [-1]], [dims[0], np.ceil(dims[1]/2), dims[2], dims[3]])
            elif i == 2:
                tmp = np.zeros([1, 1, dims[2], 2])
                tmp[0, 0, :, :] = [[1, -1]]
                qneg = qneg * np.tile(tmp, [dims[0], dims[1], np.ceil(dims[2]/2), dims[3]])
            elif i == 3:
                tmp = np.zeros([1, 1, 1, dims[3], 2])
                tmp[0, 0, 0, :, :] = [[1, -1]]
                qneg = qneg * np.tile(tmp, [dims[0], dims[1], dims[2], np.ceil(dims[3]/2)])
    
    if use_qneg:
        if qneg.shape != frame.shape:
            qneg = qneg[0:dims[0], 0:dims[1], 0:dims[2], 0:dims[3]]
        frame = frame * qneg
    
    return frame

    
def reco_phase_rotate(frame, Reco, actual_framenumber):
    # import variables
    RECO_rotate = get_value(Reco,'RECO_rotate')
    
    if RECO_rotate.shape[1] > actual_framenumber:
        RECO_rotate =  get_value(Reco,'RECO_rotate')[:, actual_framenumber]
    else:
        RECO_rotate =  get_value(Reco,'RECO_rotate')[:,0]
    
    if isinstance( get_value(Reco,'RECO_ft_mode'), list):
        RECO_ft_mode = get_value(Reco,'RECO_ft_mode')[0]
    else:
        RECO_ft_mode = get_value(Reco,'RECO_ft_mode')

    # calculate additional variables
    dims = [frame.shape[0], frame.shape[1], frame.shape[2], frame.shape[3]]

    # start process
    phase_matrix = np.ones_like(frame)
    for index in range(len(RECO_rotate)):
        f = np.arange(dims[index])

        if RECO_ft_mode in ['NO_FT']:
            phase_vector = np.ones_like(f)
        elif RECO_ft_mode in ['COMPLEX_FT']:
            phase_vector = np.exp(1j*2*np.pi*(1-RECO_rotate[index])*f)

        if index == 0:
            phase_matrix *= np.tile(phase_vector[:,np.newaxis,np.newaxis,np.newaxis], [1, dims[1], dims[2], dims[3]])
        elif index == 1:
            phase_matrix *= np.tile(phase_vector[np.newaxis,:,np.newaxis,np.newaxis], [dims[0], 1, dims[2], dims[3]])
        elif index == 2:
            tmp = np.zeros((1,1,dims[2],1), dtype=complex)
            tmp[0,0,:,0] = phase_vector
            phase_matrix *= np.tile(tmp, [dims[0], dims[1], 1, dims[3]])
        elif index == 3:
            tmp = np.zeros((1,1,1,dims[3]), dtype=complex)
            tmp[0,0,0,:] = phase_vector
            phase_matrix *= np.tile(tmp, [dims[0], dims[1], dims[2], 1])

    frame *= phase_matrix
    return frame


def reco_zero_filling(frame, Reco, signal_position):

    # Check if Reco.RECO_ft_size is not equal to size(frame)
    not_Equal = any([(i != j) for i,j in zip(frame.shape,get_value(Reco, 'RECO_ft_size'))])
        
    if not_Equal:
        if any(signal_position > 1) or any(signal_position < 0):
            raise ValueError('signal_position has to be a vector between 0 and 1')

        RECO_ft_size = get_value(Reco,'RECO_ft_size')

        # check if ft_size is correct:
        for i in range(len(RECO_ft_size)):
            if RECO_ft_size[i] < frame.shape[i]:
                raise ValueError('RECO_ft_size has to be bigger than the size of your data-matrix')

        # calculate additional variables
        dims = (frame.shape[0], frame.shape[1], frame.shape[2], frame.shape[3])

        # start process

        # Dimensions of frame and RECO_ft_size doesn't match? -> zero filling
        if not_Equal:
            newframe = np.zeros(RECO_ft_size, dtype=complex)
            startpos = np.zeros(len(RECO_ft_size), dtype=int)
            pos_ges = [None] * 4

            for i in range(len(RECO_ft_size)):
                diff = RECO_ft_size[i] - frame.shape[i] + 1
                startpos[i] = int(np.floor(diff * signal_position[i] + 1))
                if startpos[i] > RECO_ft_size[i]:
                    startpos[i] = RECO_ft_size[i]
                pos_ges[i] = slice(startpos[i] - 1, startpos[i] - 1 + dims[i])
                
            newframe[pos_ges[0], pos_ges[1], pos_ges[2], pos_ges[3]] = frame
        else:
            newframe = frame

        del startpos, pos_ges

    else:
        newframe = frame

    return newframe


def reco_phase_corr_pi(frame):
    # start process
    checkerboard = np.ones(shape=frame.shape[:4])
    # Use NumPy broadcasting to alternate the signs
    checkerboard[::2,::2,::2,0] = -1
    checkerboard[1::2,1::2,::2,0] = -1
    checkerboard[::2,1::2,1::2,0] = -1
    checkerboard[1::2,::2,1::2,0] = -1
    checkerboard * -1
    return checkerboard