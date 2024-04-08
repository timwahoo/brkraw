# -*- coding: utf-8 -*-
"""
 DEVELOPED FOR BRUKER PARAVISION 360 datasets
 Functions below are made to support functions
 in recon.py
"""

from .utils import get_value
import numpy as np
    
def phase_rotate(frame, Reco, framenumber):
    # import variables
    RECO_rotate = get_value(Reco,'RECO_rotate')
    
    if RECO_rotate.shape[1] > framenumber:
        RECO_rotate =  get_value(Reco,'RECO_rotate')[:, framenumber]
    else:
        RECO_rotate =  get_value(Reco,'RECO_rotate')[:,0]
    
    if isinstance( get_value(Reco,'RECO_ft_mode'), list):
        RECO_ft_mode = get_value(Reco,'RECO_ft_mode')[0]
    else:
        RECO_ft_mode = get_value(Reco,'RECO_ft_mode')

    # calculate additional variables
    dims = [frame.shape[0], frame.shape[1], frame.shape[2]]

    # start process
    phase_matrix = np.ones_like(frame)
    for index in range(len(RECO_rotate)):
        f = np.arange(dims[index])

        if RECO_ft_mode in ['NO_FT']:
            phase_vector = np.ones_like(f)
        else:
            phase_vector = np.exp(1j*2*np.pi*(1-RECO_rotate[index])*f)

        if index == 0:
            phase_matrix *= np.tile(phase_vector[:,np.newaxis,np.newaxis], [1, dims[1], dims[2]])
        elif index == 1:
            phase_matrix *= np.tile(phase_vector[np.newaxis,:,np.newaxis], [dims[0], 1, dims[2]])
        elif index == 2:
            tmp = np.zeros((1,1,dims[2]), dtype=complex)
            tmp[0,0,:] = phase_vector
            phase_matrix *= np.tile(tmp, [dims[0], dims[1], 1])

    frame *= phase_matrix
    return frame


def zero_filling(frame, Reco, signal_position):

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
        dims = (frame.shape[0], frame.shape[1], frame.shape[2])

        # start process

        # Dimensions of frame and RECO_ft_size doesn't match? -> zero filling
        if not_Equal:
            newframe = np.zeros(RECO_ft_size, dtype=complex)
            startpos = np.zeros(len(RECO_ft_size), dtype=int)
            pos_ges = [None] * 3

            for i in range(len(RECO_ft_size)):
                diff = RECO_ft_size[i] - frame.shape[i] + 1
                startpos[i] = int(np.floor(diff * signal_position[i] + 1))
                if startpos[i] > RECO_ft_size[i]:
                    startpos[i] = RECO_ft_size[i]
                pos_ges[i] = slice(startpos[i] - 1, startpos[i] - 1 + dims[i])
                
            newframe[pos_ges[0], pos_ges[1], pos_ges[2]] = frame
        else:
            newframe = frame

        del startpos, pos_ges

    else:
        newframe = frame

    return newframe


def phase_corr(frame):
    # start process
    checkerboard = np.ones(shape=frame.shape[:3])
    # Use NumPy broadcasting to alternate the signs
    checkerboard[::2,::2,::2] = -1
    checkerboard[1::2,1::2,::2] = -1
    checkerboard[::2,1::2,1::2] = -1
    checkerboard[1::2,::2,1::2] = -1
    checkerboard * -1
    return checkerboard