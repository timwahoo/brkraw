# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 13:11:44 2023

@author: Timothy
"""
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from brkraw.lib.raw2frame import *
import brkraw as br
from brkraw.lib.utils import get_value

import streamlit as st

@st.cache_data
def load_scans(folder):
    
    datahandler = br.load(os.path.join(folder))
    available_scans = list()
    #print(available_scans)
    for scan_id, recos in datahandler._avail.items():
        #visu_pars = datahandler._get_visu_pars(scan_id, recos[0])
        acqp = datahandler.get_acqp(scan_id)
        protocol_name = get_value(acqp, 'ACQ_scan_name')
        #print(protocol_name)
        #print(scan_id, recos)
        available_scans.append(f'{scan_id} : {protocol_name}')
    
    return available_scans
    
@st.cache_data
def load_data(folder, ExpNum, mode, recoparts = 'all'):
    from brkraw.lib.parser import Parameter
    from brkraw.lib.pvobj import PvDatasetDir
    from brkraw.lib.utils import get_value, set_value
    
    # Load my stuff
    datahandler = br.load(os.path.join(folder))
    
    # Get MetaData
    acqp = datahandler.get_acqp(ExpNum)
    meth = datahandler.get_method(ExpNum)
    
    with open(os.path.join(folder, str(ExpNum),'pdata','1','reco'),'r') as f:
        reco = Parameter(f.read().split('\n'))
    
    
    fid_binary = datahandler.get_fid(ExpNum)
    # test functions
    #start_time = time.time()
    p_raw = readBrukerRaw(fid_binary, acqp, meth).astype(np.complex64)
    #print("--- %s seconds ---" % (time.time() - start_time))
  
    #start_time = time.time()
    p_frame = convertRawToFrame(p_raw, acqp, meth).astype(np.complex64)
    #print("--- %s seconds ---" % (time.time() - start_time))

    #start_time = time.time()
    p_kdata = convertFrameToCKData(p_frame, acqp, meth)
    #print("--- %s seconds ---" % (time.time() - start_time))
 
    
    stuff = ['quadrature', 'phase_rotate', 'zero_filling', 'FT', 'phase_corr_pi']
    
    #p_image_c = brkraw_Reco(p_kdata, reco, meth, recoparts=stuff)
    #start_time = time.time()
    if mode == 'mag':
        func = np.abs
    elif mode == 'real':
        func = np.real
    elif mode == 'imag':
        func = np.imag
    elif mode == 'phase':
        func = np.angle
    else:
        func = np.abs
    
    p_image = brkraw_Reco(p_kdata, reco, meth, recoparts = stuff)#recoparts)
    #p_image = brkraw_Reco(p_kdata, reco, meth, recoparts = recoparts)

    #print("--- %s seconds ---" % (time.time() - start_time))
    out = func(p_image)
    # Normailze output
    out_norm = (out-np.min(out))/(np.max(out)-np.min(out))
    
    return out_norm

# DATA SETUP
MainDir = "/home/jac/data/Wilson-Tim-PulseDev_Wilson-Tim-PulseDev_2023_06_30/Wilson_Tim_PulseDev_Wilson_Tim_PulseDev_651534_1_Default_2023_06_30_1177317_360.3.2/20230630_175638_Wilson_Tim_PulseDev_2023_06_30_1_10/"

protocols = load_scans(MainDir)


# TITLE
st.title('BRKRAW Streamlit')
#uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=False)



# SIDEBAR-------------------------------------
st.sidebar.title("Parameters")

scan_selected = st.sidebar.selectbox(
    'Scan Protocols',
    protocols)

view = st.sidebar.radio(
    "Views",
    ('1', '2', '3'))

mode = st.sidebar.radio(
    "Perspective",
    ('mag', 'real', 'imag', 'phase'))

images = load_data(MainDir, int(scan_selected.split(':')[0].strip()), mode=mode)


# Main Section-------------------------------
N1,N2,N3,N4,N5,N6,N7 = images.shape

if N1 > 1 and view == '3':
    n1 = st.sidebar.slider('N1', 0, N1-1, int(N1/2))
else:
    n1 = 0
    
if N2 > 1 and view == '2':
    n2 = st.sidebar.slider('N2', 0, N2-1, int(N2/2))
else:
    n2 = 0

if N3 > 1 and view == '1':
    n3 = st.sidebar.slider('N3', 0, N3-1, int(N3/2))
else:
    n3 = 0
    
if N4 > 1:
    n4 = st.sidebar.slider('N4', 0, N4-1, 0)
else:
    n4 = 0
    
if N5 > 1:
    n5 = st.sidebar.slider('N5', 0, N5-1, 0)
else:
    n5 = 0

if N6 > 1:
    n6 = st.sidebar.slider('N6', 0, N6-1, 0)
else:
    n6 = 0
    
if N7 > 1:
    n7 = st.sidebar.slider('N7', 0, N7-1, 0)
else:
    n7 = 0

st.header(f'[:,:,{n3},{n4},{n5},{n6},{n7}]')
if view == '1':
    st.image(np.squeeze(images[:,:,n3,n4,n5,n6,n7])/np.max(images), use_column_width=True)
    
elif view == '2':
    st.image(np.squeeze(images[:,n2,:,n4,n5,n6,n7])/np.max(images), use_column_width=True)

elif view == '3':
    st.image(np.squeeze(images[n1,:,:,n4,n5,n6,n7])/np.max(images), use_column_width=True)