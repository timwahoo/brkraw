"""
@author: Timothy Ho (UVA) 
"""
# -*- coding: utf-8 -*-
from operator import index
from ..lib.errors import *
from .. import BrukerLoader, __version__
from ..lib.utils import set_rescale, save_meta_files, mkdir
from ..lib.recon import recon
import argparse
import os, re
import sys

import numpy as np
import nibabel as nib

_supporting_bids_ver = '1.2.2'


def main():
    parser = argparse.ArgumentParser(prog='brkrecon',
                                     description="Brkrecon command-line interface")
    parser.add_argument("-v", "--version", action='version', version='%(prog)s v{}'.format(__version__))
    

    input_str = "input raw Bruker data"
    output_dir_str = "output directory name"
    output_fnm_str = "output filename"
    bids_opt = "create a JSON file contains metadata based on BIDS recommendation"

    subparsers = parser.add_subparsers(title='Sub-commands',
                                       description='To run this command, you must specify one of the functions listed'
                                                   'below next to the command. For more information on each function, '
                                                   'use -h next to the function name to call help document.',
                                       help='description',
                                       dest='function',
                                       metavar='command')
    
    nii = subparsers.add_parser("tonii", help='Convert all scans in a dataset to kspace, or complex image matrix')
    nii.add_argument("input", help=input_str, type=str, default=None)
    nii.add_argument("-b", "--bids", help=bids_opt, action='store_true')
    nii.add_argument("-o", "--output", help=output_fnm_str, type=str, default=False)
    nii.add_argument("-s", "--scanid", help="Scan ID, option to specify a particular scan to convert.", type=str)
    nii.add_argument("-r", "--recoid", help="RECO ID (default=1), "
                                            "option to specify a particular reconstruction id to convert",
                     type=int, default=1)
    nii.add_argument("-t", "--subjecttype", help="override subject type in case the original setting was not properly set." + \
                     "available options are (Biped, Quadruped, Phantom, Other, OtherAnimal)", type=str, default=None)
    nii.add_argument("-p", "--position", help="override position information in case the original setting was not properly input." + \
                     "the position variable can be defiend as <BodyPart>_<Side>, " + \
                     "available BodyParts are (Head, Foot, Tail) and sides are (Supine, Prone, Left, Right). (e.g. Head_Supine)", type=str, default=None)
    nii.add_argument("-f", "--formatting", help="FID processing methods " + \
                    "available processing are (kdata, image)", type=str, default='image')

    nii.add_argument("--ignore-slope", help='remove slope value from header', action='store_true')
    nii.add_argument("--ignore-offset", help='remove offset value from header', action='store_true')
    nii.add_argument("--ignore-rescale", help='remove slope and offset values from header', action='store_true')
    nii.add_argument("--ignore-localizer", help='ignore the scan if it is localizer', action='store_true', default=True)    
    nii.add_argument("--phase", help='only save phase data', action='store_true', default=False)

    args = parser.parse_args()

    if args.function == 'tonii':
        path     = args.input
        scan_id  = args.scanid
        reco_id  = args.recoid
        process  = args.formatting
        study    = BrukerLoader(path)
        ignore_localizer = args.ignore_localizer
        phase_only = args.phase
        
        if study.is_pvdataset:
            if args.output:
                output = args.output
            else:
                output = '{}_{}'.format(study._pvobj.subj_id,study._pvobj.study_id)
            if scan_id:
                acqpars  = study.get_acqp(int(scan_id))
                scanname = acqpars._parameters['ACQ_scan_name']
                scanname = scanname.replace(' ','-')
                scan_id = int(scan_id)
                reco_id = int(reco_id)
                
                if ignore_localizer and is_localizer(study, scan_id, reco_id):
                    print('Identified a localizer, the file will not be converted: ScanID:{}'.format(str(scan_id)))
                else:
                    try:
                        recon2nifti(study, scan_id, reco_id, output, scanname, process, only_phase=phase_only) 
                    except:
                        print('Conversion failed: ScanID:{}, RecoID:{}'.format(str(scan_id), str(reco_id)))
            else:
                for scan_id, recos in study._pvobj.avail_reco_id.items():
                    acqpars  = study.get_acqp(int(scan_id))
                    scanname = acqpars._parameters['ACQ_scan_name']
                    scanname = scanname.replace(' ','-')
                    if ignore_localizer and is_localizer(study, scan_id, recos[0]):
                        print('Identified a localizer, the file will not be converted: ScanID:{}'.format(str(scan_id)))
                    else:
                        try:
                            recon2nifti(study, scan_id, reco_id, output, scanname, process, only_phase=phase_only)   
                        except Exception as e:
                            print('Conversion failed: ScanID:{}'.format(str(scan_id)))
        else:
            print('{} is not PvDataset.'.format(path))

def recon2nifti(pvobj, scan_id, reco_id, output, scanname, process, only_phase=False):
    output_fname = '{}-{}-{}-{}'.format(output, str(scan_id), reco_id, scanname)
    visu_pars = pvobj._get_visu_pars(scan_id, 1)
    method = pvobj._method[scan_id]
    affine = pvobj._get_affine(visu_pars, method)
    fid_binary = pvobj.get_fid(scan_id)
    acqp = pvobj.get_acqp(scan_id)
    reco = pvobj._pvobj.get_reco(scan_id, 1)
    image = recon(fid_binary, acqp, method, reco, process = process)

    if len(image.shape) > 6:
        return
    
    #[x,y,z,_,n_channel,NI,NR] 
    image = image.transpose(0,1,2,4,3,5)
    # MultiSlice Acq Correction
    if '360' in acqp._parameters['ACQ_sw_version']:
        if acqp._parameters['ACQ_dim'] == 2 and acqp._parameters['NSLICES'] > 1:
            new_shape = list(image.shape)
            new_shape[2] = acqp._parameters['NSLICES']
            new_shape[3] = int(new_shape[3]/acqp._parameters['NSLICES'])
            image = image.reshape(new_shape)
            image = image.transpose(1,0,2,3,4,5)

    else:
        if acqp._parameters['ACQ_dim'] == 2 and acqp._parameters['NSLICES'] > 1:
            new_shape = list(image.shape)
            new_shape[2] = acqp._parameters['NSLICES']
            new_shape[3] = int(new_shape[3]/acqp._parameters['NSLICES'])
            image = image.reshape(new_shape)
        image = image.transpose(1,0,2,3,4,5)

    # [x, y, z, echo, channel, NR]
    niiobj = nib.Nifti1Image(np.squeeze(np.angle(image)), affine)
    niiobj = pvobj._set_nifti_header(niiobj, visu_pars, method, slope=False, offset=False)
    niiobj.to_filename(output_fname+'-p'+'.nii.gz')
    if only_phase:
        print('NifTi file is generated... [{}]'.format(output_fname))
        return
    niiobj = nib.Nifti1Image(np.squeeze(np.abs(image)), affine)
    niiobj = pvobj._set_nifti_header(niiobj, visu_pars, method, slope=False, offset=False)
    niiobj.to_filename(output_fname+'-m'+'.nii.gz')
    print('NifTi file is generated... [{}]'.format(output_fname))
    
def is_localizer(pvobj, scan_id, reco_id):
    visu_pars = pvobj.get_visu_pars(scan_id, reco_id)
    if 'VisuAcquisitionProtocol' in visu_pars.parameters:
        ac_proc = visu_pars.parameters['VisuAcquisitionProtocol']
        if re.search('tripilot', ac_proc, re.IGNORECASE) or re.search('localizer', ac_proc, re.IGNORECASE):
            return True
        else:
            return False
    else:
        return False
        
if __name__ == '__main__':
    main()