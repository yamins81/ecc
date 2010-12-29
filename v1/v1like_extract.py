#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time
from os import path
import warnings
import pprint
import hashlib
import cPickle
import warnings

import numpy as np
import scipy as sp
from scipy import io

import v1.v1like_funcs as v1f 
import v1.v1like_math as v1m 
import v1.colorconv as colorconv

from v1.npclockit import clockit_onprofile

from dbutils import steptag,db_computation_handler

from starflow.protocols import protocolize, actualize
from starflow.utils import activate

warnings.simplefilter('ignore', UserWarning)

DEFAULT_OVERWRITE = False
DEFAULT_VERBOSE = False
WRITE_RETRY = 10

filt_l = None
conv = v1f.conv
verbose = DEFAULT_VERBOSE
    

def feature_extraction_protocol(intag,config_path,type = 'stepwise'):

    if type == 'stepwise':
        return feature_extraction_protocol_stepwise(intag,config_path)
    elif type == 'consolidated':
        return feature_extraction_protocol_consolidated(intag,config_path)
    else:
        raise ValueError, 'type not recognized.'
    
       
def feature_extraction_protocol_stepwise(intag,config_path):
    dbh = db_computation_handler
    D = [('image_to_array',dbh,(intag,image_to_array,None,config_path)),
     ('preprocessing',dbh,('array',preprocessing,config_path,config_path)),
     ('normin',dbh,[('preprocessing',normalize,config_path,config_path),{'args':('in',)}]),
     ('filter',dbh,('normin',filter,config_path,config_path)),
     ('activate',dbh,('filter',activate,config_path,config_path)),
     ('normout',dbh,[('activate',normalize,config_path,config_path),{'args':('out',)}]),
     ('sparsify',dbh, ('normout',sparsify,config_path,config_path)),
     ('pool',dbh, ('sparsify',pool,config_path,config_path)),
     ('add_features',dbh,(['normin','filter','activate','sparsify','pool','partial_preprocessing'],add_features,config_path,config_path))
     ]
    return D      

          
def feature_extraction_protocol_consolidated(intag,config_path):
    D = [('extract',db_computation_handler,(intag,full_extract,None,config_path))]
    
    return D          
    
                                                   
def get_params(config_fname):
    config_path = path.abspath(config_fname)
    if verbose: print "Config file:", config_path
    v1like_config = {}
    execfile(config_path, {}, v1like_config)
    params, featsel = v1like_config['model'][0]
    if 'conv_mode' not in params:
        params['conv_mode'] = 'same'
    if 'color_space' not in params:
        params['color_space'] = 'gray'    
    
    return params,featsel
    
    
@steptag('full_extraction')
def full_extract(fhs,config_fname):
    fh = fhs[0][0][1]
    params, featsel = get_params(config_fname)
    
    arr = image2array(params,fh)

    orig_imga,orig_imga_conv = image_preprocessing(arr,params)
    fvector_l = []
    filt_l = get_gabor_filters(params['filter'])      
    for cidx in xrange(orig_imga_conv.shape[2]):
        imga0 = map_preprocessing(orig_imga_conv[:,:,cidx],params)
        
        imga1 = v1f.v1like_norm(imga0[:,:,sp.newaxis], conv_mode, **params['normin'])
        
        imga2 = v1f.v1like_filter(imga1[:,:,0], conv_mode, filt_l)

        minout = params['activ']['minout'] # sustain activity
        maxout = params['activ']['maxout'] # saturation
        imga3 = imga2.clip(minout, maxout)

        imga4 = v1f.v1like_norm(imga3, conv_mode, **params['normout'])

        if "sparsify" in params and params["sparsify"]:
            imga4 = (imga4.max(2)[:,:,None] == imga4)

        imga5 = v1f.v1like_pool(imga4, conv_mode, **params['pool'])
        output = imga5

        fvector = include_map_level_features(imga1,imga2,imga3,imga4,imga5,output,featsel)

        fvector_l += [fvector]
        
    # -- include image-level bells & whistles
    fvector_l = include_image_level_features(orig_imga,fvector_l, featsel)

    # -- reshape  
    fvector_l = [fvector.ravel() for fvector in fvector_l]
    out = sp.concatenate(fvector_l).ravel()
    
    return cPickle.dumps(out)
    
    
@steptag(('preprocessing','partial_preprocessing'))        
def preprocessing(fhs,config_fname):

    fh = fhs[0][0][1]
    
    params, featsel = get_params(config_fname)
    arr = cPickle.loads(fh.read())
    
    orig_imga,orig_imga_conv = image_preprocessing(arr,params) 
    output = {}
    for cidx in xrange(orig_imga_conv.shape[2]):
        output[cidx] = map_preprocessing(orig_imga_conv[:,:,cidx],params)
        

    return (cPickle.dumps(output),cPickle.dumps(orig_imga))

def norm_add(x):  return 'norm' + x
   
@steptag(norm_add)
def normalize(fhs,config_fname,pset):

    fh = fhs[0][0][1]
    assert pset in ['in','out'], 'pset not recognized'
    params, featsel = get_params(config_fname)
    conv_mode = params['conv_mode']
    input = cPickle.loads(fh.read())
    
    output = {}
    for cidx in input.keys():
       if len(input[cidx].shape) == 3:
          inobj = input[cidx]
       else:
          inobj = input[cidx][:,:,sp.newaxis]
       output[cidx] = v1f.v1like_norm(inobj, conv_mode, **params['norm' + pset])

    return cPickle.dumps(output)

@steptag('filter')        
def filter(fhs,config_fname):
    fh = fhs[0][0][1]
    params, featsel = get_params(config_fname)
    conv_mode = params['conv_mode']
    input = cPickle.loads(fh.read())
    
    filt_l = get_gabor_filters(params['filter'])
    
    output = {}
    for cidx in input.keys():
       output[cidx] = v1f.v1like_filter(input[cidx][:,:,0], conv_mode, filt_l)
    
    return cPickle.dumps(output)    
   
@steptag('activate')   
def activate(fhs,config_fname):
    fh = fhs[0][0][1]
    params, featsel = get_params(config_fname)
    minout = params['activ']['minout'] # sustain activity
    maxout = params['activ']['maxout'] # saturation
    input = cPickle.loads(fh.read())
  
    
    output = {}
    for cidx in input.keys():
       output[cidx] = input[cidx].clip(minout, maxout)
       
    return cPickle.dumps(output)

@steptag('sparsify')                        
def sparsify(fhs,config_fname):
    
    params, featsel = get_params(config_fname)
    fh = fhs[0][0][1]
    
    if "sparsify" in params and params["sparsify"]:
        input = cPickle.loads(fh.read())
        output = {}
        for cidx in input.keys():
            output[cidx] = (input[cidx].max(2)[:,:,None] == imga4)
        return cPickle.dumps(output)
    else:
        return fh.read()
    

@steptag('pool')    
def pool(fhs,config_fname):
    fh = fhs[0][0][1]
    params, featsel = get_params(config_fname)
    conv_mode = params['conv_mode']
    input = cPickle.loads(fh.read())
    
    output = {}
    for cidx in input.keys():
        output[cidx] = v1f.v1like_pool(input[cidx],conv_mode,**params['pool'])
        
    return cPickle.dumps(output)

@steptag('add_features')    
def add_features(fhs,config_fname):
    params, featsel = get_params(config_fname)
    
    fh = dict(fhs[0]) 
    
    if featsel['output']:
        imga1 = cPickle.loads(fh['normin'].read())
        imga2 = cPickle.loads(fh['filter'].read())
        imga3 = cPickle.loads(fh['activate'].read())
        imga4 = cPickle.loads(fh['sparsify'].read())
        imga5 = cPickle.loads(fh['pool'].read())
        orig_imga = cPickle.load(fh['partial_preprocessing'])
            
        keys = imga5.keys()
        fvector_l = []
        for cidx in keys:
            fvector = include_map_level_features(imga1[cidx],
                                                 imga2[cidx],
                                                 imga3[cidx],
                                                 imga4[cidx],
                                                 imga5[cidx],
                                                 imga5[cidx],
                                                 featsel)
            fvector_l += [fvector]
    
        fvector_l = include_image_level_features(orig_imga,fvector_l,featsel)
        
        fvector_l = [fvector.ravel() for fvector in fvector_l]
        
    else:
        imga5 = cPickle.loads(fh['pool'].read())
        fvector_l = [fvector[key].ravel() for key in imga5]
    
    out = sp.concatenate(fvector_l).ravel()
    
    return cPickle.dumps(out)

    
@steptag('array')
def image_to_array(fhs,config_fname):
    params, featsel = get_params(config_fname)
 
    fh = fhs[0][0][1]
    
    imgarr = image2array(params,fh)
    
    return cPickle.dumps(imgarr)
    

def image2array(rep,fobj):
    resize_type = rep['preproc'].get('resize_type', 'input')
    if resize_type == 'output':
        if 'max_edge' not in rep['preproc']:
            raise NotImplementedError
        # add whatever is needed to get output = max_edge
        new_max_edge = rep['preproc']['max_edge']

        preproc_lsum = rep['preproc']['lsum_ksize']
        new_max_edge += preproc_lsum-1
            
        normin_kshape = rep['normin']['kshape']
        assert normin_kshape[0] == normin_kshape[1]
        new_max_edge += normin_kshape[0]-1

        filter_kshape = rep['filter']['kshape']
        assert filter_kshape[0] == filter_kshape[1]
        new_max_edge += filter_kshape[0]-1
        
        normout_kshape = rep['normout']['kshape']
        assert normout_kshape[0] == normout_kshape[1]
        new_max_edge += normout_kshape[0]-1
        
        pool_lsum = rep['pool']['lsum_ksize']
        new_max_edge += pool_lsum-1

        rep['preproc']['max_edge'] = new_max_edge
    
    if 'max_edge' in rep['preproc']:
        max_edge = rep['preproc']['max_edge']
        resize_method = rep['preproc']['resize_method']
        imgarr = v1f.get_image(fobj, max_edge=max_edge,
                           resize_method=resize_method)
    else:
        resize = rep['preproc']['resize']
        resize_method = rep['preproc']['resize_method']        
        imgarr = v1f.get_image2(fobj, resize=resize,
                            resize_method=resize_method)
                            
    return imgarr


def image_preprocessing(arr,params):

    arr = sp.atleast_3d(arr)

    smallest_edge = min(arr.shape[:2])

    rep = params
    
    preproc_lsum = rep['preproc']['lsum_ksize']
    if preproc_lsum is None:
        preproc_lsum = 1
    smallest_edge -= (preproc_lsum-1)
            
    normin_kshape = rep['normin']['kshape']
    smallest_edge -= (normin_kshape[0]-1)

    filter_kshape = rep['filter']['kshape']
    smallest_edge -= (filter_kshape[0]-1)
        
    normout_kshape = rep['normout']['kshape']
    smallest_edge -= (normout_kshape[0]-1)
        
    pool_lsum = rep['pool']['lsum_ksize']
    smallest_edge -= (pool_lsum-1)

    arrh, arrw, _ = arr.shape

    if smallest_edge <= 0 and rep['conv_mode'] == 'valid':
        if arrh > arrw:
            new_w = arrw - smallest_edge + 1
            new_h =  int(np.round(1.*new_w  * arrh/arrw))
            print new_w, new_h
            raise
        elif arrh < arrw:
            new_h = arrh - smallest_edge + 1
            new_w =  int(np.round(1.*new_h  * arrw/arrh))
            print new_w, new_h
            raise
        else:
            pass
    
    # TODO: finish image size adjustment
    assert min(arr.shape[:2]) > 0

    # use the first 3 channels only
    orig_imga = arr.astype("float32")[:,:,:3]

    # make sure that we don't have a 3-channel (pseudo) gray image
    if orig_imga.shape[2] == 3 \
            and (orig_imga[:,:,0]-orig_imga.mean(2) < 0.1*orig_imga.max()).all() \
            and (orig_imga[:,:,1]-orig_imga.mean(2) < 0.1*orig_imga.max()).all() \
            and (orig_imga[:,:,2]-orig_imga.mean(2) < 0.1*orig_imga.max()).all():
        orig_imga = sp.atleast_3d(orig_imga[:,:,0])

    # rescale to [0,1]
    #print orig_imga.min(), orig_imga.max()
    if orig_imga.min() == orig_imga.max():
        raise MinMaxError("[ERROR] orig_imga.min() == orig_imga.max() "
                          "orig_imga.min() = %f, orig_imga.max() = %f"
                          % (orig_imga.min(), orig_imga.max())
                          )
    
    orig_imga -= orig_imga.min()
    orig_imga /= orig_imga.max()

    # -- color conversion
    # insure 3 dims
    #print orig_imga.shape
    if orig_imga.ndim == 2 or orig_imga.shape[2] == 1:
        orig_imga_new = sp.empty(orig_imga.shape[:2] + (3,), dtype="float32")
        orig_imga.shape = orig_imga_new[:,:,0].shape
        orig_imga_new[:,:,0] = 0.2989*orig_imga
        orig_imga_new[:,:,1] = 0.5870*orig_imga
        orig_imga_new[:,:,2] = 0.1141*orig_imga
        orig_imga = orig_imga_new    


    if params['color_space'] == 'rgb':
        orig_imga_conv = orig_imga
#     elif params['color_space'] == 'rg':
#         orig_imga_conv = colorconv.rg_convert(orig_imga)
    elif params['color_space'] == 'rg2':
        orig_imga_conv = colorconv.rg2_convert(orig_imga)
    elif params['color_space'] == 'gray':
        orig_imga_conv = colorconv.gray_convert(orig_imga)
        orig_imga_conv.shape = orig_imga_conv.shape + (1,)
    elif params['color_space'] == 'opp':
        orig_imga_conv = colorconv.opp_convert(orig_imga)
    elif params['color_space'] == 'oppnorm':
        orig_imga_conv = colorconv.oppnorm_convert(orig_imga)
    elif params['color_space'] == 'chrom':
        orig_imga_conv = colorconv.chrom_convert(orig_imga)
#     elif params['color_space'] == 'opponent':
#         orig_imga_conv = colorconv.opponent_convert(orig_imga)
#     elif params['color_space'] == 'W':
#         orig_imga_conv = colorconv.W_convert(orig_imga)
    elif params['color_space'] == 'hsv':
        orig_imga_conv = colorconv.hsv_convert(orig_imga)
    else:
        raise ValueError, "params['color_space'] not understood"
        
    return orig_imga,orig_imga_conv
    

def map_preprocessing(imga0,params): 
    
    assert(imga0.min() != imga0.max())
    
    # -- 0. preprocessing
    #imga0 = imga0 / 255.0
    
    # flip image ?
    if 'flip_lr' in params['preproc'] and params['preproc']['flip_lr']:
        imga0 = imga0[:,::-1]
        
    if 'flip_ud' in params['preproc'] and params['preproc']['flip_ud']:
        imga0 = imga0[::-1,:]            
    
    # smoothing
    lsum_ksize = params['preproc']['lsum_ksize']
    conv_mode = params['conv_mode']
    if lsum_ksize is not None:
         k = sp.ones((lsum_ksize), 'f') / lsum_ksize             
         imga0 = conv(conv(imga0, k[sp.newaxis,:], conv_mode), 
                      k[:,sp.newaxis], conv_mode)
         
    # whiten full image (assume True)
    if 'whiten' not in params['preproc'] or params['preproc']['whiten']:
        imga0 -= imga0.mean()
        if imga0.std() != 0:
            imga0 /= imga0.std()

    return imga0

     
def include_image_level_features(orig_imga,fvector_l,featsel):
    # include grayscale values ?
    f_input_gray = featsel['input_gray']
    if f_input_gray is not None:
        shape = f_input_gray
        #print orig_imga.shape
        fvector_l += [sp.misc.imresize(colorconv.gray_convert(orig_imga), shape).ravel()]

    # include color histograms ?
    f_input_colorhists = featsel['input_colorhists']
    if f_input_colorhists is not None:
        nbins = f_input_colorhists
        colorhists = sp.empty((3,nbins), 'f')
        if orig_imga.ndim == 3:
            for d in xrange(3):
                h = sp.histogram(orig_imga[:,:,d].ravel(),
                                 bins=nbins,
                                 range=[0,255])
                binvals = h[0].astype('f')
                colorhists[d] = binvals
        else:
            raise ValueError, "orig_imga.ndim == 3"
            #h = sp.histogram(orig_imga[:,:].ravel(),
            #                 bins=nbins,
            #                 range=[0,255])
            #binvals = h[0].astype('f')
            #colorhists[:] = binvals

        #feat_l += [colorhists.ravel()]
        fvector_l += [colorhists.ravel()]

    return fvector_l
    

def include_map_level_features(imga1,imga2,imga3,imga4,imga5,output,featsel):
    feat_l = []

    # include input norm histograms ? 
    f_normin_hists = featsel['normin_hists']
    if f_normin_hists is not None:
        division, nfeatures = f_norminhists
        feat_l += [v1f.rephists(imga1, division, nfeatures)]

    # include filter output histograms ? 
    f_filter_hists = featsel['filter_hists']
    if f_filter_hists is not None:
        division, nfeatures = f_filter_hists
        feat_l += [v1f.rephists(imga2, division, nfeatures)]

    # include activation output histograms ?     
    f_activ_hists = featsel['activ_hists']
    if f_activ_hists is not None:
        division, nfeatures = f_activ_hists
        feat_l += [v1f.rephists(imga3, division, nfeatures)]

    # include output norm histograms ?     
    f_normout_hists = featsel['normout_hists']
    if f_normout_hists is not None:
        division, nfeatures = f_normout_hists
        feat_l += [v1f.rephists(imga4, division, nfeatures)]

    # include representation output histograms ? 
    f_pool_hists = featsel['pool_hists']
    if f_pool_hists is not None:
        division, nfeatures = f_pool_hists
        feat_l += [v1f.rephists(imga5, division, nfeatures)]

    # include representation output ?
    f_output = featsel['output']
    if f_output and len(feat_l) != 0:
        fvector = sp.concatenate([output.ravel()]+feat_l)
    else:
        fvector = output    
   
    return fvector   
   

def get_gabor_filters(params):
    """ Return a Gabor filterbank (generate it if needed)
    
    Inputs:
    params -- filters parameters (dict)

    Outputs:
    filt_l -- filterbank (list)

    """
        
    global filt_l

    if filt_l is not None:
        return filt_l

    # -- get parameters
    fh, fw = params['kshape']
    orients = params['orients']
    freqs = params['freqs']
    phases = params['phases']
    nf =  len(orients) * len(freqs) * len(phases)
    fbshape = nf, fh, fw
    xc = fw/2
    yc = fh/2
    filt_l = []
    
    # -- build the filterbank
    for freq in freqs:
        for orient in orients:
            for phase in phases:
                # create 2d gabor
                filt = v1m.gabor2d(xc,yc,xc,yc,
                               freq,orient,phase,
                               (fw,fh))
                filt_l += [filt]
                
    return filt_l


class MinMaxError(Exception): 
    pass
    
     
 
