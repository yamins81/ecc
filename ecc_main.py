#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import cPickle

import boto
import gridfs
import scipy as sp

from starflow.utils import MakeDir
from starflow.protocols import protocolize, actualize

from utils import createCertificateDict, wget
from dbutils import connect_to_db, DATA_DB_NAME, get_cert_paths, get_col_root
from v1.v1like_extract import feature_extraction_protocol


ROOT = '../raw_data/'


from starflow.utils import MakeDir, activate, uniqify

def initialize(creates = ROOT):
    MakeDir(creates)

def get_ecc_images(depends_on = '../.ec2/credentials',creates = ROOT + 'ecc_images.zip'):
    boto.config.load_credential_file('../.ec2/credentials')
    conn = boto.connect_s3()
    b = conn.get_bucket('ecc-data')
    k = b.get_key('ecc-images-0.zip')
    k.get_contents_to_filename(creates)
    
    
def unzip_ecc_images(depends_on = ROOT + 'ecc_images.zip', creates = ROOT + 'images/'):
    os.system('cd ' + ROOT + ' && unzip ecc_images.zip')
    
    
def create_ecc_image_collection(depends_on = ROOT + 'images/', creates = get_cert_paths('images',None,None)):
    conn = connect_to_db()    
    
    colroot = get_col_root(steptag = 'images')
    colfiles = colroot + '.files'
    colchunks = colroot + '.chunks'
    
    db = conn[DATA_DB_NAME]
    if colfiles in db.collection_names():
        db.drop_collection(colfiles)
        db.drop_collection(colchunks)
    
    fs = gridfs.GridFS(db,collection = colroot)
    
    p = os.path.join(depends_on,'faces','caricature')
    L = filter(lambda x : x.endswith('.png'), os.listdir(p))
    for l in L:
        contents = open(os.path.join(p,l)).read()
        metadata = {'filename' : l, 'type': 'face', 'mode' : 'cartoon','subject':'human'}
        fs.put(contents,**metadata)
    p = os.path.join(depends_on,'faces','cartoon')
    L = filter(lambda x : x.endswith('.png'), os.listdir(p))
    for l in L:
        contents = open(os.path.join(p,l)).read()
        metadata = {'filename' : l, 'type': 'face', 'mode' : 'cartoon','subject':'monkey'}
        fs.put(contents,**metadata)
    p = os.path.join(depends_on,'faces','human')
    L = filter(lambda x : x.endswith('.png'), os.listdir(p))
    for l in L:
        contents = open(os.path.join(p,l)).read()
        metadata = {'filename' : l, 'type': 'face', 'mode' : 'photo','subject':'human'}
        fs.put(contents,**metadata)
    p = os.path.join(depends_on,'faces','monkey')
    L = filter(lambda x : x.endswith('.png'), os.listdir(p))
    for l in L:
        contents = open(os.path.join(p,l)).read()
        metadata = {'filename' : l, 'type': 'face', 'mode' : 'photo','subject':'monkey'}
        fs.put(contents,**metadata)    
        
    attdict = {}
    attdict['bodies'] = {'mode':'photo','subject':'body'}
    attdict['cars'] = {'mode':'photo','subject':'car'}
    attdict['cats'] = {'mode':'photo','subject':'cat'}
    attdict['cubes'] = {'mode':'generated','subject':'cube'}
    attdict['familar'] = {'mode':'photo','subject':'familiar'}
    attdict['round'] = {'mode':'photo','subject':'round'}
    attdict['smooths'] = {'mode':'generated','subject':'smooth'}
    attdict['spikes'] = {'mode':'generated','subject':'spike'}
    p = os.path.join(depends_on,'objects')
    L = filter(lambda x : os.path.isdir(os.path.join(p,x)),os.listdir(p))
    for l in L:
        p1 = os.path.join(p,l)
        L1 = filter(lambda x : x.endswith('.png'), os.listdir(p1))
        for l1 in L1:
            contents = open(os.path.join(p1,l1)).read()
            metadata = {'filename' : l1, 'type': 'object', 'mode':attdict[l]['mode'] , 'subject' : attdict[l]['subject']}
            fs.put(contents,**metadata)
      
    createCertificateDict(creates[0],{'root':colroot,'steptag':'images'})
    
    
@protocolize()
def v1like_a_feature_instantiator():
    D = feature_extraction_protocol('images','../v1/config/v1like_a.py')
    actualize(D)
  

from svm import classify, ova_classify, multi_classify
from splits import generate_split, generate_multi_split

def test_v1like_a_results_on_faces(depends_on = get_cert_paths('add_features','../v1/config/v1like_a.py',None) + ('../v1/config/v1like_a.py',), creates = '../faces_results/'):
    train_test('../v1/config/v1like_a.py','../faces_results/',{'type':'face'},30,80,N=30)

def test_v1like_a_results_on_human_faces(depends_on = get_cert_paths('add_features','../v1/config/v1like_a.py',None) + ('../v1/config/v1like_a.py',), creates = '../human_faces_results/'):
    train_test('../v1/config/v1like_a.py','../human_faces_results/',{'type':'face','subject':'human','mode':'photo'},20,60,N=15,universe={'$or':[{'type':'face','subject':'human','mode':'photo'},{'type':'object'}]})

def test_v1like_a_results_on_human_faces_vs_familiar(depends_on = get_cert_paths('add_features','../v1/config/v1like_a.py',None) + ('../v1/config/v1like_a.py',), creates = '../human_faces_results_vs_familiar/'):
    train_test('../v1/config/v1like_a.py',creates,{'type':'face','mode':'photo'},15,30,N=15,universe={'$or':[{'type':'face','mode':'photo'},{'type':'object','subject':'familiar'}]})

    
def test_v1like_a_results_on_humans_vs_monkey_faces(depends_on = get_cert_paths('add_features','../v1/config/v1like_a.py',None) + ('../v1/config/v1like_a.py',), creates = '../humans_vs_monkey_faces_results/'):
    train_test('../v1/config/v1like_a.py',creates,{'subject':'human'},15,40,universe={'type':'face'})

def test_v1like_a_results_on_photo_vs_cartoonfaces(depends_on = get_cert_paths('add_features','../v1/config/v1like_a.py',None) + ('../v1/config/v1like_a.py',), creates = '../photo_vs_cartoon_faces_results/'):
    train_test('../v1/config/v1like_a.py',creates,{'mode':'photo'},15,40,universe={'type':'face'})
   
def test_v1like_a_results_on_faces_vs_animals(depends_on = get_cert_paths('add_features','../v1/config/v1like_a.py',None) + ('../v1/config/v1like_a.py',), creates = '../faces_vs_objects_vs_animals_results/'):
    train_test('../v1/config/v1like_a.py',creates,[{'type':'face'},{'type':'object','subject':{'$in':['human','cat']}}],30,80)
    
def test_v1like_a_results_on_faces_vs_animals_cramer_singer(depends_on = get_cert_paths('add_features','../v1/config/v1like_a.py',None) + ('../v1/config/v1like_a.py',), creates = '../faces_vs_objects_vs_animals_results_cs/'):
    train_test('../v1/config/v1like_a.py',creates,[{'type':'face'},{'type':'object','subject':{'$in':['human','cat']}}],30,80,classifier = multi_classify, classifier_kwargs = {'multi_class':True})

def test_v1like_a_results_on_faces_vs_familiar_cramer_singer(depends_on = get_cert_paths('add_features','../v1/config/v1like_a.py',None) + ('../v1/config/v1like_a.py',), creates = '../faces_vs_objects_vs_familiar_results_cs/'):
    train_test('../v1/config/v1like_a.py',creates,[{'type':'face'},{'type':'object','subject':{'$in':['human','cat','familiar','car']}}],30,80,classifier = multi_classify, classifier_kwargs = {'multi_class':True})

  
def test_v1like_a_results_on_faces_vs_humans_cramer_singer(depends_on = get_cert_paths('add_features','../v1/config/v1like_a.py',None) + ('../v1/config/v1like_a.py',), creates = '../faces_vs_human_results_cs/'):
    train_test('../v1/config/v1like_a.py',creates,[{'type':'face'},{'type':'object','subject':'human'}],30,80,classifier = multi_classify, classifier_kwargs = {'multi_class':True})
    
def test_v1like_a_results_on_faces_vs_humans(depends_on = get_cert_paths('add_features','../v1/config/v1like_a.py',None) + ('../v1/config/v1like_a.py',), creates = '../faces_vs_human_results/'):
    train_test('../v1/config/v1like_a.py',creates,[{'type':'face'},{'type':'object','subject':'human'}],30,80)
   

def train_test(config_path,outdir,query,ntrain,ntest,classifier = None,classifier_kwargs = {},N=10,universe=None):
    MakeDir(outdir)

    if isinstance(query,dict):
        splitter = generate_split
        classifier = classify
    else:
        splitter = generate_multi_split
        if classifier is None:
            classifier = ova_classify
               
    split_data = []
    results = []

    for i in range(N):
        print(i)
        split = splitter(config_path,'add_features',query,ntrain,ntest,universe=universe)
        train_data = split['train_data']
        train_features = split['train_features']
        train_labels = split['train_labels']
        test_data = split['test_data']
        test_features = split['test_features']
        test_labels = split['test_labels']

        if (not classifier_kwargs.get('multi_class')) or len(uniqify(train_labels)) > 2:
            train_filenames = [t['filename'] for t in train_data]
            test_filenames = [t['filename'] for t in test_data]
            split_data.append({'train_filenames':train_filenames,'train_labels':train_labels,
                           'test_filenames': test_filenames,'test_labels':test_labels})
                           
            assert set(train_filenames).intersection(test_filenames) == set([])
            res = classifier(train_features,train_labels,test_features,test_labels,**classifier_kwargs)
        
            results.append(res)

    stats = ['test_accuracy','ap','auc','mean_ap','mean_auc','train_accuracy']
    
    output = {'split_results' : results}
    
    for stat in stats:
        if stat in results[0] and results[0][stat] != None:
            output[stat] = sp.array([result[stat] for result in results]).mean()
    

    F = open(os.path.join(outdir,'splits.pickle'),'w')
    cPickle.dump(split_data,F)
    F.close()
    F = open(os.path.join(outdir,'results.pickle'),'w')
    cPickle.dump(output,F)
    F.close()
    


    