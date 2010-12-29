import cPickle 
import pickle
import time
import os
import hashlib

import pymongo
import gridfs

from utils import createCertificateDict
from starflow.utils import activate, MakeDir, is_string_like


CERTIFICATE_ROOT = '../.db_certificates'

def initialize_certificates(creates = CERTIFICATE_ROOT):
    MakeDir(creates)

def initialize_ecc_db(creates = '../mongodb/'):
    initialize_db(creates,'ecc_db')

def initialize_db(path,name,host=None,port=None):
    os.mkdir(path)
  
    #make config
    config = {}
    config['dbpath'] = os.path.abspath(path)
    config['logpath'] = os.path.abspath(os.path.join(path,'log'))
    config['startlog'] = os.path.abspath(os.path.join(path,'startlog'))
    config['name'] = name
    
    confpath = os.path.join(path,'conf')
    F = open(confpath,'w')
    pickle.dump(config,F)
    F.close()
    
    start_db(path,host,port)
    time.sleep(10)
    conn = pymongo.Connection(host,port)
    
    db = conn['__info__']
    coll = db['__info__']
        
    coll.insert({'_id' : 'ID', 'path': path, 'name' : name},safe=True)


DB_BASE_PATH = '../mongodb/'
DATA_DB_NAME = 'data'
DATACOL_COL_NAME = '__datacols__'

def connect_to_db(depends_on = DB_BASE_PATH,host = None, port = None,verify=True):

    path = depends_on
    try:
        conn = pymongo.Connection(host,port)
    except pymongo.errors.AutoReconnect:
        start_db(path,host,port)  
        time.sleep(10)
        conn = pymongo.Connection(host,port)
    else:
        pass

    if verify:
        verify_db(conn,path)
    
    return conn
    
    
def verify_db(conn,path):
    confpath = os.path.join(path,'conf')
    
    config = pickle.load(open(confpath))
    name = config['name']
    
    if '__info__' not in conn.database_names():
        raise NoInfoDBError()
    infodb = conn['__info__']
    
    if '__info__' not in infodb.collection_names():
        raise NoInfoCollError()
    infocoll = infodb['__info__']
        
    X = infocoll.find_one({'_id' : 'ID'})
    if not X or not X.get('name') or not X.get('path'):
        raise NoIDRecError()
        
    if not X['name'] == name: 
        raise WrongNameError(name,X['name'])
    
    if not X['path'] == path:
        raise WrongPathError(path,X['path'])
    
class DBError(BaseException):
    pass    
        
class NoInputCollectionError(DBError):
    def __init__(self,incolname):
        self.msg = 'Input collection %s not found in db.' % incolname

class VerificationError(BaseException):
    pass
       
class NoInfoDBError(VerificationError):
    def __init__(self):
        self.msg = 'No __info__ database found.'
    
class NoInfoCollError(VerificationError):
    def __init__(self):
        self.msg = 'No __info__ collection found.'
 
class NoIDRecError(VerificationError):
    def __init__(self):
        self.msg = 'No ID rec found.'

class WrongNameError(VerificationError):
    def __init__(self,name,xname):
        self.msg = 'Wrong name: should be %s but is %s.' % (name,xname)
        
class WrongPathError(VerificationError):
    def __init__(self,name,xname):
        self.msg = 'Wrong path: should be %s but is %s.' % (name,xname)    
               
def start_db(path,host=None,port=None):
    confpath = os.path.join(path,'conf')
    
    config = pickle.load(open(confpath))
    
    dbpath = config['dbpath']
    logpath = config['logpath']
    startlog = config['startlog']

    optstring = '--dbpath ' + dbpath + ' --fork --logpath ' + logpath 
    
    if host != None:
        optstring += ' --bind_ip ' + host 
    if port != None:
        optstring += ' --port ' + str(port)
        
    print("DOING",'mongod ' + optstring + ' > ' + startlog)    
    os.system('mongod ' + optstring + ' > ' + startlog)
    
import json
def get_config(config_path):
    config = {}
    config_path = os.path.abspath(config_path)
    execfile(config_path,config)
    keys = config.keys()
    for k in keys:
        try:
            json.dumps(config[k])
        except TypeError:
            config.pop(k)
        else:
            pass
    
    return config

def get_col_root(config_path = None, config = None, steptag = None,conn = None):

    if conn is None:
       conn = connect_to_db()
   
    db = conn[DATA_DB_NAME]
   
    colcol = db[DATACOL_COL_NAME]
    colcol.ensure_index('root',unique=True)
    
    if config is None:
        if config_path is not None:
            config = get_config(config_path)
            
    existing = colcol.find_one({'steptag': steptag, 'config' : config})
    if not existing or not existing.get('root'):
        str = repr(config) + steptag
        root = hashlib.sha1(str).hexdigest()
        colcol.insert({'root':root,'config' : config, 'steptag': steptag},safe=True)
    else:
        root = existing['root']
        

    return root

    
class DataCollection(pymongo.collection.Collection):
    def __init__(self,config_path = None, config = None, steptag = None, root = None,conn = None):
        if conn is None:
           conn = connect_to_db()
        db = conn[DATA_DB_NAME]
        
        if root is None:
            root = get_col_root(config_path=config_path,config = config,steptag=steptag,conn=conn)
        name = root + '.files'
   
        pymongo.collection.Collection.__init__(self,db,name)
       
        self.fs = gridfs.GridFS(db,root)
        colcol = db[DATACOL_COL_NAME]  
        rec = colcol.find_one({'root' : root})
        self.config = rec['config']
        self.steptag = rec['steptag']
        
    def get(self,query):
        L = self.find(query)
        return [self.fs.get(l['_id']) for l in L]
    
    def put(self,contents,**args):
        self.fs.put(contents,**args)
        

def steptag(tag):
    def tagger(f):
        f.steptags = tag
        return f
    return tagger
    
def get_cert_paths(tags,config_path,query):
    if is_string_like(tags):
        tags = [tags]
    
    qstr = '__' + repr(query) if query else ''
    return tuple([os.path.join(CERTIFICATE_ROOT,get_col_root(config_path = config_path,steptag = tag) + qstr) for tag in tags])
    
    
def get_tags(func,args,kwargs):

    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if hasattr(func.steptags,'__call__'):
        outsteptags = func.steptags(*args,**kwargs)
    else:
        outsteptags = func.steptags
    if is_string_like(outsteptags):
        outsteptags = (outsteptags,)

    assert isinstance(outsteptags,tuple) and all([is_string_like(x) for x in outsteptags])
    
    return outsteptags
    
@activate(lambda x : get_cert_paths(x[0],x[2],x['inquery'])  + ((x[2],) if x[2] else ()),lambda x : get_cert_paths(get_tags(x[1],x['args'],x['kwargs']),x[3],x['outquery']))
def db_computation_handler(insteptags,func,inconfig_path,outconfig_path,filebuncher = None, args = None, kwargs = None,inquery = None, outquery = None):
    

    if filebuncher is None:
       filebuncher = lambda X : [[x] for x in X]
       
    conn = connect_to_db()
    db = conn[DATA_DB_NAME]
    
    if is_string_like(insteptags):
        insteptags = [insteptags]
        
    incertpaths = get_cert_paths(insteptags,inconfig_path,inquery)    
    incertdicts =  [cPickle.load(open(incertpath)) for incertpath in incertpaths]
    incolroots = [incertdict['root'] for incertdict in incertdicts]
    incolnames = [incolroot + '.files' for incolroot in incolroots]
    
    if inquery is None:
        inquery = {}
    if outquery is None:
        outquery = {}
    assert inquery <= outquery
    
    for incolname in incolnames:
        if incolname not in db.collection_names():
            raise NoInputCollectionError(incolname)
    incols = [db[incolname] for incolname in incolnames]
    inrecs = zip(*[list(incol.find(inquery)) for incol in incols])
    inrecs = [zip(insteptags,rec) for rec in inrecs]
       
    outsteptags = get_tags(func,args,kwargs)

    outcolroots = [get_col_root(steptag = steptag,config_path = outconfig_path,conn = conn) for steptag in outsteptags]

    recgroups = filebuncher(inrecs)
     
    in_fs = dict([(intag,gridfs.GridFS(db,collection = incolroot)) for (intag,incolroot) in zip(insteptags,incolroots)])
    out_fs = dict([(outtag,gridfs.GridFS(db,collection = outcolroot)) for (outtag,outcolroot) in zip(outsteptags,outcolroots)])
    
    if kwargs is None:
       kwargs = {}
    if args is None:
       args = ()    
    
    for recs in recgroups:    

        in_fhs = [[(r[0],in_fs[r[0]].get(r[1]['_id'])) for r in rec] for rec in recs]
        results = func(in_fhs,outconfig_path,*args,**kwargs)
        if is_string_like(results):
            results = (results,)
        if isinstance(results,tuple):
            results = [results]
       
        assert len(results) == len(recs), 'something wrong in your function'  
        assert all([len(result) == len(outsteptags) for result in results])
       
        for (rec,result) in zip(recs,results):
            for (outtag,res) in zip(outsteptags,result):
                print(rec[0][1]['_id'])
                out_fs[outtag].delete(rec[0][1]['_id'])
                out_fs[outtag].put(res,**rec[0][1])
    
    outcertpaths = get_cert_paths(outsteptags,outconfig_path,outquery)
    for (outcertpath,outsteptag,outcolroot) in zip(outcertpaths,outsteptags,outcolroots):
        createCertificateDict(outcertpath,{'config_path': outconfig_path, 'steptag': outsteptag, 'root' : outcolroot,'query':outquery})
    
    
           
   
   
       