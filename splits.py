import cPickle
from dbutils import DataCollection
from starflow.utils import uniqify, ListUnion
import scipy as sp

"""
generate splits from the DB. 
"""

def generate_split(config_path,tag,task_query,ntrain,ntest,universe = None):

    if universe is None:
        universe = {}

    data = DataCollection(config_path = config_path, steptag = tag)
    task_query.update(universe)
    
    task_data = list(data.find(task_query))
    task_ids = [x['_id'] for x in task_data]
    N_task = len(task_data)
    
    nontask_query = {'_id':{'$nin':task_ids}}
    nontask_query.update(universe)
    nontask_data = list(data.find(nontask_query))
    N_nontask = len(nontask_data)
    
    all_data = task_data + nontask_data
         
    assert ntrain + ntest <= N_task + N_nontask, "Too many training and/or testing examples."
    
    perm = sp.random.permutation(len(all_data))
         
    train_data = [all_data[i] for i in perm[:ntrain]]
    test_data = [all_data[i] for i in perm[ntrain:ntrain + ntest]]
    
    train_labels = sp.array([x['_id'] in task_ids for x in train_data])
    test_labels = sp.array([x['_id'] in task_ids for x in test_data])

    train_features = sp.row_stack([cPickle.loads(data.fs.get(r['_id']).read()) for r in train_data])
    test_features = sp.row_stack([cPickle.loads(data.fs.get(r['_id']).read()) for r in test_data])
    
    return {'train_data': train_data, 'test_data' : test_data, 'train_features' : train_features,'train_labels':train_labels,'test_features':test_features,'test_labels':test_labels}


def validate(idseq):
    ids = ListUnion(idseq)
    ids1 = [id[1] for id in ids]
    assert len(uniqify(ids1)) == sum([len(X) for X in idseq]), 'Classes are not disjoint.'
    return ids
    
    
def generate_multi_split(config_path,tag,queries,ntrain,ntest,universe = None):

    if universe is None:
        universe = {}
    
    for q in queries:
        q.update(universe)
        
    data = DataCollection(config_path = config_path, steptag = tag)
    
    task_data_list = [list(data.find(query)) for query in queries]
    task_id_list = [[(i,x['_id']) for x in X] for (i,X) in enumerate(task_data_list)]
    task_data = ListUnion(task_data_list)
    task_ids = validate(task_id_list)
    task_dist, task_ids = zip(*task_ids)
    task_dist = list(task_dist) ; task_ids = list(task_ids)
        
    nontask_query = {'_id':{'$nin':task_ids}}    
    nontask_query.update(universe)
    nontask_data = list(data.find(nontask_query)) 
    nontask_ids = [x['_id'] for x in nontask_data]
        
    all_ids = task_ids + nontask_ids
    all_data = task_data + nontask_data
    all_dist = task_dist + [len(queries)]*len(nontask_ids)
    
    assert ntrain + ntest <= len(all_ids)
    
    perm = sp.random.permutation(len(all_ids))
  
    train_ids = [all_ids[i] for i in perm[:ntrain]]
    test_ids = [all_ids[i] for i in perm[ntrain:ntrain + ntest]]
        
    train_data = [all_data[i] for i in perm[:ntrain]]
    test_data = [all_data[i] for i in perm[ntrain:ntrain+ntest]]
    
    train_labels = sp.array([all_dist[i] for i in perm[:ntrain]])
    test_labels = sp.array([all_dist[i] for i in perm[ntrain:ntrain+ntest]]) 

    train_features = sp.row_stack([cPickle.loads(data.fs.get(r).read()) for r in train_ids])
    test_features = sp.row_stack([cPickle.loads(data.fs.get(r).read()) for r in test_ids])
    
    return {'train_data': train_data,
            'test_data' : test_data, 
            'train_features' : train_features,
            'train_labels':train_labels,
            'test_features':test_features,
            'test_labels':test_labels
           }

 