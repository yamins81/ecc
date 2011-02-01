import scipy as sp
import numpy as np
from scikits.learn import svm
from scikits.learn.linear_model.logistic import LogisticRegression

'''
SVM classifier module
'''
    

def classify(train_features,
                     train_labels,
                     test_features,
                     test_labels, sphere=True):

    '''Classify data and return
        accuracy
        area under curve
        average precision
        and svm raw data in a dictianary'''

    #mapping labels to 0,1
    labels = sp.unique(sp.concatenate((train_labels, test_labels)))
    assert labels.size == 2
    label_to_id = dict([(k,v) for v, k in enumerate(labels)])

    train_ys = sp.array([label_to_id[i] for i in train_labels])
    test_ys = sp.array([label_to_id[i] for i in test_labels])

    #train
    model = classifier_train(train_features, train_ys,
                            test_features,sphere=sphere)

    #test
    weights = model.coef_.ravel()
    bias = model.intercept_.ravel()
    test_predictor = sp.dot(test_features, weights) + bias
    test_prediction = model.predict(test_features)
    train_prediction = model.predict(train_features)

    #raw data to be saved for future use
    cls_data = {'test_prediction' : test_prediction,  
                'test_lables' : test_labels, 
                'coef' : model.coef_, 
                'intercept' : model.intercept_
               }

    #accuracy
    test_accuracy = 100*(test_prediction == test_ys).sum()/float(len(test_ys))
    train_accuracy = 100*(train_prediction == train_ys).sum()/float(len(train_ys))
    
    #precison and recall
    c = test_predictor
    si = sp.argsort(-c)
    tp = sp.cumsum(sp.single(test_ys[si] == 1))
    fp = sp.cumsum(sp.single(test_ys[si] == 0))
    rec = tp /sp.sum(test_ys > 0)
    prec = tp / (fp + tp)
    
    ap = 0
    rng = sp.arange(0, 1.1, .1)
    for th in rng:
        p = prec[rec>=th].max()
        if p == []:
               p =0
        ap += p / rng.size

    #area under curve
    h = sp.diff(rec)
    auc = sp.sum(h * (prec[1:] + prec[:-1])) / 2.0


    return {'auc':auc,
            'ap':ap, 
            'train_accuracy': train_accuracy,
            'test_accuracy' : test_accuracy,
            'cls_data':cls_data
           }


        
def ova_classify(train_features,
                     train_labels,
                     test_features,
                     test_labels):
                     
    """
    Classifier using one-vs-all on top of liblinear binary classification.  
    Computes mean average precision (mAP) and mean area-under-the-curve (mAUC)
    by averaging these measure of the binary results. 
    """
                     
    train_features, test_features = __sphere(train_features, test_features)

    labels = sp.unique(sp.concatenate((train_labels, test_labels)))
    label_to_id = dict([(k,v) for v, k in enumerate(labels)])

    train_ids = sp.array([label_to_id[i] for i in train_labels])
    test_ids = sp.array([label_to_id[i] for i in test_labels])
    all_ids = sp.array(range(len(labels)))

    classifiers = []
    aps = []
    aucs = []
    cls_datas = []

    signs = []
    for id in all_ids: 
        binary_train_ids = sp.array([2*int(l == id) - 1 for l in train_ids])
        binary_test_ids = sp.array([2*int(l == id) - 1 for l in test_ids])
        signs.append(binary_train_ids[0])   
        
        res = classify(train_features, binary_train_ids, test_features, binary_test_ids,sphere=False)
        
        aps.append(res['ap'])
        aucs.append(res['auc'])
        cls_datas.append(res['cls_data'])
    
    mean_ap = sp.array(aps).mean()
    mean_auc = sp.array(aucs).mean()
    
    signs = sp.array(signs)
    weights = signs * (sp.row_stack([cls_data['coef'] for cls_data in cls_datas]).T)
    bias = signs * (sp.row_stack([cls_data['intercept'] for cls_data in cls_datas]).T)
    
    predictor = max_predictor(weights,bias,labels)
  
    test_prediction = predictor(test_features)
    test_accuracy = 100*(test_prediction == test_labels).sum() / len(test_prediction)

    train_prediction = predictor(train_features)
    train_accuracy = 100*(train_prediction == train_labels).sum() / len(train_prediction)

    cls_data = {'coef' : weights, 
     'intercept' : bias, 
     'train_labels': train_labels,
     'test_labels' : test_labels,
     'train_prediction': train_prediction, 
     'test_prediction' : test_prediction,
     'labels' : labels
     }


    return {'cls_data' : cls_data,
     'train_accuracy' : train_accuracy,
     'test_accuracy' : test_accuracy,
     'mean_ap' : mean_ap,
     'mean_auc' : mean_auc
     }
     

def multi_classify(train_features,
                     train_labels,
                     test_features,
                     test_labels,
                     multi_class = False):
    """
    Classifier using the built-in multi-class classification capabilities of liblinear
    """

    labels = sp.unique(sp.concatenate((train_labels, test_labels)))
    label_to_id = dict([(k,v) for v, k in enumerate(labels)])
 
    train_ids = sp.array([label_to_id[i] for i in train_labels])
    test_ids = sp.array([label_to_id[i] for i in test_labels])
    
    classifier = classifier_train(train_features, train_ids, test_features, multi_class = multi_class)
    weights = classifier.coef_.T
    bias = classifier.intercept_
        
    test_prediction_ids = classifier.predict(test_features)
    test_prediction = labels[test_prediction_ids]
    test_accuracy = 100*(test_prediction == test_labels).sum() / len(test_prediction)
 
    train_prediction = labels[classifier.predict(train_features)]
    train_accuracy = 100*(train_prediction == train_labels).sum() / len(train_prediction)

    cls_data = {'coef' : weights, 
     'intercept' : bias, 
     'train_labels': train_labels,
     'test_labels' : test_labels,
     'train_prediction': train_prediction, 
     'test_prediction' : test_prediction,
     'labels' : labels
     }


    return {'cls_data' : cls_data,
     'train_accuracy' : train_accuracy,
     'test_accuracy' : test_accuracy,
     'mean_ap' : None,
     'mean_auc' : None
     }
 
          
def classifier_train(train_features,
                     train_labels,
                     test_features,
                     svm_eps = 1e-5,
                     svm_C = 10**4,
                     classifier_type = "liblinear",
                     multi_class=False,
                     sphere = True
                     ):
    """ Classifier training using SVMs

    Input:
    train_features = training features (both positive and negative)
    train_labels = corresponding label vector
    svm_eps = eps of svm
    svm_C = C parameter of svm
    classifier_type = liblinear or libsvm"""

    #sphering
    if sphere:
        train_features, test_features = __sphere(train_features, test_features)

    if classifier_type == 'liblinear':
        clf = svm.LinearSVC(eps = svm_eps, C = svm_C,multi_class=multi_class)
    if classifier_type == 'libSVM':
        clf = svm.SVC(eps = svm_eps, C = svm_C, probability = True)
    elif classifier_type == 'LRL1':
        clf = LogisticRegression(C=svm_C, penalty = 'l1')
    elif classifier_type == 'LRL2':
        clf = LogisticRegression(C=svm_C, penalty = 'l1')

    clf.fit(train_features, train_labels)
    
    return clf

#sphere data
def __sphere(train_data, test_data):
    '''make data zero mean and unit variance'''

    fmean = train_data.mean(0)
    fstd = train_data.std(0)

    train_data -= fmean
    test_data -= fmean
    fstd[fstd==0] = 1
    train_data /= fstd
    test_data /= fstd

    return train_data, test_data
     
def max_predictor(weights,bias,labels):
    return lambda v : labels[(sp.dot(v,weights) + bias).argmax(1)]

def liblinear_predictor(clas, bias, labels):
    return lambda x : labels[liblinear_prediction_prediction_function(x,clas,labels)]

def liblinear_prediction_function(farray , clas, labels):

    if len(labels) > 2:
        nf = farray.shape[0]
        nlabels = len(labels)
        
        weights = clas.raw_coef_.ravel()
        nw = len(weights)
        nv = nw / nlabels
        
        D = np.column_stack([farray,np.array([.5]).repeat(nf)]).ravel().repeat(nlabels)
        W = np.tile(weights,nf)
        H = W * D
        H1 = H.reshape((len(H)/nw,nv,nlabels))
        H2 = H1.sum(1)
        predict = H2.argmax(1)
        
        return predict
    else:
    
        weights = clas.coef_.T
        bias = clas.intercept_
        
        return (1 - np.sign(np.dot(farray,weights) + bias) )/2
        
