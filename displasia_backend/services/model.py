import os
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import sklearn as sk
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif

from sklearn.neural_network import MLPClassifier
from itertools import combinations
from anchor import utils
from anchor import anchor_tabular

from sklearn.inspection import PartialDependenceDisplay
np.random.seed(1)
import shap

import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from ..models import Classe, GlobalData, Names
from ..utils import util
from ..db.repository import amostra

import warnings
warnings.filterwarnings('ignore')

global_data = GlobalData()
names = np.array(Names().attributes_used, dtype=str)
global_data.data = util.init_data(global_data.class_names)
for estado in global_data.data.keys():
            global_data.data[estado] = global_data.data[estado][names]
global_data.data, global_data.target = util.get_data(global_data.data, global_data.class_names)
global_data.data.columns = np.array(Names().eng_names, dtype=str)
global_data.train_data, global_data.test_data, global_data.train_target, global_data.test_target = train_test_split(global_data.data, global_data.target, train_size = 0.8, random_state = 1)

#index is the name in train data
#index is id - 1
def train():
    
    global_data.train_data, global_data.test_data = remove_low_variance(global_data.train_data, global_data.test_data, 0.01)
    global_data.train_data, global_data.test_data = remove_correlated(global_data.train_data, global_data.train_target, global_data.test_data, 0.99)
    corr_target = show_correlated(global_data.train_data, global_data.train_target, global_data.test_data, 0.97)
    corr_plot(corr_target, global_data.target, global_data.global_corr.path)
    global_data.scaler, global_data.scaled_train_data, global_data.scaled_test_data =scale_data()
    model = MLPClassifier(max_iter = 500, random_state=1)
    model.fit(global_data.scaled_train_data, global_data.train_target)

    test_pred_prob = model.predict_proba(global_data.scaled_test_data)
    log_loss = metrics.log_loss(global_data.test_target, test_pred_prob)
    auc = metrics.roc_auc_score(global_data.test_target, test_pred_prob, multi_class = 'ovr')

    test_pred = model.predict(global_data.scaled_test_data)
    acc = metrics.accuracy_score(global_data.test_target, test_pred)
    fscore = metrics.f1_score(global_data.test_target, test_pred, average='macro')
    recall = metrics.recall_score(global_data.test_target, test_pred, average='macro')
    print("Log Loss: ", log_loss)
    print("AUC: ", auc)
    print("Acuracia: ", acc)
    print("F-score: ", fscore)
    print("Recall: ", recall)
    with open("displasia_backend/static/model.joblib", 'wb') as file:
        joblib.dump(model, file)
    modelos_bin = train_bin_models(global_data.class_names, global_data.scaled_train_data, global_data.train_target)
    with open("displasia_backend/static/modelbin.joblib", 'wb') as file:
        joblib.dump(modelos_bin, file)
    #global_shap(model)

def remove_low_variance(train_data, test_data, threshold):
    selector = VarianceThreshold()
    selector.fit(train_data)
    mask = selector.get_support()
    train_data = train_data.loc[:,mask]
    test_data = test_data.loc[:,mask]
    
    return(train_data, test_data)
    

def remove_correlated(train_data, train_target, test_data, threshold):
    train_data['target'] = train_target
    corr_mat = train_data.corr().abs()
    removed = np.zeros(len(train_data.columns) - 1, dtype=bool)
    for f_pos, feat in enumerate(train_data.columns):
        if(feat == 'target'):
            continue
        if(removed[f_pos]):
            continue
            
        correlated_group = []
        for feat2 in train_data.drop('target',axis = 1).columns[np.invert(removed)]:
            if(feat2 == 'target'):
                continue
            if(corr_mat[feat][ feat2] >= threshold):
                correlated_group.append(feat2)
        
        
        if len(correlated_group) > 1:
            better_of_group = None
            better_corr = 0
            for f in correlated_group:
                if (corr_mat[f]['target'] > better_corr):
                    better_corr = corr_mat[f]['target']
                    better_of_group = f
            correlated_group.remove(better_of_group)
            for duplicate_feat in correlated_group:
                removed[train_data.columns.get_loc(duplicate_feat)] = True
    train_data.drop('target',axis = 1, inplace=True)
    train_data.drop(train_data.columns[removed], axis = 1, inplace=True)
    test_data.drop(test_data.columns[removed],axis = 1, inplace= True)
    return (train_data, test_data)

def show_correlated(train_data, train_target, test_data, threshold):
    train_data['target'] = train_target
    corr_mat = train_data.corr().abs()
    removed = np.zeros(len(train_data.columns) - 1, dtype=bool)
    for f_pos, feat in enumerate(train_data.columns):
        if(feat == 'target'):
            continue
        if(removed[f_pos]):
            continue
            
        correlated_group = []
        for feat2 in train_data.drop('target',axis = 1).columns[np.invert(removed)]:
            if(feat2 == 'target'):
                continue
            if(corr_mat[feat][ feat2] >= threshold):
                correlated_group.append(feat2)
        
        
            
        for selected_feat in correlated_group:
                removed[train_data.columns.get_loc(selected_feat)] = True
    train_data.drop('target',axis = 1, inplace=True)
    return (corr_mat['target'].drop('target'))

def corr_plot(corr_target, target, path):
    handle = path
    corr_target.sort_values().head(10).plot(kind='barh',x = target, ylabel = 'Atributo', xlabel = 'Correlação com a variável alvo')
    plt.savefig(handle, bbox_inches='tight')
    return handle


def scale_data():
    scaler = StandardScaler()
    scaled_train_data = pd.DataFrame(scaler.fit_transform(global_data.train_data), columns = global_data.train_data.columns, index = global_data.train_data.index)
    scaled_test_data = pd.DataFrame(scaler.transform(global_data.test_data), columns = global_data.test_data.columns, index = global_data.test_data.index)
    return (scaler, scaled_train_data, scaled_test_data)



def local_shap(model, scaler, sample):
    handler = 'displasia_backend/static/shap.save'
    if not os.path.exists(handler) or os.path.getsize(handler) == 0:
        return
    with open(handler, 'rb') as file:   
        shap_explainer = joblib.load(file)
    pred = model.predict(scaler.transform([sample]))[0]
    scaled_sample = scaler.transform([sample])[0]
    np.random.seed(10)
    local_shap_values = shap_explainer.shap_values(scaled_sample)
    shap.force_plot(shap_explainer.expected_value[pred], local_shap_values[pred], sample, link="logit", matplotlib=True, show=False)
    plt.savefig('displasia_backend/static/shap_local.jpg', bbox_inches='tight')
    return local_shap_values

def global_shap(model):
    handler = 'displasia_backend/static/shap.save'
    if os.path.exists(handler) and os.path.getsize(handler) != 0:
        with open(handler, 'rb') as file:
            shap_explainer = joblib.load(file)
    else:
        train_data_summary = shap.kmeans(global_data.scaled_train_data, int(len(global_data.scaled_train_data)*15/100))
        shap_explainer = shap.KernelExplainer(model.predict_proba, train_data_summary)
        
    shap_values = shap_explainer.shap_values(global_data.scaled_train_data)
    shap.summary_plot(shap_values, global_data.scaled_test_data, feature_names=global_data.scaled_train_data.columns,class_names=['saudavel','leve','moderada','severa'], show=False)
    plt.savefig('displasia_backend/static/shap.jpg', bbox_inches='tight')
    plt.close()
    with open(handler, 'wb') as file:
        joblib.dump(shap_explainer, file)
    sv = np.array(shap_values)
    impacts = []
    for label in range(sv.shape[0]):
        impacts.append([np.average(abs(sv[label,:,i])) for i in range(sv.shape[2])])
    imp_df = pd.DataFrame(np.transpose(np.array(impacts)),columns=['saudavel','leve','moderada','severa'])
    imp_df.index = global_data.scaled_train_data.columns
    imp_df['total'] = imp_df['saudavel'] + imp_df['leve'] + imp_df['moderada'] + imp_df['severa']
    imp_df = imp_df.sort_values(by='total',ascending=False)

    mi_shap = imp_df.index[:2]

def train_bin_models(class_names, scaled_train_data, train_target):
    modelos_bin = []
    for i,(c1,c2) in enumerate(combinations(range(len(class_names)),2)):
        bin_train_target = train_target[(train_target == c1) | (train_target == c2)]
        bin_train_data = scaled_train_data[(train_target == c1) | (train_target == c2)]
        
        b_model = MLPClassifier(max_iter = 500, random_state=1)
        b_model.fit(bin_train_data, bin_train_target)
        modelos_bin.append((b_model,(c1,c2)))
    return modelos_bin

def predict(sample_id):
    model_handler = 'displasia_backend/static/model.joblib'
    modelbin_handler = 'displasia_backend/static/modelbin.joblib'
    conclusivo = False
    if os.path.exists(model_handler) and os.path.getsize(model_handler) != 0:
        with open(model_handler, 'rb') as file:
            model = joblib.load(file)
    if os.path.exists(modelbin_handler) and os.path.getsize(modelbin_handler) != 0:
        with open(modelbin_handler, 'rb') as file:
            modelbin = joblib.load(file)
    
    sample_data = amostra.get_amostra_by_id_dt(sample_id)
    sample_data = util.used_names(Names().attributes_used, sample_data)
    sample_data = util.change_name(Names().eng_names, sample_data)
    
    for key in sample_data.keys():
        if not key in global_data.test_data.keys().values:
            sample_data.drop(key, inplace=True)

    
    pred = model.predict(global_data.scaler.transform([sample_data]))[0]
    print("predição do modelo multiclasse:",pred)
    
    pred_bs = []
    
    for bin_model in modelbin:
        if pred in bin_model[1]:
            pred_b = bin_model[0].predict([sample_data])[0]
            pred_bs.append(pred_b)
    
    qtds = {}
    for p in pred_bs:
        if (p not  in qtds):
            qtds[p] = 1
        else:
            qtds[p] += 1
    
    c_qtd = 0
    for (key,value) in qtds.items():
        if(value > c_qtd):
            c_maioria = key
            c_qtd = value

    if(len(qtds.keys()) == 3):
        print("nao há consensso")
    else:
        print("a maioria dos modelos binarios concorda com:",c_maioria)
        conclusivo = True
    
    for i in range(3):
        if(i not in qtds.keys()):
            print('classe', i, 'foi descartada')

    preds = [pred,pred_bs]
    print(preds)
    exps = local_anchor(global_data.class_names, model, global_data.scaler, global_data.train_data, sample_data)
    rules = rule_extractor(exps[pred].names())
    local_shap_values= local_shap(model, global_data.scaler, sample_data)
    mi_shap = best_atributes_shap(local_shap_values, global_data.test_data, pred)
    print(mi_shap)
    mi_anchors = best_atributes_anchor(rules)
    selected_data = best_values(mi_shap, mi_anchors, sample_data, global_data.test_data, global_data.train_data, pred, global_data.train_target)
    print(selected_data.sort_values(by='distance').head(3))
    return preds


#anchor model
def predictor_creator(scaler, model, class_index):
    def predictor(samples):
        samples = scaler.transform(samples)
        pred = model.predict(samples)
        binary_pred = [1 if p == class_index else 0 for p in pred]

        return np.array(binary_pred,dtype='int64')
    return predictor

def local_anchor(class_names, model, scaler, train_data, sample):
    binary_predictors = {}
    pred = model.predict(scaler.transform([sample]))[0]
    for i,class_name in enumerate(class_names):
        binary_predictors[class_name] = predictor_creator(scaler, model, i)

    explainers = {}
    for i, class_name in enumerate(class_names):
        bin_classes_names = ["not_" + class_name, class_name]
        explainer = anchor_tabular.AnchorTabularExplainer(
                                        bin_classes_names,
                                        train_data.columns,
                                        train_data.values)
        explainers[class_name] = explainer
    exps = []
    for class_name in class_names:
        exp = explainers[class_name].explain_instance(
                sample.values,binary_predictors[class_name] , threshold=0.95)
        exps.append(exp)

    for i,exp in enumerate(exps):
        if (i == pred):
            print("--- classe predita ({0}) ---".format(class_names[pred]))
        else:
            print("--- {0} ---".format(class_names[i]))
        print('anchor: ',' '.join(exp.names()))
        print('precision: ',exp.precision())
        print('coverage: ',exp.coverage())

    return exps

def rule_extractor(names):
    rules = []
    for rule_name in names:
        r = rule_name.split(' ')
        rules.append((r[0],r[2],r[1]))
    return rules

def best_atributes_shap(local_shap_values, test_data, pred):
    sv = pd.Series(np.abs(local_shap_values[pred]), index = test_data.columns)
    mi_shap = sv.sort_values(ascending=False).index[:3]
    return mi_shap

def best_atributes_anchor(rules):
    mi_anchors = np.array([r[0] for r in rules])
    if (len(mi_anchors) > 3):
        mi_anchors = mi_anchors[:3]
    return mi_anchors

def best_values(mi_shap, mi_anchors, sample, test_data, train_data, pred, train_target):
    if(len(mi_anchors) < 3):
    # Selecionando atributos com Shap alto diferentes dos atributos do Anchors
        mi_shap_ = [feat for feat in mi_shap if feat not in mi_anchors]
        # Completando a lista de atributos
        mi_anchors = np.append([mi_anchors], [list(mi_shap_[:3-len(mi_anchors)])])
    nomes_classes = ['saudável','leve','moderada','severa']
    small_sample = sample[mi_anchors].copy()
    sample_class = nomes_classes[pred]
    selected_data = train_data[mi_anchors].copy() 
    selected_data['target'] = train_target
    selected_data['distance'] = np.sqrt(np.sum([(selected_data[col]- small_sample[col])**2 for  col in mi_anchors],axis=0))
    return selected_data

    