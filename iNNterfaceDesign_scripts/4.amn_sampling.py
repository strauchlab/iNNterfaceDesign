import sys, os, numpy as np
from modules import functions as f, Amino_acid_dictionaries as ad
from tensorflow.keras.models import load_model
import tensorflow as tf


flat = lambda x: [i for j in x for i in j]
rev_ad = {ad.amn_mapping[i]:i for i in ad.amn_mapping}

def get_path():
    global ipath
    global dir_models
    path = [a for a in sys.path if 'iNNterfaceDesign' in a]
    if path == []:
        print("Path to iNNterfaceDesign does not exist")
        sys.exit()
    else:
        ipath = path[0]
        dir_models = ipath + '/modules/models/'


def arr_ranges(data):
    data = [f.make_ranges(data[k], 50) for k in range(len(data))]
    data = [[a[k] for a in data] for k in range(len(data[0]))]
    return data

def data_for_aas_func(k):
    inp3 = np.zeros((k, 2))
    inp3[:, 0] = 1
    return [inp3]

def make_pssm(y_pred):
    data_new = []
    for i in range(len(y_pred[:])):
        pred = np.argsort(y_pred[i], axis=1)
        pssm = [{ad.amn_mapping1letter[rev_ad[m + 1]]: round(float(y_pred[i][lx, m]),3) for mx, m in enumerate(l)} for lx, l in enumerate(pred)]
        if {} in pssm:
            e = [k for k in range(len(pssm)) if pssm[k] =={}]
            for ex in e:
                print(y_pred[i][ex])
        data_new.append(pssm)
    return data_new

def aas_prediction_func(data, model_name, keywords):
    model = load_model(dir_models + model_name + '.hdf5', custom_objects={'tf': tf, 'K': tf.keras.backend})
    test_ch = arr_ranges(data)
    y_pred = [model.predict(i, verbose=0) for i in test_ch]
    y_pred = np.vstack(y_pred)
    #print(y_pred.shape)
    pred = np.argmax(y_pred, axis=2)
    pred_prob = np.max(y_pred, axis=2)
    
    pred_seq = [[[ad.amn_mapping1letter[rev_ad[j + 1]], pred_prob[ix][jx], jx] for jx, j in enumerate(i)] for ix, i in enumerate(pred)]
    if model_name == 'PepSep6':
        pred_seq = [[a[k:k+6] for k in range(6)] for a in pred_seq]
    else:
        pred_seq = [[a] for a in pred_seq]
    if ('amn_prob_distr' in keywords) and (keywords['amn_prob_distr'] == 'True'):
        pssm = make_pssm(y_pred)
        if model_name == 'PepSep6':
            pssm =  [[a[k:k+6] for k in range(6)] for a in pssm]
        return [pred_seq, pssm]
    else:
        return [pred_seq]

def merge_designs(res, model_name):
    def make_aa(designs):
        pos_e = max([a[3] for a in designs])
        aa = []
        for x in range(pos_e+1):
            aa_x = [i[:3] for i in designs if i[3] == x]
            if aa_x == []:
                continue
            aa_x.sort(key=lambda x: x[1])
            if (aa_x[-1][0] != 'G') or (len(aa_x) == 1):
                aa.append([aa_x[-1][0], aa_x[-1][2]])
            else:
                aa.append([aa_x[-2][0], aa_x[-2][2]])
        return aa

    
    names = np.unique([x[0] for x in res])
    res_new = []
    #print(res[0])
    for name in names:
        pssm = []
        res_name = [a for a in res if a[0] == name]
        if model_name == 'PepSep1':
            designs = flat([[[x[0], x[1], [a[1], x[2]], x[2] + a[1]] for x in a[2][0]] for a in res_name])
            aa1 = make_aa(designs)
            aa = ''.join([a[0] for a in aa1])
            if len(res[0]) == 4:
                for p in aa1:
                    pssm.append([x[3][p[1][1]] for x in res_name if x[1] == p[1][0]][0])
        elif model_name == 'PepSep6':
            aa = []
            for k in range(6):
                pssm_k = []
                designs = flat([[[x[0], x[1], [a[1], x[2]], x[2] + a[1]-k] for x in a[2][k]] for a in res_name])
                aa_k1 = make_aa(designs)
                aa_k = ''.join([a[0] for a in aa_k1])
                aa.append(aa_k)
                
                if len(res[0]) == 4:
                    for p in aa_k1:                        
                        pssm_k.append([x[3][k][p[1][1]-k] for x in res_name if x[1] == p[1][0]][0])
                    pssm.append(pssm_k)
                
        if pssm != []:
            res_new.append([name, aa, pssm])
        else:
            res_new.append([name, aa])
    return res_new

def get_aas_designs(data, model, keywords):
    res = []
    data = f.readjson(data)
    data = [np.array(i)[:] for i in data]
    p_geom, p_amn, p_seq, shape, interf_type, name, p_f = data
    data_for_amns_n = data_for_aas_func(len(name))
    designs = aas_prediction_func([p_geom, p_amn,p_seq,  shape, interf_type] + data_for_amns_n, model, keywords)
    #print(len(designs))
    idx1 = np.where(p_f == -1)[0]
    idx_int = np.where(p_f != -1)[0]
    if len(idx1) != 0:
        res1 = [[name[ix]] + [designs[x][ix] for x in range(len(designs))] for ix in range(len(name)) if ix in idx1]
        #print(res1[0])
        res1 = [[a[0], [''.join([aa[0] for aa in x]) for x in a[1]]] + a[2:] for a in res1[:]]
        res += res1
    if len(idx_int) != 0:
        res_int = [[name[ix], int(p_f[ix])] + [designs[x][ix] for x in range(len(designs))] for ix in range(len(name)) if ix in idx_int]
        fragms = np.unique([a[0] for a in res_int])
        for fx in fragms:
            res_fx = [a for a in res_int if a[0] == fx]
            res_fx = merge_designs(res_fx, model)
            res += res_fx
    return res

def filter_designs(designs, prefix):
    designs_n = []
    #print(designs[:])
    for seq in designs[:]:
        if os.path.exists(prefix + '/binders_id/' + seq[0][:-6] + '_id.pdb'):
            seq_k = []
            for a in seq[1]:
                a_k = [x for x in a]
                a3 = [len(np.unique(a_k[x:x+3])) for x in range(len(a)-2)]
                if len([x for x in a3 if x==1]) != 0:
                    continue
                d_n = len([x for x in a_k if x=='D'])
                y_n = len([x for x in a_k if x=='Y'])
                g_n = len([x for x in a_k if x=='G'])
                p_n = len([x for x in a_k if x=='P'])
                e_n = len([x for x in a_k if x=='E'])


                #if (len(np.unique(a_k)) >= 3) and (d_n < 1+ len(a)/2) and (y_n < 1 + len(a)/2):
                if (len(np.unique(a_k)) >= 3) and (d_n <= 3) and (y_n <= 4) and (p_n <=2) and (g_n <= 3) and (d_n+y_n <=6) and (d_n + e_n <=4): 
                    seq_k.append(a)
            #print(seq_k)
            if seq_k != []:
                designs_n.append([seq[0], seq_k])
    return designs_n

get_path()
input_file = sys.argv[1]
keywords = f.keywords(input_file)
prefix = keywords['prefix']
model = keywords['amn_design']
data = prefix + '/' + prefix  + '_data_aas.json'

designs = get_aas_designs(data, model, keywords)
if ('filter_designs' in keywords) and (keywords['filter_designs'] == 'True'):
    designs = filter_designs(designs, prefix)
#print(designs)
f.writejson(prefix + '/' + prefix + '_aas.json', designs)
