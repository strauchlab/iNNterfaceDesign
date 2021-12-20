import sys,numpy as np
from modules import functions as f, transform_coords as tc
from tensorflow.keras.models import load_model

sst_dict = {'H': 0, 'E':1, 'L': 2}
dir_models = '/home/raulia/binders_nn/modules/models/'
def arr_ranges(data):
    data = [f.make_ranges(data[k], 50) for k in range(len(data))]
    data = [[a[k] for a in data] for k in range(len(data[0]))]
    return data

def data_for_orient_func(k):
    inp3 = np.zeros((k, 2))
    inp3[:, 0] = 1
    return [inp3]

def data_for_binder_func(k, sstr):
    inp3 = np.zeros((k, 2))
    inp3[:, 0] = 1
    temp = np.zeros((k, 6, 3))
    for idx in range(k):
        for jdx in range(2, 4):
            temp[idx, jdx, int(sstr[idx, jdx-2])] = 1
    return [temp, inp3]

def orient_predictions_func(data, keywords, ld, ldd, model):
    test_ch = arr_ranges(data)
    y_pred = [model.predict(i, verbose=0) for i in test_ch]
    y_pred = np.vstack(y_pred)
    pred_order = np.fliplr(np.argsort(y_pred, axis=1))
    pred_order = [[[i,y_pred[jx, i]]  for i in j] for jx, j in enumerate(pred_order)]
    pred_order = [[[x[0] // 6, x[0] % 6, x[1]] for x in j] for j in pred_order]
    pos_thr = 0.01 if 'pos_thr' not in keywords else float(keywords['pos_thr'])
    pred_order = [[x for x in a if x[2] > pos_thr] for a in pred_order]
   
    if ld[0] is not None:
        pred_order = [[x for x in a if x[0] in ld[0]] for a in pred_order]
    if ld[1] is not None:
        pred_order = [[x for x in a if x[1] in ld[1]] for a in pred_order]
    #print(pred_order)
    if (ld[0] is None) and (ld[1] is None):
        #print(ld, ldd)
        if ldd[0] is not None:
            pred_order = [[x for x in a if x[0] not in ldd[0]] for a in pred_order]
        if ldd[1] is not None:
            pred_order = [[x for x in a if x[1] in ldd[1]] for a in pred_order]
    max_pos = 1 if 'max_pos' not in keywords else int(keywords['max_pos'])
    pred_order = [i[:max_pos] for i in pred_order]
    print(pred_order)
    return pred_order

def sstr_predictions_func(data, keywords, model):
    #print([i.shape for i in data])
    test_ch = arr_ranges(data)
    y_pred = [model.predict(i, verbose=0) for i in test_ch]
    y_pred = np.vstack(y_pred)
    pred_order = np.fliplr(np.argsort(y_pred, axis=1))
    pred_order = [[[i,y_pred[jx, i]]  for i in j if y_pred[jx, i]] for jx, j in enumerate(pred_order)]
    pred_order = [[[x[0] // 3, x[0] % 3, x[1]] for x in j] for j in pred_order]
    sst_thr = 0.01 if 'sst_thr' not in keywords else float(keywords['sst_thr'])
    pred_order = [[x for x in a if x[2] > sst_thr] for a in pred_order]
    if ('sst_type' in keywords):
        sst_type = [sst_dict[a.strip()] for a in keywords['sst_type'].split(',')]
        pred_order = [[x for x in a if len([j for j in x[:2] if j in sst_type]) == 2] for a in pred_order]
    #else
   
    max_sst = 1 if 'max_sst' not in keywords else int(keywords['max_sst'])
    pred_order = [i[:max_sst] for i in pred_order]
    return pred_order

def binders_predictions_func(data, o, s, model):
    o, s = [str(np.round(int(i[-1]*100))) for i in o], [str(np.round(int(i[-1]*100))) for i in s]
    test_ch = arr_ranges(data)
    y_pred = [model.predict(i, verbose=0) for i in test_ch]
    y_pred = np.vstack(y_pred)
    return [y_pred, o, s]

def make_binders(data, data2, keywords):
    model_pos = load_model(dir_models + 'Orn.hdf5')
    model_sst = load_model(dir_models + 'SecS.hdf5')
    model_bb = load_model(dir_models + 'PepBB.hdf5')
    swap = 'False'
    if ('swap_pose' in keywords) and (keywords['swap_pose'] == 'True'):
        swap = 'True'

    pdb_data_all = []
    binder_files = []
    data = f.readjson(data)
    data2 = f.readjson(data2)
    p_geom, p_sec, p_amn, p_mask, name = [np.array(data[x]) for x in ['p_xyz1', 'sec1', 'p_amn1', 'mask', 'name']]
    data_for_orient_n = data_for_orient_func(len(name))
    orients = orient_predictions_func([p_geom, p_sec, p_amn, p_mask] + data_for_orient_n, keywords, data2[0]['ld'], data2[0]['ldd'], model_pos)
    num_orient = max([len(i) for i in orients])
    for ornt in range(num_orient):
        idx = [idx for idx in range(len(orients)) if len(orients[idx]) > ornt]
        p_geom_idx, p_sec_idx, p_amn_idx, p_mask_idx, name_idx = [a[idx] for a in [p_geom, p_sec, p_amn, p_mask, name]]
        ornt_idx = np.array([o[ornt] for ox, o in enumerate(orients) if ox in idx])[:, :2]
        data_for_sstr_n = data_for_orient_func(len(name_idx))
        sstrs = sstr_predictions_func([p_geom_idx, p_sec_idx, p_amn_idx, p_mask_idx, ornt_idx] + data_for_sstr_n, keywords, model_sst)    
        num_sstr = max([len(i) for i in sstrs])
        for sstr in range(num_sstr):
            jdx = [idx for idx in range(len(sstrs)) if len(sstrs[idx]) > sstr]
            p_geom_jdx, p_sec_jdx, p_amn_jdx, p_mask_jdx, name_jdx = [a[jdx] for a in [p_geom_idx, p_sec_idx, p_amn_idx, p_mask_idx, name_idx]]
            p_ld_jdx = ornt_idx[jdx]
            sstr_jdx = np.array([s[sstr] for sx, s in enumerate(sstrs) if sx in jdx])[:, :2]
            data_for_binder_n = data_for_binder_func(len(name_jdx), sstr_jdx)
            binders_data = binders_predictions_func([p_geom_jdx, p_sec_jdx, p_amn_jdx, p_mask_jdx, p_ld_jdx] + data_for_binder_n, ornt_idx, sstr_jdx, model_bb)
            pdb_data = tc.calculate_rmsd(binders_data, data2, name_jdx, [str(ornt), str(sstr)], sstr_jdx.tolist(), keywords['prefix'], swap)
            pdb_data_all += pdb_data
            #print(pdb_data_all)
    return pdb_data_all

def make_binders_E(data, data2, keywords, e, c_names = None):
    nbem = 1
    swap = 'False'
    if ('swap_pose' in keywords) and (keywords['swap_pose'] == 'True'):
        swap = 'True'

    if ('num_pepbbe_m' in keywords) and (int(keywords['num_pepbbe_m']) in [1, 3]):
        nbem = int(keywords['num_pepbbe_m'])
    if e == 1:
        model_pos = load_model(dir_models + 'Orn_C.hdf5')
        model_sst = load_model(dir_models + 'SecS_C.hdf5')
        model_bb = [load_model(dir_models + 'PepBB_C' + str(k) + '.hdf5') for k in range(1,nbem+1)]


    elif (e == 0) or (e==10):
        model_pos = load_model(dir_models + 'Orn_N.hdf5')
        model_sst = load_model(dir_models + 'SecS_N.hdf5')
        model_bb = [load_model(dir_models + 'PepBB_N' + str(k) + '.hdf5') for k in range(1,nbem+1)]



    p_geom, p_sec, p_amn, p_mask, name, name_b, l_geom = data
    
    pdb_data_all = []
    binder_files = []

    data2 = f.readjson(data2)
    data_for_orient_n = data_for_orient_func(len(name))
    orients = orient_predictions_func([p_geom, p_sec, p_amn, p_mask] + data_for_orient_n + [l_geom], keywords, data2[0]['ld'], data2[0]['ldd'], model_pos)
    num_orient = max([len(i) for i in orients])
    for ornt in range(num_orient):
        idx = [idx for idx in range(len(orients)) if len(orients[idx]) > ornt]
        p_geom_idx, p_sec_idx, p_amn_idx, p_mask_idx, name_idx, name_b_idx, l_geom_idx = [a[idx] for a in [p_geom, p_sec, p_amn, p_mask, name, name_b, l_geom]]
        ornt_idx = np.array([o[ornt] for ox, o in enumerate(orients) if ox in idx])[:, :2]
        data_for_sstr_n = data_for_orient_func(len(name_idx))
        sstrs = sstr_predictions_func([p_geom_idx, p_sec_idx, p_amn_idx, p_mask_idx, ornt_idx] + data_for_sstr_n + [l_geom_idx], keywords, model_sst)    
        num_sstr = max([len(i) for i in sstrs]) 
        for sstr in range(num_sstr):
            for mm in range(len(model_bb)):
                jdx = [idx for idx in range(len(sstrs)) if len(sstrs[idx]) > sstr]
                p_geom_jdx, p_sec_jdx, p_amn_jdx, p_mask_jdx, name_jdx,  name_b_jdx, l_geom_jdx = [a[jdx] for a in [p_geom_idx, p_sec_idx, p_amn_idx, p_mask_idx, name_idx, name_b_idx, l_geom_idx]]
                p_ld_jdx = ornt_idx[jdx]
                sstr_jdx = np.array([s[sstr] for sx, s in enumerate(sstrs) if sx in jdx])[:, :2]
                data_for_binder_n = data_for_binder_func(len(name_jdx), sstr_jdx)
                binders_data = binders_predictions_func([p_geom_jdx, p_sec_jdx, p_amn_jdx, p_mask_jdx, p_ld_jdx] + data_for_binder_n + [l_geom_jdx[:,:,:,[0,1,2,4]]], ornt_idx, sstr_jdx, model_bb[mm])
                pdb_data = tc.calculate_rmsde(binders_data, data2, name_jdx, name_b_jdx, [str(ornt), str(sstr), str(mm)], sstr_jdx.tolist(), keywords['prefix'], e, swap, c_names)
                pdb_data_all += pdb_data
    return pdb_data_all

def make_binders_end(data, data2, keywords):
    def get_e_data(data, e):
        idx = np.where(np.array(data['b_end']) == e)[0]
        data = [np.array([a for ax, a in enumerate(data[x]) if ax in idx]) for x in data if x != 'b_end']
        return data

    data = f.readjson(data)
    pdb_names_all = []
    if 'binder_end' in keywords:
        if keywords['binder_end'] == 'C':
            data_C  = get_e_data(data, 1)
            pdb_names = make_binders_E(data_C, data2, keywords, 1)   
        elif keywords['binder_end'] == 'N':
            data_N  = get_e_data(data, 0)
            pdb_names = make_binders_E(data_N, data2, keywords, 0) 
    else:
        data_C  = get_e_data(data, 1)
        pdb_names = make_binders_E(data_C, data2, keywords, 1)         
        data_N  = get_e_data(data, 0)
        pdb_names = make_binders_E(data_N, data2, keywords, 10, pdb_names)   
    pdb_names_all += pdb_names
    return pdb_names

input_file = sys.argv[1]
keywords = f.keywords(input_file)
prefix = keywords['prefix']
data = prefix + '/' + prefix  + '_b.json'
data2 = prefix + '/' + prefix + '_2b.json'
if 'add_residues' in keywords:
    pdb_names = make_binders_end(data, data2, keywords)
else:
    pdb_names = make_binders(data, data2, keywords)
print(pdb_names)
f.writejson(prefix + '/' + prefix + '_names.json', pdb_names)
