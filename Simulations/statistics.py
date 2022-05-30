import os
import numpy as np
import pandas as pd
import csv


def statistics_learning_curve_real(n_seeds = 10, loss = 'square_loss', path_to_res_folder = './results'):

    res = {'p': [], 'alpha': [], 'which_real_dataset': [],
              'which_transform': [], 'which_non_lin': [],
              'loss': [], 'lamb': [], 'train_loss':[]}

    for s in range(n_seeds):
        df = pd.read_csv(path_to_res_folder + "/seeds/sim_%s_seed_%d_real.csv"%(loss,s))
        res['alpha'].append(df['alpha'])
        res['train_loss'].append(df['train_loss'])
        res['p'].append(df['p'])
        res['which_real_dataset'].append(df['which_real_dataset'])
        res['which_transform'].append(df['which_transform'])
        res['which_non_lin'].append(df['which_non_lin'])
        res['loss'].append(df['loss'])
        res['lamb'].append(df['lamb'])


    res_tmp = res.copy()
    res_tmp =  pd.DataFrame.from_dict(res_tmp)
    res_tmp.which_real_dataset = res_tmp.which_real_dataset.apply(str)
    res_tmp.which_transform = res_tmp.which_transform.apply(str)
    res_tmp.which_non_lin = res_tmp.which_non_lin.apply(str)
    res_tmp.lamb = res_tmp.lamb.apply(str)
    res_stat = res_tmp.groupby(['which_real_dataset', 'which_transform', 'which_non_lin', 'lamb']).aggregate(mean_train_loss = ('train_loss', lambda x: np.vstack(x).mean(axis=0).tolist()),
                                                std_train_loss = ('train_loss', lambda x: np.vstack(x).std(axis=0).tolist())).reset_index()

    if os.path.isfile(path_to_res_folder + "/sim_%s_real.csv"%(loss)) == False:
        with open(path_to_res_folder + "/sim_%s_real.csv"%(loss), mode='w') as f:
            f.write("alpha,mean_train_loss,std_train_loss,p,n_seeds,which_real_dataset,which_transform,which_which_non_lin,loss,lamb\n")

    with open(path_to_res_folder + "/sim_%s_real.csv"%(loss), mode='a') as f:
        wr = csv.writer(f, dialect='excel')
        for i in range(len(res_stat.index)):
            j = 0
            n_seeds = len(res['alpha'])
            for alpha in res['alpha'][0]:
                wr.writerow([alpha, res_stat['mean_train_loss'][i][j], res_stat['std_train_loss'][i][j]/np.sqrt(n_seeds),
                n_seeds, res['p'][i][j], res['which_real_dataset'][i][j], res['which_transform'][i][j], res['which_non_lin'][i][j],
                res['loss'][i][j], res['lamb'][i][j]])
                j += 1

    return


def statistics_learning_curve_synthetic(n_seeds = 10, loss = 'square_loss', which_synthetic_dataset = "single_gaussian", path_to_res_folder = './results'):

    res = {'p': [], 'alpha': [], 'which_synthetic_dataset': [],
              'cov_identifier': [], 'mean_identifier': [],
              'loss': [], 'lamb': [], 'train_loss':[]}

    for s in range(n_seeds):
        df = pd.read_csv(path_to_res_folder + "/seeds/sim_%s_%s_seed_%d.csv"%(loss,which_synthetic_dataset,s))
        res['p'].append(df['p'])
        res['alpha'].append(df['alpha'])
        res['which_synthetic_dataset'].append(df['which_synthetic_dataset'])
        res['cov_identifier'].append(df['cov_identifier'])
        res['mean_identifier'].append(df['mean_identifier'])
        res['loss'].append(df['loss'])
        res['lamb'].append(df['lamb'])
        res['train_loss'].append(df['train_loss'])

    res_tmp = res.copy()
    res_tmp =  pd.DataFrame.from_dict(res_tmp)
    res_tmp.which_synthetic_dataset = res_tmp.which_synthetic_dataset.apply(str)
    res_tmp.cov_identifier = res_tmp.cov_identifier.apply(str)
    res_tmp.mean_identifier = res_tmp.mean_identifier.apply(str)
    res_tmp.lamb = res_tmp.lamb.apply(str)
    res_stat = res_tmp.groupby(['which_synthetic_dataset', 'cov_identifier', 'mean_identifier', 'lamb']).aggregate(mean_train_loss = ('train_loss', lambda x: np.vstack(x).mean(axis=0).tolist()),
                                                std_train_loss = ('train_loss', lambda x: np.vstack(x).std(axis=0).tolist())).reset_index()

    if os.path.isfile(path_to_res_folder + "/sim_%s_%s.csv"%(loss,which_synthetic_dataset)) == False:
        with open(path_to_res_folder + "/sim_%s_%s.csv"%(loss,which_synthetic_dataset), mode='w') as f:
            f.write("alpha,mean_train_loss,std_train_loss,n_seeds,p,which_synthetic_dataset,cov_identifier,mean_identifier,loss,lamb\n")

    with open(path_to_res_folder + "/sim_%s_%s.csv"%(loss,which_synthetic_dataset), mode='a') as f:
        wr = csv.writer(f, dialect='excel')
        for i in range(len(res_stat.index)):
            j = 0
            n_seeds = len(res['alpha'])
            for alpha in res['alpha'][0]:
                wr.writerow([alpha, res_stat['mean_train_loss'][i][j], res_stat['std_train_loss'][i][j]/np.sqrt(n_seeds), n_seeds, res['p'][i][j],
                res['which_synthetic_dataset'][i][j],
                res['cov_identifier'][i][j], res['mean_identifier'][i][j], res['loss'][i][j], res['lamb'][i][j]])

                j += 1

    return
