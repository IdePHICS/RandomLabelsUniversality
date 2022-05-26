import os
import numpy as np
import pandas as pd
import csv


def statistics_learning_curve(n_seeds = 10, path_to_results_directory = './results'):

    theory = {'N': [], 'alpha': [], 'data_type': [], 'which_dataset': [],
              'which_transform': [], 'non_lin': [], 'which_cov': [], 'which_mean': [],
              'loss': [], 'lamb': [], 'train_loss':[]}
    print(path_to_results_directory)
    for s in range(n_seeds):
        df = pd.read_csv(path_to_results_directory + "/seeds/" + resfile + "_seed_%d.csv"%(s))
        theory['N'].append(df['N'])
        theory['alpha'].append(df['alpha'])
        theory['data_type'].append(df['data_type'])
        theory['which_dataset'].append(df['which_dataset'])
        theory['which_transform'].append(df['which_transform'])
        theory['non_lin'].append(df['non_lin'])
        theory['which_cov'].append(df['which_cov'])
        theory['which_mean'].append(df['which_mean'])
        theory['loss'].append(df['loss'])
        theory['lamb'].append(df['lamb'])
        theory['train_loss'].append(df['train_loss'])

    theory_tmp = theory.copy()
    theory_tmp =  pd.DataFrame.from_dict(theory_tmp)
    theory_tmp.data_type = theory_tmp.data_type.apply(str)
    theory_tmp.which_dataset = theory_tmp.which_dataset.apply(str)
    theory_tmp.which_transform = theory_tmp.which_transform.apply(str)
    theory_tmp.non_lin = theory_tmp.non_lin.apply(str)
    theory_tmp.which_cov = theory_tmp.which_cov.apply(str)
    theory_tmp.which_mean = theory_tmp.which_mean.apply(str)
    theory_tmp.lamb = theory_tmp.lamb.apply(str)
    theory_stat = theory_tmp.groupby(['data_type', 'which_dataset', 'which_transform', 'non_lin', 'which_cov', 'which_mean', 'lamb']).aggregate(mean_train_loss = ('train_loss', lambda x: np.vstack(x).mean(axis=0).tolist()),
                                                std_train_loss = ('train_loss', lambda x: np.vstack(x).std(axis=0).tolist())).reset_index()

    if os.path.isfile(path_to_results_directory + "/seeds/" + resfile + ".csv") == False:
        with open(path_to_results_directory + "/seeds/" + resfile + ".csv", mode='w') as f:
            f.write("n_seeds,N,alpha,data_type,which_dataset,which_transform,non_lin,which_cov,which_mean,loss,lamb,mean_train_loss,std_train_loss\n")

    with open(path_to_results_directory + "/seeds/" + resfile + ".csv", mode='a') as f:
        wr = csv.writer(f, dialect='excel')
        for i in range(len(theory_stat.index)):
            j = 0
            n_seeds = len(theory['alpha'])
            for alpha in theory['alpha'][0]:
                wr.writerow([n_seeds, theory['N'][i][j], alpha, theory['data_type'][i][j],
                theory['which_dataset'][i][j], theory['which_transform'][i][j], theory['non_lin'][i][j],
                theory['which_cov'][i][j], theory['which_mean'][i][j], theory['loss'][i][j], theory['lamb'][i][j],
                theory_stat['mean_train_loss'][i][j], theory_stat['std_train_loss'][i][j]/np.sqrt(n_seeds)])

                j += 1

    return
