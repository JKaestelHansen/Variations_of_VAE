# %%
import torch
import math
import pickle
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from helper_functions import *
from models import *
from data_handler import *
from training import *


def get_elbos(model, wt, mutants, ensemble_count, labels):
    with torch.no_grad():
        acc_m_elbo = 0
        acc_wt_elbo = 0

        batch = torch.cat([wt.unsqueeze(0), mutants])

        for i in range(ensemble_count):
            elbos, *_ = model.protein_logp(batch, labels)
            wt_elbo = elbos[0]
            m_elbo = elbos[1:]
            acc_m_elbo += m_elbo
            acc_wt_elbo += wt_elbo

        mutants_logp = acc_m_elbo / ensemble_count
        wt_logp = acc_wt_elbo / ensemble_count

    return mutants_logp, wt_logp


def pipeline_mut_effect_pred(model, mutants, all_data, device, ensemble_count = 500):
    model.eval()
    mutants = torch.stack([torch.tensor(seq, device=device) for seq in mutants]).long()
    wt = torch.tensor(all_data['seqs'][0], device=device).long()
    mutants_logp, wt_logp = get_elbos(model, wt, mutants, ensemble_count, [None])
    deltaELBO = mutants_logp - wt_logp
    return deltaELBO


def onehot_(arr):
    return F.one_hot(torch.stack([torch.tensor(seq, device='cpu').long() for seq in arr]), num_classes=max(IUPAC_SEQ2IDX.values())+1).float().flatten(1)


# Device
device = 'cpu'

# determine VAE
latent_dim = 30
names = ['blat', 'calm', 'mth3', 'brca', 'timb', 'ubqt']
names = ['toxi']
DE_dict = {}
for name in names:
    if name == 'blat':
        name = name.upper()
    
    df = pickle.load( open('files/'+name+'_data_df.pkl', "rb" ) )
    query_seqs = df['seqs'][0]
    assay_df = df.dropna(subset=['assay']).reset_index(drop=True)

    random_weighted_sampling = True
    use_sparse_interactions = True
    use_bayesian = True
    use_param_loss = True
    labels = [None]

    assay_index = df.index
    train_idx = [i for i in df.index if i not in assay_index.values]
    all_data, train_data, val_data = get_datasets(data=df,
                                                train_ratio=1,
                                                device = device,
                                                SSVAE=0,
                                                SSCVAE=0,
                                                CVAE=0,
                                                regCVAE=0,
                                                assays_to_include=['assay'],
                                                train_index=train_idx,
                                                train_with_assay_seq=0,
                                                only_assay_seqs=0)

    # prep downstream data
    data_size = all_data[0][0].size(-1) * alphabet_size
    label_pred_layer_sizes = [0]
    z2yalpha = 0.01 * len(all_data)
    seq2yalpha = 0.01 * len(all_data)
    model = VAE_bayes(
        [data_size] + [1500, 1500, latent_dim, 100, 2000] + [data_size],
        alphabet_size,
        z_samples = 1,
        dropout = 0,
        use_bayesian = use_bayesian,
        use_param_loss = use_param_loss,
        use_sparse_interactions = use_sparse_interactions,
        rws=random_weighted_sampling,
        conditional_data_dim=0,
        SSVAE = 0,
        SSCVAE = 0,
        CVAE=0,
        VAE=1,
        regCVAE=0,
        multilabel=0,
        seq2yalpha=seq2yalpha,
        z2yalpha=z2yalpha,
        label_pred_layer_sizes = label_pred_layer_sizes,
        pred_from_latent=0,
        pred_from_seq=0,
        warm_up = 0,
        batchnorm=0,
        device = device
    )

    if name.lower()=='blat':
        path_torch = 'trained_models/D2021610T154BLAT_30dim_bestMSE_VAE.torch'
    if name.lower()=='brca':
        path_torch = 'trained_models/D2021610T154brca_30dim_bestMSE_VAE.torch'
    if name.lower()=='mth3':
        path_torch = 'trained_models/D2021610T154mth3_30dim_bestMSE_VAE.torch'
    if name.lower()=='timb':
        path_torch = 'trained_models/D2021610T154timb_30dim_bestMSE_VAE.torch'
    if name.lower()=='calm':
        path_torch = 'trained_models/D2021610T155calm_30dim_bestMSE_VAE.torch'
    if name.lower()=='ubqt':
        path_torch = 'trained_models/D2021729T173ubqt_30dim_bestMSE_VAE.torch'
    if name.lower()=='toxi':
        path_torch = 'results/D2022526T1232toxi_30dim_bestMSE_VAE.torch'

    model.load_state_dict(torch.load(path_torch, map_location=torch.device(device))['model_state_dict'])

    DE = pipeline_mut_effect_pred(model, assay_df['seqs'].values, assay_df, device, ensemble_count = 500)
    DE_dict[name] = DE
    from scipy.stats import spearmanr
    print(spearmanr(DE.numpy(), assay_df['assay']))

import pickle
pickle.dump(DE_dict, open('results/'+name+'_DE_dict.pkl', 'wb'))
