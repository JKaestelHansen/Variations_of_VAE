import argparse
from pathlib import Path

from datetime import datetime
import pickle
import math

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,pearsonr
from Bio import SeqIO

from models import VAE_bayes
from data_handler import alphabet_size, IUPAC_SEQ2IDX, IUPAC_IDX2SEQ, seq2idx, idx2seq, seq2idx_removegaps, add_unknown_filler_labels


def make_mutants(protein_data_path, mutation_data_path, sheet, metric_column, device, backbone_idx=0):
    is_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    print("Making mutants...")
    if sheet == 'BLAT_ECOLX_Ranganathan2015':
        print('BLAT')
        # load mutation and experimental pickle
        proteins = pd.read_excel(mutation_data_path, sheet_name=sheet)
        p = proteins.dropna(subset=['mutation_effect_prediction_vae_ensemble']).reset_index(drop=True)
        # load first value
        wt_seq = next(SeqIO.parse(protein_data_path, "fasta"))
        # get indices for every position 0....262
        wt_indices = np.array([i for i, c in enumerate(str(wt_seq.seq))])
        # convert aa to integer and tensor
        wt = seq2idx(wt_seq, device)
        # get offset from seq id
        offset = int(wt_seq.id.split("/")[1].split("-")[0])
        # position is wt_indives + registred offset 24...286
        positions = wt_indices + offset
        # translate mutation position into actual position in string
        positions_dict = {pos: i for i, pos in enumerate(positions)}
        # zip mutation with its score, filter nans using filter(filterfunc, sequence)
        # create a list from this filtration and unpack all - zip this which enables the equal sign more at geeks4geeks zip()
        mutants_list, scores = zip(*list(filter(lambda t: not math.isnan(t[1]), zip(p.mutant, p[metric_column]))))
        # convert from tuple to list
        mutants_list, scores = list(mutants_list), list(scores)
        wt_present = mutants_list[-1].lower() == "wt"
        if wt_present:
            del mutants_list[-1]
            del scores[-1]
        data_size = len(mutants_list)
        # create tensor of row size data_size and col size 1 filled with the wt backbone
        mutants = wt.repeat(data_size, 1)

        for i, position_mutations in enumerate(mutants_list):

            mutations = position_mutations.split(":")
            for mutation in mutations:
                # wt aa and mutant aa for given position
                wildtype = IUPAC_SEQ2IDX[mutation[0]]
                mutant = IUPAC_SEQ2IDX[mutation[-1]]

                # handle special offset case
                if sheet == "parEparD_Laub2015_all":
                    offset = 103
                else:
                    offset = 0
                # get location of mutation
                location = positions_dict[int(mutation[1:-1]) + offset]


                assert mutants[i, location] == wildtype, f"{IUPAC_IDX2SEQ[mutants[i, location].item()]}, {IUPAC_IDX2SEQ[wildtype]}, {location}, {i}"
                mutants[i, location] = mutant

    if 'Supernatant' in sheet:
        print('Lipase')
        print(mutation_data_path)
        proteins = pickle.load( open( mutation_data_path, 'rb' ) )
        p = proteins.dropna(subset=['Detected Mutations', sheet]).reset_index(drop=True)

        wt_seq = next(x for i,x in enumerate(SeqIO.parse(protein_data_path, "fasta")) if i==backbone_idx)
        wt = seq2idx_removegaps(wt_seq, device)
        mutants_list, scores = zip(*list(filter(lambda t: not math.isnan(t[1]), zip(p['Detected Mutations'], p[sheet]))))
        mutants_list, scores = list(mutants_list), list(scores)
        data_size = len(mutants_list)
        # create tensor of row size data_size and col size 1 filled with the wt backbone
        mutants = wt.repeat(data_size, 1)
        keep_idx = []
        discard_idx = []
        for i, position_mutations in enumerate(mutants_list):
            counter=0
            s = ''.join( c for c in position_mutations if  c not in '{}' )
            site_mutations = np.array([x.strip() for x in s.split(',')])
            _, idx = np.unique(site_mutations, return_index=True)
            site_mutations = site_mutations[np.sort(idx)]
            for mutation in site_mutations:
                try:
                    wildtype_aa = IUPAC_SEQ2IDX[mutation[0]]
                    mutant_aa = IUPAC_SEQ2IDX[mutation[-1]]
                    mut_location = int(mutation[1:-1])-1
                    assert scores[i] != np.nan
                    assert mut_location in list(range(data_size)), f"{site_mutations},{i}"
                    assert mutation[0] in is_aa, f"{site_mutations},{i}"
                    assert mutation[-1] in is_aa, f"{site_mutations},{i}"
                    assert mutants[i, mut_location] == wildtype_aa, f"{site_mutations}, {IUPAC_IDX2SEQ[mutants[i, mut_location].item()]}, {IUPAC_IDX2SEQ[wildtype_aa]}, {mut_location}, {i}"
                except (KeyError, AssertionError):
                    discard_idx.append(i)
                    break
                mutants[i, mut_location] = mutant_aa
            if counter == 0:
                if i not in discard_idx:
                    counter+=1
                    keep_idx.append(i)
        print('number of kept seqs', len(keep_idx))

        mutants = mutants[keep_idx]
        scores = np.array(scores)[keep_idx]
        assert np.sum(np.sum((mutants==wt).cpu().detach().numpy(),1)<269)==mutants.size()[0]


    while True:
        yield mutants, wt, scores

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


def mutation_effect_prediction(model, protein_data_path, mutation_data_path, sheet, metric_column, device, ensemble_count = 500, return_scores = False, return_logps = False, conditional_data_dim = None):
    model.eval()
    mutants_fn = pickle.load( open( 'data_handler/files/assay_data/BLAT_mutants_and_scores.pkl', "rb" ) )
    mutants, scores = torch.tensor(mutants_fn['seqs'], device=device).long(), mutants_fn['assay']
    wt = torch.tensor(pickle.load( open( 'data_handler/files/assay_data/BLAT_data_df.pkl', "rb" ) ).loc[0]['seqs'], device=device).long()


    if return_scores:
        return scores

    if conditional_data_dim > 0:
        labels = add_unknown_filler_labels(conditional_data_dim, len(mutants))
        labels = torch.tensor(labels, device=device).to(torch.float)
    else:
        labels = [None]

    mutants_logp, wt_logp = get_elbos(model, wt, mutants, ensemble_count, labels)

    if return_logps:
        return mutants_logp, wt_logp

    predictions = mutants_logp - wt_logp
    spearman_corr, _ = spearmanr(scores, predictions.cpu())
    pearson_cor, _ = pearsonr(scores, predictions.cpu())
    return spearman_corr, pearson_cor, predictions


def mutation_effect_prediction_lipase(model, protein_data_path, mutant_df, sheet, device, ensemble_count = 500,  conditional_data_dim = None, backbone_idx=0):
    model.eval()

    mutants = torch.stack([torch.tensor(seq, device=device) for seq in mutant_df['Sequence']])
    scores = mutant_df[sheet]

    if conditional_data_dim > 0:
        labels = add_unknown_filler_labels(conditional_data_dim, len(mutants))
        labels = torch.tensor(labels, device=device).to(torch.float)
    else:
        labels = [None]

    wt_seq = next(x for i,x in enumerate(SeqIO.parse(protein_data_path, "fasta")) if i==backbone_idx)
    wt = seq2idx_removegaps(wt_seq, device)

    mutants_logp, wt_logp = get_elbos(model, wt, mutants, ensemble_count, labels)

    predictions = mutants_logp - wt_logp
    spearman_corr, _ = spearmanr(scores, predictions.cpu())
    pearson_cor, _ = pearsonr(scores, predictions.cpu())
    return spearman_corr, pearson_cor, predictions

def mutation_effect_prediction_PDE(model, protein_data_path, mutant_df, sheet, device, ensemble_count = 500,  conditional_data_dim = None, backbone_idx=0):
    model.eval()

    mutants = torch.stack([torch.tensor(seq, device=device) for seq in mutant_df['Sequence']])
    scores = mutant_df[sheet]

    wt_seq = next(x for i,x in enumerate(SeqIO.parse(protein_data_path, "fasta")) if i==backbone_idx)
    wt = seq2idx_removegaps(wt_seq, device)[25:]

    # add labels
    # DECIDE whether to give actual labels or filler_labels
    if conditional_data_dim > 0:
        labels = add_unknown_filler_labels(conditional_data_dim, len(mutants))
        labels = torch.tensor(labels, device=device).to(torch.float)
    else:
        labels = [None]

    mutants_logp, wt_logp = get_elbos(model, wt, mutants, ensemble_count, labels)

    predictions = mutants_logp - wt_logp
    spearman_corr, _ = spearmanr(scores, predictions.cpu())
    pearson_cor, _ = pearsonr(scores, predictions.cpu())
    return spearman_corr, pearson_cor, predictions


def pipeline_mut_effect_pred(model, mutant_df, all_data_df, sheet, device, ensemble_count = 500, conditional_data_dim=0, backbone_idx=0):
    model.eval()
    mutants, scores = torch.stack([torch.tensor(seq, device=device) for seq in mutant_df['seqs']]).long(), mutant_df[sheet].values
    wt = torch.tensor(all_data_df['seqs'][backbone_idx], device=device).long()

    if conditional_data_dim > 0:
        labels = add_unknown_filler_labels(conditional_data_dim, len(mutants))
        labels = torch.tensor(labels, device=device).to(torch.float)
    else:
        labels = [None]

    mutants_logp, wt_logp = get_elbos(model, wt, mutants, ensemble_count, labels)
    predictions = mutants_logp - wt_logp
    spearman_corr, _ = spearmanr(scores, predictions.cpu())
    pearson_cor, _ = pearsonr(scores, predictions.cpu())
    return spearman_corr, pearson_cor, predictions


def ensemble_deltaELBOS(predictions):
    return np.mean(predictions, axis=1)
