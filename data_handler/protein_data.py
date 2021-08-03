from pathlib import Path
from collections import OrderedDict
from collections import defaultdict
import random
import multiprocessing
import threading
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader, WeightedRandomSampler
from Bio import SeqIO
from bioservices import UniProt
import pandas as pd
import re
IUPAC_IDX_AMINO_PAIRS_decoding = list(enumerate([
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "-",
    'B'
]))
IUPAC_IDX_AMINO_PAIRS = list(enumerate([
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "<mask>",
    'B'
]))
IUPAC_AMINO_IDX_PAIRS = [(a, i) for (i, a) in IUPAC_IDX_AMINO_PAIRS]

alphabet_size = len(IUPAC_AMINO_IDX_PAIRS)

IUPAC_SEQ2IDX = OrderedDict(IUPAC_AMINO_IDX_PAIRS)
IUPAC_IDX2SEQ = OrderedDict(IUPAC_IDX_AMINO_PAIRS)

# Add gap tokens as the same as mask
IUPAC_SEQ2IDX["-"] = IUPAC_SEQ2IDX["<mask>"]
IUPAC_SEQ2IDX["."] = IUPAC_SEQ2IDX["<mask>"]

IUPAC_IDX2SEQ_decoding = OrderedDict(IUPAC_IDX_AMINO_PAIRS_decoding)

def seq2idx(seq, device = None):
    return torch.tensor([IUPAC_SEQ2IDX[s.upper() if len(s) < 2 else s] for s in seq], device = device)
    # return torch.tensor([IUPAC_SEQ2IDX[s] for s in seq if len(s) > 1 or (s == s.upper() and s != ".")], device = device)

def seq2idx_removegaps(seq, device=None):
    seq = np.array([IUPAC_SEQ2IDX[aa] for aa in np.array(seq)])
    keep_cols = []
    for i, aa in enumerate(seq):
        if IUPAC_IDX2SEQ[aa] != '<mask>':
            keep_cols.append(i)
    seq = seq[keep_cols]
    return torch.tensor(seq, device=device)

def idx2seq(idxs):
    return "".join([IUPAC_IDX2SEQ[i] for i in idxs])

def idx2seq_decoding(idxs):
    return "".join([IUPAC_IDX2SEQ_decoding[i] for i in idxs])

def save_weights_file(datafile, save_file):
    seqs = list(SeqIO.parse(datafile, "fasta"))
    dataset = ProteinDataset(seqs)
    dataset.write_to_file(save_file)


class LipaseDataset(Dataset):
    def __init__(self, seqs, backbone=0, gappy_colx_threshold=1, device = None, SSVAE=False, SSCVAE=False, CVAE=False,
                 ssl_deg=False, ssl_iniact=False, ssl_pnp=False, ssl_glad=False, tom_odor=False, tom_perf=False,
                  ogt=False, topt=False, add_ssl_seqs=False, add_tom_seqs=False, only_tom_seqs=False,
                  tom_val_index=None, over_sample=0):
        super().__init__()
        self.device = device
        self.seqs = seqs if isinstance(seqs, list) else list(SeqIO.parse(seqs, "fasta"))
        if len(self.seqs) == 0:
            self.encoded_seqs = torch.Tensor()
            self.weights = torch.Tensor()
            self.neff = 0
            return


        prepro_seqs, kept_seqs = prepro_alignment(self.seqs, backbone, gappy_colx_threshold)
        self.encoded_seqs = torch.stack([torch.tensor(seq, device=device) for seq in prepro_seqs])
        self.encoded_seqs = postprocess_aln_coverage(self.encoded_seqs, threshold=0.5)

        discretize = True if SSCVAE or CVAE else False
        if ssl_deg or ssl_iniact or ssl_pnp or ssl_glad or add_ssl_seqs:
            ssl_seqs_and_labels = prep_ssl_data(ssl_deg=ssl_deg, ssl_iniact=ssl_iniact, ssl_pnp=ssl_pnp, ssl_glad=ssl_glad, discretize=discretize, ast_threshold=30, quantiles=[0, 0.25 ,0.75, 1])
            for col in ssl_seqs_and_labels.columns:
                if 'UNKNOWN' in col and SSCVAE:
                    ssl_seqs_and_labels = ssl_seqs_and_labels.drop(col, axis=1)
            if ssl_deg or ssl_iniact or ssl_pnp or ssl_glad:
                ssl_labels, ssl_filler_labels, _ = prep_assay_labels(ssl_seqs_and_labels, device, self.encoded_seqs.size(0), cvae=CVAE)
                ssl_labels = torch.cat((ssl_filler_labels, ssl_labels))
            else:
                ssl_labels = None
            ssl_seqs = torch.tensor(ssl_seqs_and_labels['Sequence'], device=device)
            self.encoded_seqs = torch.cat((self.encoded_seqs, ssl_seqs))
        else:
            ssl_labels = None


        if tom_odor or tom_perf or add_tom_seqs:
            tom_seqs_and_labels = prep_tom_labels(val_index=tom_val_index, tom_odor=tom_odor, tom_perf=tom_perf, discretize=discretize, quantiles=[0, 0.25 ,0.75, 1])
            for col in tom_seqs_and_labels.columns:
                if 'UNKNOWN' in col and SSCVAE:
                    tom_seqs_and_labels = tom_seqs_and_labels.drop(col, axis=1)
            if tom_odor or tom_perf:
                tom_labels, tom_filler_labels, _ = prep_assay_labels(tom_seqs_and_labels, device, self.encoded_seqs.size(0), cvae=CVAE)
                tom_labels = torch.cat((tom_filler_labels, tom_labels))
            else:
                tom_labels = None
            tom_seqs = torch.tensor(tom_seqs_and_labels['Sequence'], device=device)
            if not only_tom_seqs:
                self.encoded_seqs = torch.cat((self.encoded_seqs, tom_seqs))
            else:
                self.encoded_seqs = tom_seqs
        else:
            tom_labels = None

        if ssl_labels != None and tom_labels == None:
            self.labels = ssl_labels
        if ssl_labels == None and tom_labels != None:
            self.labels = tom_labels
        if ssl_labels != None and tom_labels != None:
            extra_filler_labels = fill_missing(tom_seqs.size(0), ssl_seqs_and_labels, discretize, device)
            ssl_labels = torch.cat((ssl_labels, extra_filler_labels))
            self.labels = torch.cat((ssl_labels, tom_labels), dim=1)
        if ssl_labels == None and tom_labels == None:
            self.labels = None
        if only_tom_seqs and tom_odor or tom_perf:
            self.labels = tom_labels

        if ogt or topt:
            if self.labels != None:
                ogt_topt_labels = match_ogt_n_topt(kept_seqs, ogt, topt, device)
                for col in ogt_topt_labels:
                    if 'UNKNOWN' in col:
                        if SSCVAE:
                            ogt_topt_labels = ogt_topt_labels.drop(col, axis=1)
                num_extra_seqs = self.labels.size(0)-len(ogt_topt_labels)
                filler_labels = fill_missing(num_extra_seqs, ogt_topt_labels, discretize, device)
                ogt_topt_labels = torch.cat((torch.tensor(ogt_topt_labels.values, device=device).float(), filler_labels))
                self.labels = torch.cat((self.labels, ogt_topt_labels), dim=1)
            if self.labels == None:
                ogt_topt_labels = match_ogt_n_topt(kept_seqs, ogt, topt, device)
                for col in ogt_topt_labels:
                    if 'UNKNOWN' in col:
                        if SSCVAE:
                            ogt_topt_labels = ogt_topt_labels.drop(col, axis=1)
                self.labels = torch.tensor(ogt_topt_labels.values, device=device).float()

        if not SSVAE and not CVAE and not SSCVAE:
            self.labels = None
        # Calculate weights
        weights = []
        flat_one_hot = F.one_hot(self.encoded_seqs, num_classes=max(IUPAC_SEQ2IDX.values())+1).float().flatten(1)
        weight_batch_size = 1000
        for i in range(self.encoded_seqs.size(0) // weight_batch_size + 1):
            x = flat_one_hot[i * weight_batch_size : (i + 1) * weight_batch_size]
            similarities = torch.mm(x, flat_one_hot.T)
            lengths = (self.encoded_seqs[i * weight_batch_size : (i + 1) * weight_batch_size] != IUPAC_SEQ2IDX["<mask>"]).sum(1).unsqueeze(-1)
            w = 1.0 / (similarities / lengths).gt(0.8).sum(1).float()
            weights.append(w)
        self.weights = torch.cat(weights)
        # TODO fix oversampling to be indifferent of label index
        self.neff = self.weights.sum()
        print(self.neff)
        if over_sample>0 and tom_odor:
            self.weights[-(self.labels.size(0)-tom_filler_labels.size(0)):] = self.weights[-(self.labels.size(0)-tom_filler_labels.size(0)):] + over_sample
        if over_sample>0 and not tom_odor:
            print('######################################### USED OVER_SAMPLE FOR LABEL THAT ISNT TOM #######################################')


    def write_to_file(self, filepath):
        for s, w in zip(self.seqs, self.weights):
            s.id = s.id + ':' + str(float(w))
        SeqIO.write(self.seqs, filepath, 'fasta')

    def __len__(self):
        return len(self.encoded_seqs)

    def __getitem__(self, i):
        if type(self.labels) == type(None):
            labels = self.labels
        else:
            labels = self.labels[i]
        return self.encoded_seqs[i], self.weights[i], self.neff, labels


class BLATDataset(Dataset):
    def __init__(self, seqs, device = None, SSVAE=False, SSCVAE=False, CVAE=False,
                 assay=False, add_assay_seqs=False, val_index=None):
        super().__init__()
        self.device = device
        self.seqs = seqs if isinstance(seqs, list) else list(SeqIO.parse(seqs, "fasta"))
        if len(self.seqs) == 0:
            self.encoded_seqs = torch.Tensor()
            self.weights = torch.Tensor()
            self.neff = 0
            return

        self.encoded_seqs = torch.stack([seq2idx(seq, device) for seq in self.seqs])
        num_sequences = self.encoded_seqs.size(0)

        if SSCVAE or CVAE or SSVAE:
            discretize = True if SSCVAE or CVAE else False
            if assay or add_assay_seqs:
                assay_seqs_and_labels = prep_any_labels('data_handler/files/assay_data/Blat_assay_data.pkl',
                                                        ['assay'],
                                                        val_index=val_index,
                                                        discretize=discretize)
                for col in assay_seqs_and_labels.columns:
                    if 'UNKNOWN' in col and SSCVAE:
                        assay_seqs_and_labels = assay_seqs_and_labels.drop(col, axis=1)
                if assay:
                    assay_labels, assay_filler_labels, _ = prep_assay_labels(assay_seqs_and_labels, device, self.encoded_seqs.size(0), cvae=CVAE)
                    assay_labels = torch.cat((assay_filler_labels, assay_labels))
                else:
                    assay_labels = None
                assay_seqs = torch.tensor(assay_seqs_and_labels['Sequence'], device=device)
                self.encoded_seqs = torch.cat((self.encoded_seqs, assay_seqs))
            else:
                assay_labels = None

            if assay_labels != None:
                self.labels = assay_labels
            if assay_labels == None:
                self.labels = None

        if not SSVAE and not CVAE and not SSCVAE:
            self.labels = None

        # Calculate weights
        weights = []
        flat_one_hot = F.one_hot(self.encoded_seqs, num_classes=max(IUPAC_SEQ2IDX.values())+1).float().flatten(1)
        weight_batch_size = 1000
        for i in range(self.encoded_seqs.size(0) // weight_batch_size + 1):
            x = flat_one_hot[i * weight_batch_size : (i + 1) * weight_batch_size]
            similarities = torch.mm(x, flat_one_hot.T)
            lengths = (self.encoded_seqs[i * weight_batch_size : (i + 1) * weight_batch_size] != IUPAC_SEQ2IDX["<mask>"]).sum(1).unsqueeze(-1)
            w = 1.0 / (similarities / lengths).gt(0.8).sum(1).float()
            weights.append(w)
        self.weights = torch.cat(weights)
        self.neff = self.weights.sum()

    def write_to_file(self, filepath):
        for s, w in zip(self.seqs, self.weights):
            s.id = s.id + ':' + str(float(w))
        SeqIO.write(self.seqs, filepath, 'fasta')

    def __len__(self):
        return len(self.encoded_seqs)

    def __getitem__(self, i):
        if self.labels == None:
            labels = self.labels
        else:
            labels = self.labels[i]
        return self.encoded_seqs[i], self.weights[i], self.neff, labels


class PDEDataset(Dataset):
    def __init__(self, seqs, backbone=0, gappy_colx_threshold=1, device = None, SSVAE=False, SSCVAE=False, CVAE=False,
                 logHIF=False, ogt=False, topt=False, add_logHIF_seqs=False, val_index=None):
        super().__init__()
        self.device = device
        self.seqs = seqs if isinstance(seqs, list) else list(SeqIO.parse(seqs, "fasta"))
        if len(self.seqs) == 0:
            self.encoded_seqs = torch.Tensor()
            self.weights = torch.Tensor()
            self.neff = 0
            return

        prepro_seqs, _ = prepro_alignment(self.seqs, backbone, gappy_colx_threshold)
        self.encoded_seqs = torch.stack([torch.tensor(seq, device=device) for seq in prepro_seqs])
        self.encoded_seqs = torch.stack([seq[25:] for seq in self.encoded_seqs])
        self.encoded_seqs = postprocess_aln_coverage(self.encoded_seqs, threshold=0.5)
        num_sequences = self.encoded_seqs.size(0)

        if SSCVAE or CVAE or SSVAE:
            discretize = True if SSCVAE or CVAE else False
            if logHIF or add_assay_seqs:
                assay_seqs_and_labels = prep_any_labels('data_handler/files/assay_data/PDE_logHIF_data.pkl',
                                                        ['logHIF'],
                                                        val_index=val_index,
                                                        discretize=discretize)
                for col in assay_seqs_and_labels.columns:
                    if 'UNKNOWN' in col and SSCVAE:
                        assay_seqs_and_labels = assay_seqs_and_labels.drop(col, axis=1)
                if assay:
                    assay_labels, assay_filler_labels, _ = prep_assay_labels(assay_seqs_and_labels, device, self.encoded_seqs.size(0), cvae=CVAE)
                    assay_labels = torch.cat((assay_filler_labels, assay_labels))
                else:
                    assay_labels = None
                assay_seqs = torch.tensor(assay_seqs_and_labels['Sequence'], device=device)
                self.encoded_seqs = torch.cat((self.encoded_seqs, assay_seqs))
            else:
                assay_labels = None

            if assay_labels != None:
                self.labels = assay_labels
            if assay_labels == None:
                self.labels = None

            if ogt or topt:
                if self.labels != None:
                    ogt_topt_labels = match_ogt_n_topt(kept_seqs, ogt, topt, device)
                    for col in ogt_topt_labels:
                        if 'UNKNOWN' in col:
                            if SSCVAE:
                                ogt_topt_labels = ogt_topt_labels.drop(col, axis=1)
                    num_extra_seqs = self.labels.size(0)-len(ogt_topt_labels)
                    filler_labels = fill_missing(num_extra_seqs, ogt_topt_labels, discretize, device)
                    ogt_topt_labels = torch.cat((torch.tensor(ogt_topt_labels.values, device=device).float(), filler_labels))
                    self.labels = torch.cat((self.labels, ogt_topt_labels), dim=1)
                if self.labels == None:
                    ogt_topt_labels = match_ogt_n_topt(kept_seqs, ogt, topt, device)
                    for col in ogt_topt_labels:
                        if 'UNKNOWN' in col:
                            if SSCVAE:
                                ogt_topt_labels = ogt_topt_labels.drop(col, axis=1)
                    self.labels = torch.tensor(ogt_topt_labels.values, device=device).float()

        if not SSVAE and not CVAE and not SSCVAE:
            self.labels = None
        # Calculate weights
        weights = []
        flat_one_hot = F.one_hot(self.encoded_seqs, num_classes=max(IUPAC_SEQ2IDX.values())+1).float().flatten(1)
        weight_batch_size = 1000
        for i in range(self.encoded_seqs.size(0) // weight_batch_size + 1):
            x = flat_one_hot[i * weight_batch_size : (i + 1) * weight_batch_size]
            similarities = torch.mm(x, flat_one_hot.T)
            lengths = (self.encoded_seqs[i * weight_batch_size : (i + 1) * weight_batch_size] != IUPAC_SEQ2IDX["<mask>"]).sum(1).unsqueeze(-1)
            w = 1.0 / (similarities / lengths).gt(0.8).sum(1).float()
            weights.append(w)
        self.weights = torch.cat(weights)
        self.neff = self.weights.sum()

    def write_to_file(self, filepath):
        for s, w in zip(self.seqs, self.weights):
            s.id = s.id + ':' + str(float(w))
        SeqIO.write(self.seqs, filepath, 'fasta')

    def __len__(self):
        return len(self.encoded_seqs)

    def __getitem__(self, i):
        if self.labels == None:
            labels = self.labels
        else:
            labels = self.labels[i]
        return self.encoded_seqs[i], self.weights[i], self.neff, labels


def get_datasets_from_Lipase(file=None, backbone_idx=0, train_ratio=1, gappy_colx_threshold=1, device = None,
                             SSVAE=False, SSCVAE=False, CVAE=False, ssl_deg=False, ssl_iniact=False,
                             ssl_pnp=False, ssl_glad=False, tom_odor=False, tom_perf=False,
                             ogt=False, topt=False, add_ssl_seqs=False, add_tom_seqs=False, only_tom_seqs=False,
                             tom_val_index=None, over_sample=False):

    seqs = list(SeqIO.parse(file, "fasta"))
    backbone = seqs[backbone_idx]

    data_len = len(seqs)
    seq_len = len(seqs[0])

    # Split into train/validation
    train_length = int(train_ratio * data_len)
    val_length = data_len - train_length

    indices = list(range(data_len))
    random.shuffle(indices)
    train_indices = indices[:train_length]
    val_indices = indices[train_length:]
    train_seqs = [seqs[i] for i in train_indices]
    val_seqs = [seqs[i] for i in val_indices]


    all_data = LipaseDataset(seqs, backbone=backbone, gappy_colx_threshold=gappy_colx_threshold, device = device, SSVAE=SSVAE, SSCVAE=SSCVAE, CVAE=CVAE,
                             ssl_deg=ssl_deg, ssl_iniact=ssl_iniact, ssl_pnp=ssl_pnp, ssl_glad=ssl_glad, tom_odor=tom_odor, tom_perf=tom_perf,
                             ogt=ogt, topt=topt, add_ssl_seqs=add_ssl_seqs, add_tom_seqs=add_tom_seqs, only_tom_seqs=only_tom_seqs,
                             tom_val_index=tom_val_index, over_sample=over_sample)

    train_data = LipaseDataset(train_seqs, backbone=backbone, gappy_colx_threshold=gappy_colx_threshold, device = device, SSVAE=SSVAE, SSCVAE=SSCVAE, CVAE=CVAE,
                             ssl_deg=ssl_deg, ssl_iniact=ssl_iniact, ssl_pnp=ssl_pnp, ssl_glad=ssl_glad, tom_odor=tom_odor, tom_perf=tom_perf,
                             ogt=ogt, topt=topt, add_ssl_seqs=add_ssl_seqs, add_tom_seqs=add_tom_seqs, only_tom_seqs=only_tom_seqs,
                             tom_val_index=tom_val_index, over_sample=over_sample)

    val_data = LipaseDataset(val_seqs, backbone=backbone, gappy_colx_threshold=gappy_colx_threshold, device = device, SSVAE=SSVAE, SSCVAE=SSCVAE, CVAE=CVAE,
                             ssl_deg=ssl_deg, ssl_iniact=ssl_iniact, ssl_pnp=ssl_pnp, ssl_glad=ssl_glad, tom_odor=tom_odor, tom_perf=tom_perf,
                             ogt=ogt, topt=topt, add_ssl_seqs=add_ssl_seqs, add_tom_seqs=add_tom_seqs, only_tom_seqs=only_tom_seqs,
                             tom_val_index=tom_val_index, over_sample=over_sample)

    return all_data, train_data, val_data


def get_datasets_from_BLAT(file=None, train_ratio=1, device = None, SSVAE=False, SSCVAE=False, CVAE=False,
                            assay=False, add_assay_seqs=False, val_index=None):
    seqs = list(SeqIO.parse(file, "fasta"))

    data_len = len(seqs)
    seq_len = len(seqs[0])

    # Split into train/validation
    train_length = int(train_ratio * data_len)
    val_length = data_len - train_length

    indices = list(range(data_len))
    random.shuffle(indices)
    train_indices = indices[:train_length]
    val_indices = indices[train_length:]
    train_seqs = [seqs[i] for i in train_indices]
    val_seqs = [seqs[i] for i in val_indices]


    all_data = BLATDataset(seqs, device = device, SSVAE=SSVAE, SSCVAE=SSCVAE, CVAE=CVAE,
                 assay=assay, add_assay_seqs=add_assay_seqs, val_index=val_index)
    train_data = BLATDataset(train_seqs, device = device, SSVAE=SSVAE, SSCVAE=SSCVAE, CVAE=CVAE,
                 assay=assay, add_assay_seqs=add_assay_seqs, val_index=val_index)
    val_data = BLATDataset(val_seqs, device = device, SSVAE=SSVAE, SSCVAE=SSCVAE, CVAE=CVAE,
                 assay=assay, add_assay_seqs=add_assay_seqs, val_index=val_index)

    return all_data, train_data, val_data


def get_datasets_from_PDE(file=None, train_ratio=1, backbone_idx=25295, gappy_colx_threshold=1, device='cuda', SSVAE=False, SSCVAE=False, CVAE=False,
                        logHIF=False, ogt=False, topt=False, add_logHIF_seqs=False, val_index=None):
    seqs = list(SeqIO.parse(file, "fasta"))
    backbone = seqs[backbone_idx]
    data_len = len(seqs)
    seq_len = len(seqs[0])

    # Split into train/validation
    train_length = int(train_ratio * data_len)
    val_length = data_len - train_length

    indices = list(range(data_len))
    random.shuffle(indices)
    train_indices = indices[:train_length]
    val_indices = indices[train_length:]
    train_seqs = [seqs[i] for i in train_indices]
    val_seqs = [seqs[i] for i in val_indices]

    all_data = PDEDataset(seqs, backbone=backbone, gappy_colx_threshold=gappy_colx_threshold, device = device, SSVAE=SSVAE, SSCVAE=SSCVAE, CVAE=CVAE,
                            logHIF=logHIF, ogt=ogt, topt=topt, add_logHIF_seqs=add_logHIF_seqs, val_index=val_index)

    train_data = PDEDataset(train_seqs, backbone=backbone, gappy_colx_threshold=gappy_colx_threshold, device = device, SSVAE=SSVAE, SSCVAE=SSCVAE, CVAE=CVAE,
                 logHIF=logHIF, ogt=ogt, topt=topt, add_logHIF_seqs=add_logHIF_seqs, val_index=val_index)

    val_data = PDEDataset(val_seqs, backbone=backbone, gappy_colx_threshold=gappy_colx_threshold, device = device, SSVAE=SSVAE, SSCVAE=SSCVAE, CVAE=CVAE,
                 logHIF=logHIF, ogt=ogt, topt=topt, add_logHIF_seqs=add_logHIF_seqs, val_index=val_index)

    return all_data, train_data, val_data


def seqs_collate(tensors):
    encoded_seq, weights, neffs, labels = zip(*tensors)
    labels = labels if type(labels[0]) == type(None) else torch.stack(labels)
    return torch.stack(encoded_seq), neffs[0], labels, torch.stack(weights)


def get_protein_dataloader(dataset, batch_size = 128, shuffle = False, get_seqs = False, random_weighted_sampling = False):
    #sampler = WeightedRandomSampler(weights = dataset.weights, num_samples = len(dataset.weights), replacement = True) if random_weighted_sampling else None
    #return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle if not random_weighted_sampling else not random_weighted_sampling, collate_fn = seqs_collate, sampler = sampler)
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = seqs_collate)


# ensure that alignments used have an amino acid more than a threshold percentage to ensure we dont have too gappy alignments included
def postprocess_aln_coverage(alignment, threshold=0.5):
    keep_colx = []
    for i, x in enumerate(alignment):
        coverage = len(x[x!=IUPAC_SEQ2IDX['<mask>']])/alignment.size(1)
        if coverage > threshold:
            keep_colx.append(i)
    seqs = alignment[keep_colx]
    return seqs


def prepro_alignment(alignment, backbone, threshold):

    seqs_in_int = []
    seqs_aa = []
    is_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    gap = ['-','.']
    for i, record in enumerate(alignment):
        skip = False
        for aa in record.seq.upper():
            try:
                assert aa in is_aa or aa in gap, f"{aa}"
            except AssertionError:
                skip = True
                break

        if not skip:
            seqs_in_int.append([IUPAC_SEQ2IDX[aa] for aa in str(record.seq).upper()])
            seqs_aa.append(str(record.seq).upper())

    seqs_in_int = np.array(seqs_in_int)
    keep_cols = []
    for i, aa in enumerate([IUPAC_SEQ2IDX[aa] for aa in np.array(backbone)]):
        if IUPAC_IDX2SEQ[aa] != '<mask>':
            keep_cols.append(i)

        if IUPAC_IDX2SEQ[aa] == '<mask>':
            gappy_col = seqs_in_int[:,i]
            gap_char = IUPAC_SEQ2IDX['<mask>']
            res_procent = 1 - len(gappy_col[gappy_col==gap_char]) / len(gappy_col)
            if res_procent>=threshold:
                keep_cols.append(i)

    seqs_in_int = seqs_in_int[:,keep_cols]

    return seqs_in_int, seqs_aa


def fill_missing(num_seqs, label_df, cvae, device):
    """
    num_seqs: number of rows that are needed so after concat axis=0 it matches desired .size(0)
    label_df: dataframe that has the desired .size(1) and the col idx to fill unknowns
    cvae: set true if sscvae or cvae else false
    """
    if cvae:
        if 'Sequence' in label_df.columns:
            label_df = conditional_data.drop('Sequence', axis=1)

        # get idx of unknown cols
        col_idx_unknown = label_df.columns.get_indexer(label_df.filter(regex='UNKNOWN').columns.values)

        # initiate filler labels and fill with labels for unknown
        filler_labels = np.zeros((num_tom_sequences, len(label_df.columns)))
        filler_labels[:,col_idx_unknown] = 1
        filler_labels = torch.tensor(filler_labels, device=device).to(torch.float)

    else:
        filler_labels = np.ones((num_seqs, num_cols))*-float('nan')
        filler_labels = torch.tensor(filler_labels, device=device).to(torch.float)


    return filler_labels


def prep_assay_labels(conditional_data, device, num_sequences, cvae=False):
    """
    input df that if discretized have nans in separate col as UNKNOWN

    output: labels and labelled seqs for cvae or semi-sup
    """
    label_df = conditional_data.drop('Sequence', axis=1)
    labels = torch.tensor(label_df.values.astype(float), device=device).to(torch.float)
    labelled_seqs = torch.tensor(conditional_data['Sequence'], device=device)

    # get idx of unknown cols
    col_idx_unknown = label_df.columns.get_indexer(label_df.filter(regex='UNKNOWN').columns.values)

    # initiate filler labels and fill with labels for unknown
    if cvae:
        filler_labels = np.zeros((num_sequences, len(label_df.columns)))
        filler_labels[:,col_idx_unknown] = 1
        filler_labels = torch.tensor(filler_labels, device=device).to(torch.float)

    else:
        filler_labels = torch.ones((num_sequences, len(label_df.columns)))*float('nan')
        filler_labels = torch.tensor(filler_labels, device=device).to(torch.float)



    return labels, filler_labels, labelled_seqs


def curate_ssl_data(sheet, ast_threshold=30):
    """
    Create df of ssl data filtered at ast conc value and nan dropped
    returns path
    """
    # load premade df
    df = pickle.load( open( 'data_handler/files/assay_data/All_lipase_ssl_data.pkl', 'rb' ) )
    # filter using AST conc
    df = df.dropna(subset=['Supernatant AST measurement_concentration ppm'])
    df = df[df['Supernatant AST measurement_concentration ppm']>ast_threshold]
    # keep only relevant colx
    df = df[[sheet, 'Sequence']]
    # filter
    df = df.dropna(subset=[sheet])
    # dump
    pickle.dump(df[[sheet, 'Sequence']], open( 'curated_lipase_data_combined.pkl', "wb" ) )

    return df

def encode_data(model, dataset, batch_size=0):

    dataloader = get_protein_dataloader(dataset, batch_size = batch_size, get_seqs = False)

    mus = []
    stds = []
    with torch.no_grad():
        for encoded_seq, weights, neffs, labels in dataloader:
                mu, logvar = model.encoder(encoded_seq, labels)
                mus.append(mu.cpu())
                stds.append(logvar.mul(0.5).exp().cpu())
    mus = torch.cat(mus)
    stds = torch.cat(stds)
    return mus.numpy(), stds.numpy()


def add_unknown_filler_labels(conditional_data_dim, length_mutants):
    num_labels_to_add = conditional_data_dim // 4
    extra_for_wt = 1
    labels = np.array([[1,0,0,0]*num_labels_to_add for i in range(length_mutants+extra_for_wt)])
    return labels


def prep_TOM_data(model, device, val_index=[None], conditional_data_dim=None, ssvae=False, sscvae=False):
    if len(val_index) > 0:
        TOM_data = pickle.load( open( 'TOM_data_combined.pkl', "rb" ) ).loc[val_index].reset_index(drop=True)
    else:
        TOM_data = pickle.load( open( 'TOM_data_combined.pkl', "rb" ) )
    y_perf = TOM_data['TOM RP(Wash)']
    y_odor = TOM_data['TOM RP(odor)']
    X = torch.stack([torch.tensor(seq, device=device) for seq in TOM_data['Sequence']])
    if not sscvae:
        if conditional_data_dim > 0 and not ssvae:
            labels = add_unknown_filler_labels(conditional_data_dim, len(X)-1)
            labels = torch.tensor(labels, device=device).to(torch.float)
        else:
            labels = [None]
    if sscvae:
        try:
            x, _ = model.pretrained_model.encoder(X, [None])
        except AttributeError:
            x = X
        logits = model.predict_label(x, i=0)
        labels = torch.zeros(X.size(0), logits.size(1)).cuda().scatter_(1, torch.argmax(logits,1,keepdim=True), 1) if torch.cuda.is_available() else torch.zeros(x.size(0), logits.size(1)).scatter_(1, torch.argmax(logits,1,keepdim=True), 1)
    return X, y_perf, y_odor, labels


def filler(x):
    if 'Discretized topt_i_high' in x:
        col = 'Discretized topt_i_high'
    else:
        col = 'Discretized ogt_high'
    if pd.isna(x[col]):
        return 1
    else:
        return 0

def merge_ogt(df):
    # load
    assay = 'ogt'
    lip_ogt = pd.read_csv('/z/bio/bioinfo/jaks_project/sp400_ogt.tsv', sep="\t")

    no_dup = lip_ogt.copy()
    no_dup = lip_ogt.drop(lip_ogt[lip_ogt['sequence'].duplicated(keep=False)].index, axis=0)
    dup = lip_ogt[lip_ogt['sequence'].duplicated(keep=False)==True]
    dup['mean'] = dup.groupby(['sequence']).ogt.transform('mean')
    dup = dup.drop_duplicates(subset='sequence').drop(assay, axis=1).rename(columns={'mean': assay})
    lip_ogt = pd.concat((no_dup, dup)).reset_index(drop=True)

    if cvae:
        lip_ogt['Discretized '+ assay] = pd.qcut(lip_ogt[assay], [0, 0.25 ,0.75, 1], labels=['low', 'mid', 'high']).astype(object).fillna("UNKNOWN")
        lip_ogt_ohc = pd.get_dummies(lip_ogt, columns=['Discretized '+ assay])
        df_new = df.merge(lip_ogt_ohc, how='left', on='sequence')
        df_new['Discretized '+ assay + '_UNKNOWN'] = df_new.apply(lambda x: filler(x),  axis=1)
        df_new = df_new.fillna(0)
    else:
        df_new = df.merge(lip_ogt, how='left', on='sequence')

    return df_new


def merge_Topt(df):
    assay = 'topt_i'

    assays = ['ogt', 'topt_i']
    lip_topt = pd.read_csv('/z/bio/bioinfo/jaks_project/sp400_protein_temp_opt.tsv', sep="\t")

    # take not duplicated
    no_dup = lip_topt.copy()
    no_dup = lip_topt.drop(lip_topt[lip_topt['sequence'].duplicated(keep=False)].index, axis=0)
    # combine duplicated into mean
    dup = lip_topt[lip_topt['sequence'].duplicated(keep=False)==True]
    dup['mean'] = dup.groupby(['sequence'])['topt_i'].transform('mean')
    dup = dup.drop_duplicates(subset='sequence').drop(assay, axis=1).rename(columns={'mean': assay})
    lip_topt = pd.concat((no_dup, dup)).reset_index(drop=True)

    if cvae:
        lip_topt['Discretized '+ assay] = pd.qcut(lip_topt[assay], [0, 0.25 ,0.75, 1], labels=['low', 'mid', 'high']).astype(object).fillna("UNKNOWN")
        lip_topt_ohc = pd.get_dummies(lip_topt, columns=['Discretized '+ assay])

        df_new = df.merge(lip_topt_ohc, how='left', on='sequence')
        df_new['Discretized '+ assay + '_UNKNOWN'] = df_new.apply(lambda x: filler(x),  axis=1)
        df_new = df_new.fillna(0)
    else:
        df_new = df.merge(lip_topt, how='left', on='sequence')

    return df_new

def merge_Topt_n_ogt(df):
    assays = ['ogt', 'topt_i']
    lip_assay = pd.read_csv('/z/bio/bioinfo/jaks_project/sp400_protein_temp_opt.tsv', sep="\t")

    # take not duplicated
    no_dup = lip_assay.copy()
    no_dup = lip_assay.drop(lip_assay[lip_assay['sequence'].duplicated(keep=False)].index, axis=0)

    # combine duplicated into mean
    dup = lip_assay[lip_assay['sequence'].duplicated(keep=False)==True]
    dup['mean_ogt'] = dup.groupby(['sequence'])['ogt'].transform('mean')
    dup['mean_topt_i'] = dup.groupby(['sequence'])['topt_i'].transform('mean')
    dup = dup.drop_duplicates(subset='sequence').drop(assays, axis=1).rename(columns={'mean_ogt': 'ogt', 'mean_topt_i':'topt_i'})

    lip_assay = pd.concat((no_dup, dup)).reset_index(drop=True)

    if cvae:
        lip_assay['Discretized '+ 'ogt'] = pd.qcut(lip_assay['ogt'], [0, 0.25 ,0.75, 1], labels=['low', 'mid', 'high']).astype(object).fillna("UNKNOWN")
        lip_assay['Discretized '+ 'topt_i'] = pd.qcut(lip_assay['topt_i'], [0, 0.25 ,0.75, 1], labels=['low', 'mid', 'high']).astype(object).fillna("UNKNOWN")
        lip_assay_ohc = pd.get_dummies(lip_assay, columns=['Discretized ogt', 'Discretized topt_i'])

        lip_assay = df.merge(lip_assay_ohc, how='left', on='sequence')
        lip_assay['Discretized '+ 'ogt' + '_UNKNOWN'] = lip_assay.apply(lambda x: filler(x),  axis=1)
        lip_assay['Discretized '+ 'topt_i' + '_UNKNOWN'] = lip_assay.apply(lambda x: filler(x),  axis=1)
        lip_assay = lip_assay.fillna(0)
    else:
        lip_assay = df.merge(lip_assay, how='left', on='sequence')

    return lip_assay


def match_ogt_n_topt(seqs, ogt, topt, device):
    df = pd.DataFrame()
    if type(seqs)==torch.Tensor:
        seqs=seqs.cpu().numpy()
    df['alignments'] = [seq for seq in seqs]
    df['sequence'] = seqs
    df['sequence'] = [re.sub('-', '', "".join(seq)) for seq in df['sequence']]
    if ogt and not topt:
        combined = merge_ogt(df)
        OGT = combined.drop(['ogt', 'alignments', 'sequence', 'key'], axis=1)

    if topt and not ogt:
        combined = merge_Topt(df)
        Topt = combined.drop(['ogt', 'alignments', 'sequence', 'key', 'topt_i', 'topt_err'], axis=1)

    if topt and ogt:
        combined = merge_Topt_n_ogt(df)
        if cvae:
            assay_data = combined.drop(['ogt', 'alignments', 'sequence', 'key', 'topt_i', 'topt_err'], axis=1)
        else:
            assay_data = combined.drop(['alignments_x', 'alignments_y', 'sequence', 'key', 'topt_err'], axis=1)
    if ogt and not topt:
        return OGT # torch.tensor(OGT, device=device).float()
    if topt and not ogt:
        return Topt #torch.tensor(Topt, device=device).float()
    if ogt and topt:
        return assay_data #torch.tensor(assay_data, device=device).float()
    return None


def prep_tom_labels(val_index=None, tom_odor=0, tom_perf=0, discretize=False, quantiles=[0, 0.25 ,0.75, 1]):
    p = pickle.load( open( 'TOM_data_combined.pkl', "rb" ) ).drop(val_index, axis=0).reset_index(drop=True)
    if discretize:
        cond_label_cols = ['TOM RP(odor)', 'TOM RP(Wash)']
        for assay in cond_label_cols:
            p['Discretized '+ assay] = pd.qcut(p[assay], quantiles, labels=['low', 'mid', 'high']).astype(object).fillna("UNKNOWN")

        d_odor = 'Discretized ' + 'TOM RP(odor)'
        d_perf = 'Discretized ' + 'TOM RP(Wash)'
        cond_label_cols = [d_odor, d_perf]
        discretized_p = p[['Sequence', d_odor, d_perf]]
        TOM_ohc_discretized_p = pd.get_dummies(discretized_p, columns=[d_odor, d_perf])
        TOM_ohc_discretized_p['Discretized TOM RP(odor)_UNKNOWN'] = [0]*len(TOM_ohc_discretized_p)
        TOM_ohc_discretized_p['Discretized TOM RP(Wash)_UNKNOWN'] = [0]*len(TOM_ohc_discretized_p)

        col_order = ['Sequence',
                    'Discretized TOM RP(odor)_UNKNOWN',
                    'Discretized TOM RP(odor)_high',
                    'Discretized TOM RP(odor)_mid',
                    'Discretized TOM RP(odor)_low',
                    'Discretized TOM RP(Wash)_UNKNOWN',
                    'Discretized TOM RP(Wash)_high',
                    'Discretized TOM RP(Wash)_mid',
                    'Discretized TOM RP(Wash)_low']
        df = TOM_ohc_discretized_p[col_order]

    else:
        df = p

    keep_col = []
    if tom_odor:
        keep_col += [col for col in df.columns if 'odor' in col]
    if tom_perf:
        keep_col += [col for col in df.columns if 'Wash' in col]
    keep_col += ['Sequence']
    return df[keep_col]


def prep_ssl_data(ssl_deg=0, ssl_iniact=0, ssl_pnp=0, ssl_glad=0, discretize=False, ast_threshold=30, quantiles=[0, 0.25 ,0.75, 1]):
    p = pickle.load( open( 'data_handler/files/assay_data/All_lipase_ssl_data.pkl', 'rb' ) )
    p = p.dropna(subset=['Supernatant AST measurement_concentration ppm'])
    p = p[p['Supernatant AST measurement_concentration ppm']>ast_threshold].reset_index(drop=True)
    if discretize:
        deg = 'Supernatant IWS observation_degradation_factor Rinso 10.2 30.0 1.75'
        ini_act = 'Supernatant IWS observation_initial_activity Rinso 10.2 30.0 1.75'
        pnp = 'Supernatant pNP observation_normalized_lipex_units Model-X-like'
        glad = 'Supernatant GLAD observation_normalized_lipex_units Model-Y-like'
        cond_label_cols = [deg, ini_act, pnp, glad]

        for assay in cond_label_cols:
            p['Discretized '+ assay] = pd.qcut(p[assay], quantiles,
                                               labels=['low', 'mid', 'high']).astype(object).fillna("UNKNOWN")

        p['Discretized Supernatant pNP observation_normalized_lipex_units Model-X-like'].value_counts()

        d_deg = 'Discretized Supernatant IWS observation_degradation_factor Rinso 10.2 30.0 1.75'
        d_act = 'Discretized Supernatant IWS observation_initial_activity Rinso 10.2 30.0 1.75'
        d_pnp = 'Discretized Supernatant pNP observation_normalized_lipex_units Model-X-like'
        d_gla = 'Discretized Supernatant GLAD observation_normalized_lipex_units Model-Y-like'
        cond_label_cols = [d_deg, d_act, d_pnp, d_gla]
        discretized_p = p[['Sequence', d_deg, d_act, d_pnp, d_gla]]
        ohc_discretized_p = pd.get_dummies(discretized_p, columns=[d_deg, d_act, d_pnp, d_gla])
        ohc_discretized_p['Discretized Supernatant IWS observation_initial_activity Rinso 10.2 30.0 1.75_UNKNOWN'] = [0]*len(ohc_discretized_p)

        col_order = ['Sequence',
                    'Discretized Supernatant IWS observation_degradation_factor Rinso 10.2 30.0 1.75_UNKNOWN',
                    'Discretized Supernatant IWS observation_degradation_factor Rinso 10.2 30.0 1.75_high',
                    'Discretized Supernatant IWS observation_degradation_factor Rinso 10.2 30.0 1.75_mid',
                     'Discretized Supernatant IWS observation_degradation_factor Rinso 10.2 30.0 1.75_low',
                     'Discretized Supernatant IWS observation_initial_activity Rinso 10.2 30.0 1.75_UNKNOWN',
                     'Discretized Supernatant IWS observation_initial_activity Rinso 10.2 30.0 1.75_high',
                     'Discretized Supernatant IWS observation_initial_activity Rinso 10.2 30.0 1.75_mid',
                     'Discretized Supernatant IWS observation_initial_activity Rinso 10.2 30.0 1.75_low',
                     'Discretized Supernatant pNP observation_normalized_lipex_units Model-X-like_UNKNOWN',
                     'Discretized Supernatant pNP observation_normalized_lipex_units Model-X-like_high',
                     'Discretized Supernatant pNP observation_normalized_lipex_units Model-X-like_mid',
                     'Discretized Supernatant pNP observation_normalized_lipex_units Model-X-like_low',
                     'Discretized Supernatant GLAD observation_normalized_lipex_units Model-Y-like_UNKNOWN',
                     'Discretized Supernatant GLAD observation_normalized_lipex_units Model-Y-like_high',
                     'Discretized Supernatant GLAD observation_normalized_lipex_units Model-Y-like_mid',
                     'Discretized Supernatant GLAD observation_normalized_lipex_units Model-Y-like_low',
                    ]
        df = ohc_discretized_p[col_order]
    else:
        df = p

    keep_col = []
    if ssl_deg:
        keep_col += [col for col in df.columns if 'degradation' in col]
    if ssl_pnp:
        keep_col += [col for col in df.columns if 'pNP' in col]
    if ssl_iniact:
        keep_col += [col for col in df.columns if 'initial_activity' in col]
    if ssl_glad:
        keep_col += [col for col in df.columns if 'GLAD' in col]
    keep_col += ['Sequence']
    return df[keep_col]

class ProteinDataset(Dataset):
    def __init__(self, data, device = None, SSVAE=False, SSCVAE=False, CVAE=False, regCVAE=False,
                 assays_to_include=[], train_index=None, train_with_assay_seq=False,
                 only_assay_seqs=False, training=True):
        super().__init__()
        if len(data) == 0:
            self.encoded_seqs = torch.Tensor()
            self.weights = torch.Tensor()
            self.neff = 0
            return
        if training:
            actual_labels = data.dropna(subset=assays_to_include)
            train_labelled_data = actual_labels.loc[train_index]

        if 'blast' in data.columns and not train_with_assay_seq:
            data = data.drop(data.index[data['blast'] != 1].tolist(), axis=0)
        if 'blast' not in data.columns and training:
            data = data.drop(actual_labels.index, axis=0)

        if training:
            if SSVAE or CVAE or SSCVAE or regCVAE or train_with_assay_seq:
                data = pd.concat((data, train_labelled_data), axis=0)

        if only_assay_seqs:
            data = train_labelled_data
        self.encoded_seqs = torch.stack([torch.tensor(seq, device=device) for seq in data['seqs'].values]).long()
        num_sequences = self.encoded_seqs.size(0)
        self.labels = torch.Tensor()

        discretize = True if SSCVAE or CVAE else False

        labels = prep_any_labels(data,
                                 assays_to_include, SSCVAE=SSCVAE,
                                 discretize=discretize, training=training)

        for col in labels.columns:
            if 'UNKNOWN' in col and SSCVAE:
                labels = labels.drop(col, axis=1)
                labels[labels.eq(0).all(axis=1)==True] = labels[labels.eq(0).all(axis=1)==True]*float('nan')
        self.labels = torch.tensor(labels.values.astype(float), device=device).to(torch.float)

        if not SSVAE and not CVAE and not SSCVAE and not regCVAE:
            self.labels = None

        # Calculate weights
        weights = []
        flat_one_hot = F.one_hot(self.encoded_seqs, num_classes=max(IUPAC_SEQ2IDX.values())+1).float().flatten(1)
        weight_batch_size = 1000
        for i in range(self.encoded_seqs.size(0) // weight_batch_size + 1):
            x = flat_one_hot[i * weight_batch_size : (i + 1) * weight_batch_size]
            similarities = torch.mm(x, flat_one_hot.T)
            lengths = (self.encoded_seqs[i * weight_batch_size : (i + 1) * weight_batch_size] != IUPAC_SEQ2IDX["<mask>"]).sum(1).unsqueeze(-1)
            w = 1.0 / (similarities / lengths).gt(0.8).sum(1).float()
            weights.append(w)
        self.weights = torch.cat(weights)
        self.neff = self.weights.sum()
        print('Neff', self.neff)
    def write_to_file(self, filepath):
        for s, w in zip(self.seqs, self.weights):
            s.id = s.id + ':' + str(float(w))
        SeqIO.write(self.seqs, filepath, 'fasta')

    def __len__(self):
        return len(self.encoded_seqs)

    def __getitem__(self, i):
        if self.labels == None:
            labels = self.labels
        else:
            labels = self.labels[i]
        return self.encoded_seqs[i], self.weights[i], self.neff, labels


def get_datasets(data=None, train_ratio=0, device = None, SSVAE=False, SSCVAE=False, CVAE=False, regCVAE=False, assays_to_include=[], train_index=None, train_with_assay_seq=False, only_assay_seqs=False):
    seqs = data['seqs']
    data_len = len(seqs)

    # Split into train/validation
    train_length = int(train_ratio * data_len)
    val_length = data_len - train_length

    if 'blast' in data.columns and train_ratio < 1:
        val_blast_index = data.drop(data.index[data['blast'] != 1].tolist(), axis=0).sample(frac=(1-train_ratio), replace=False, random_state=42).index
        train_blast_index = [idx for idx in data.drop(data.index[data['blast'] != 1].tolist(), axis=0).index if idx not in val_blast_index]
        val_seqs = data.drop(data.index[data['blast'] != 1].tolist(), axis=0).drop(train_blast_index, axis=0)
        train_seqs = data.drop(data.index[data['blast'] != 1].tolist(), axis=0).drop(val_blast_index, axis=0)
        labelled_data = data.dropna(subset=assays_to_include)
        train_seqs = pd.concat((train_seqs, labelled_data))
    if train_ratio == 1:
        train_seqs = data
        val_seqs = []
    if 'blast' not in data.columns and train_ratio < 1:
        labelled_data = data.dropna(subset=assays_to_include)
        val_blast_index = data.drop(labelled_data.index, axis=0).sample(frac=(1-train_ratio), replace=False, random_state=42).index
        train_blast_index = [idx for idx in data.drop(labelled_data.index, axis=0).index if idx not in val_blast_index]
        val_seqs = data.drop(labelled_data.index, axis=0).drop(train_blast_index, axis=0)
        train_seqs = data.drop(labelled_data.index, axis=0).drop(val_blast_index, axis=0)
        train_seqs = pd.concat((train_seqs, labelled_data))
    all_data = ProteinDataset(data, device = device, SSVAE=SSVAE, SSCVAE=SSCVAE, CVAE=CVAE, regCVAE=regCVAE, assays_to_include=assays_to_include, train_index=train_index, train_with_assay_seq=train_with_assay_seq, only_assay_seqs=only_assay_seqs)
    train_data = ProteinDataset(train_seqs, device = device, SSVAE=SSVAE, SSCVAE=SSCVAE, CVAE=CVAE, regCVAE=regCVAE, assays_to_include=assays_to_include, train_index=train_index, train_with_assay_seq=train_with_assay_seq, only_assay_seqs=only_assay_seqs)
    keep_val_idx = train_blast_index = [idx for idx in data.dropna(subset=assays_to_include).index if idx not in train_index]
    val_data = ProteinDataset(val_seqs, device = device, SSVAE=SSVAE, SSCVAE=SSCVAE, CVAE=CVAE, regCVAE=regCVAE, assays_to_include=assays_to_include, train_index=keep_val_idx, train_with_assay_seq=train_with_assay_seq, only_assay_seqs=only_assay_seqs, training=False)
    return all_data, train_data, val_data


def prep_any_labels(data, wanted_cols, discretize=False, SSCVAE=False, training=True, quantiles=[0, 0.25 ,0.75, 1]):
    if discretize:
        d_assay_names = []
        col_order = []
        if training:
            for assay in wanted_cols:
                    data['Discretized '+ assay] = pd.qcut(data[assay], quantiles, labels=['low', 'mid', 'high']).astype(object).fillna("UNKNOWN")
                    d_assay_names.append('Discretized ' + assay)
                    col_order.append(['Discretized '+assay+'_UNKNOWN',
                                      'Discretized '+assay+'_high',
                                      'Discretized '+assay+'_mid',
                                      'Discretized '+assay+'_low'])
            discretized_data = data[['seqs']+d_assay_names]
            ohc_discretized_data = pd.get_dummies(discretized_data, columns=d_assay_names)
            ohc_discretized_data['Discretized'+assay+'_UNKNOWN'] = [0]*len(ohc_discretized_data)
            col_order = [item for sublist in col_order for item in sublist]
            col_order = col_order
            labels = ohc_discretized_data[col_order]
        else:
            if SSCVAE:
                labels = pd.DataFrame()
                labels['Discretized TOM RP(odor)_UNKNOWN'] = np.tile([1], len(data))
                labels['Discretized TOM RP(odor)_high'] = np.tile([0], len(data))
                labels['Discretized TOM RP(odor)_mid'] = np.tile([0], len(data))
                labels['Discretized TOM RP(odor)_low'] = np.tile([0], len(data))

    else:
        labels = data[wanted_cols]

    return labels
