import math
import time
from collections import defaultdict
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

def train_epoch(epoch, model, optimizer, scheduler, train_loader):
    """
        epoch: Index of the epoch to run
        model: The model to run data through. Forward should return a tuple of (loss, metrics_dict).
        optimizer: The optimizer to step with at every batch
        train_loader: PyTorch DataLoader to generate batches of training data
        log_interval: Interval in seconds of how often to log training progress (0 to disable batch progress logging)
    """
    train_loss = 0
    train_count = 0

    if scheduler is not None:
        learning_rates = []

    acc_metrics_dict = defaultdict(lambda: 0)

    for batch_idx, xb in enumerate(train_loader):
        batch_size = xb.size(0) if isinstance(xb, torch.Tensor) else xb[0].size(0)
        loss, batch_metrics_dict, px_z = train_batch(model, optimizer, xb, scheduler)

        # Model saves loss types in dict calculate accumulated metrics
        semisup_metrics =  ["seq2y_loss",
                            "z2y_loss",
                            "labelled seqs",
                            "unlabelled seqs",
                            "unlabelled_loss",
                            "labelled_loss"]

        for key, value in batch_metrics_dict.items():
            if key not in semisup_metrics:
                acc_metrics_dict[key] += value * batch_size
                acc_metrics_dict[key + "_count"] += batch_size

            if 'seq2y' in key or 'z2y' in key:
                if batch_metrics_dict['labelled seqs'] != None:
                    acc_metrics_dict[key] += value * batch_metrics_dict['labelled seqs'].size(0)
                    acc_metrics_dict[key + "_count"] += batch_metrics_dict['labelled seqs'].size(0)
                else:
                    acc_metrics_dict[key] += 0
                    acc_metrics_dict[key + "_count"] += 1

            if key == "unlabelled_loss":
                acc_metrics_dict[key] += value * batch_metrics_dict['unlabelled seqs'].size(0)
                acc_metrics_dict[key + "_count"] += batch_metrics_dict['unlabelled seqs'].size(0)

            if key == "labelled_loss" and batch_metrics_dict['labelled seqs'] != None:
                acc_metrics_dict[key] += value * batch_metrics_dict['labelled seqs'].size(0)
                acc_metrics_dict[key + "_count"] += batch_metrics_dict['labelled seqs'].size(0)
            else:
                acc_metrics_dict[key] += 0
                acc_metrics_dict[key + "_count"] += 1



        metrics_dict = {k: acc_metrics_dict[k] / acc_metrics_dict[k + "_count"] for k in acc_metrics_dict.keys() if not k.endswith("_count")}

        train_loss += loss.item() * batch_size
        train_count += batch_size

        if scheduler is not None:
            learning_rates.append(scheduler.get_last_lr())

    average_loss = train_loss / train_count

    if scheduler is not None:
        metrics_dict['learning_rates'] = learning_rates

    return average_loss, metrics_dict, px_z

def train_batch(model, optimizer, xb, scheduler = None):
    model.train()

    # Reset gradient for next batch
    optimizer.zero_grad()
    # Push whole batch of data through model.forward() account for protein_data_loader pushes more than tensor through
    if isinstance(xb, Tensor):
        loss, batch_metrics_dict, px_z = model(xb)
    else:
        loss, batch_metrics_dict, px_z = model(*xb)
    # Calculate the gradient of the loss w.r.t. the graph leaves
    loss.backward()
    clip_grad_value = 200
    if clip_grad_value is not None:
        clip_grad_value_(model.parameters(), clip_grad_value)
    # for n, p in model.named_parameters():
    #     try:
    #         if p.grad.norm().item() > 100:
    #             print(n, p.grad.norm().item())
    #     except AttributeError:
    #         continue
    # Step in the direction of the gradient
    optimizer.step()

    # Schedule learning rate
    if scheduler is not None:
        scheduler.step()

    return loss, batch_metrics_dict, px_z

def validate(epoch, model, validation_loader):
    model.eval()

    validation_loss = 0
    validation_count = 0
    with torch.no_grad():
        acc_metrics_dict = defaultdict(lambda: 0)
        for i, xb in enumerate(validation_loader):
            batch_size = xb.size(0) if isinstance(xb, torch.Tensor) else xb[0].size(0)
            # Push whole batch of data through model.forward()
            if isinstance(xb, Tensor):
                loss, batch_metrics_dict, px_z = model(xb)
            else:
                loss, batch_metrics_dict, px_z = model(*xb)


            semisup_metrics =  ["seq2y_loss",
                                "z2y_loss",
                                "labelled seqs",
                                "unlabelled seqs",
                                "unlabelled_loss",
                                "labelled_loss"]
            # Calculate accumulated metrics
            for key, value in batch_metrics_dict.items():
                if key not in semisup_metrics:
                    acc_metrics_dict[key] += value * batch_size
                    acc_metrics_dict[key + "_count"] += batch_size

                if 'seq2y' in key or 'z2y' in key:
                    if batch_metrics_dict['labelled seqs'] != None:
                        acc_metrics_dict[key] += value * batch_metrics_dict['labelled seqs'].size(0)
                        acc_metrics_dict[key + "_count"] += batch_metrics_dict['labelled seqs'].size(0)
                    else:
                        acc_metrics_dict[key] += 0
                        acc_metrics_dict[key + "_count"] += 1

                if key == "unlabelled_loss":
                    acc_metrics_dict[key] += value * batch_metrics_dict['unlabelled seqs'].size(0)
                    acc_metrics_dict[key + "_count"] += batch_metrics_dict['unlabelled seqs'].size(0)

                if key == "labelled_loss" and batch_metrics_dict['labelled seqs'] != None:
                    acc_metrics_dict[key] += value * batch_metrics_dict['labelled seqs'].size(0)
                    acc_metrics_dict[key + "_count"] += batch_metrics_dict['labelled seqs'].size(0)
                else:
                    acc_metrics_dict[key] += 0
                    acc_metrics_dict[key + "_count"] += 1

            validation_loss += loss.item() * batch_size
            validation_count += batch_size


            metrics_dict = {k: acc_metrics_dict[k] / acc_metrics_dict[k + "_count"] for k in acc_metrics_dict.keys() if not k.endswith("_count")}

    average_loss = validation_loss / validation_count
    return average_loss, metrics_dict, px_z
