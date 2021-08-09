from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import torch
from torch.nn import functional as F
from data_handler import add_unknown_filler_labels
from torch.distributions.normal import Normal


def predict_downstream(X, y, X_val, y_val, model, downstream_model=None, labels_val=[None], labels_train=[None], cv=10, seed=None):
    np.random.seed(seed)
    assert downstream_model != None, "Please specify downstream model"

    ae = []
    se = []
    try:
        X, _ = model.pretrained_model.encoder(X, labels_train)
        X_val, _ = model.pretrained_model.encoder(X_val, labels_val)
    except: AttributeError

    if len(X_val) == 0:
        kf = KFold(n_splits=cv)
        for train_index, test_index in kf.split(X):
            if type(labels_train[0]) != type(None):
                labels_training = labels_train[train_index]
                labels_test = labels_train[test_index]
            else:
                labels_training = labels_train
                labels_test = [None]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            encoded_X_train, _ = model.encoder(X_train, labels_training)#.cuda() if torch.cuda.is_available() else model.encoder(X_train, labels_training)
            encoded_X_test, _ = model.encoder(X_test, labels_test)#.cuda() if torch.cuda.is_available() else model.encoder(X_test, labels_test)

            downstream_model.fit(encoded_X_train.cpu().detach().numpy(), y_train)
            pred = downstream_model.predict(encoded_X_test.cpu().detach().numpy())

            se.append(mean_squared_error(pred, y_test))
            ae.append(mean_absolute_error(pred, y_test))
    else:
        encoded_X_train, _ = model.encoder(X, labels_train)
        encoded_X_test, _ = model.encoder(X_val, labels_val)
        downstream_model.fit(encoded_X_train.cpu().detach().numpy(), y)
        pred = downstream_model.predict(encoded_X_test.cpu().detach().numpy())
        se.append(mean_squared_error(pred, y_val))
        ae.append(mean_absolute_error(pred, y_val))


    mse = np.mean(se)
    mae = np.mean(ae)
    mse_std = np.std(se, ddof=1) if len(se)>1 else 0
    mae_std = np.std(ae, ddof=1) if len(ae)>1 else 0

    return mse, mae, mse_std, mae_std

def baseline(X, y, cv=10):
    mae = 0
    mse = 0

    kf = KFold(n_splits=cv)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        mse += mean_squared_error([np.mean(y_train)]*len(y_test), y_test)
        mae += mean_absolute_error([np.mean(y_train)]*len(y_test), y_test)

    return mse/cv, mae/cv



def validate_predict_label_from_z(X_val, y_val, model, labels_val=[None], seed=None):
    np.random.seed(seed)

    try:
        X_val, _ = model.pretrained_model.encoder(X_val, labels_val)
    except: AttributeError

    encoded_X_val, _ = model.encoder(X_val, labels_val)
    pred = model.predict_label_from_z(encoded_X_val, i=0)
    mse = np.mean(mean_squared_error(pred.cpu().numpy(), y_val))
    mae = np.mean(mean_absolute_error(pred.cpu().numpy(), y_val))

    return mse, mae


def validate_predict_label_from_seq(X_val, y_val, model, labels_val=[None], seed=None):
    np.random.seed(seed)

    try:
        X_val, _ = model.pretrained_model.encoder(X_val, labels_val)
    except: AttributeError

    pred = model.predict_label(X_val, i=0)
    mse = np.mean(mean_squared_error(pred.cpu().numpy(), y_val))
    mae = np.mean(mean_absolute_error(pred.cpu().numpy(), y_val))

    return mse, mae



def predict_ohc_baseline(X, y, X_val, y_val, model, downstream_model=None, labels_val=[None], labels_train=[None], cv=10, seed=None):
    np.random.seed(seed)
    assert downstream_model != None, "Please specify downstream model"

    ae = []
    se = []
    try:
        X, _ = model.pretrained_model.encoder(X, labels_train)
        X_val, _ = model.pretrained_model.encoder(X_val, labels_val)
    except: AttributeError

    if len(X_val) == 0:
        kf = KFold(n_splits=cv)
        for train_index, test_index in kf.split(X):
            if type(labels_train[0]) != type(None):
                labels_training = labels_train[train_index]
                labels_test = labels_train[test_index]
            else:
                labels_training = labels_train
                labels_test = [None]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            downstream_model.fit(X_train.cpu().detach().numpy(), y_train)
            pred = downstream_model.predict(X_test.cpu().detach().numpy())

            se.append(mean_squared_error(pred, y_test))
            ae.append(mean_absolute_error(pred, y_test))
    else:
        downstream_model.fit(X.cpu().detach().numpy(), y)
        pred = downstream_model.predict(X_val.cpu().detach().numpy())
        se.append(mean_squared_error(pred, y_val))
        ae.append(mean_absolute_error(pred, y_val))


    mse = np.mean(se)
    mae = np.mean(ae)
    mse_std = np.std(se, ddof=1) if len(se)>1 else 0
    mae_std = np.std(ae, ddof=1) if len(ae)>1 else 0

    return mse, mae, mse_std, mae_std


# TODO decide rather seq2y and z2y should have labels if cvae or regcvae
def pipeline_val_z2y(D_val, sheet, model, seed=None, device='cuda'):
    np.random.seed(seed)
    X_val = torch.stack([torch.tensor(seq, device=device) for seq in D_val['seqs']])
    y_val = D_val[sheet]

    try:
        X_val, _ = model.pretrained_model.encoder(X_val, [None])
    except: AttributeError

    encoded_X_val, _ = model.encoder(X_val, [None])

    pred = model.predict_label_from_z(encoded_X_val, i=0).squeeze(1)
    mse = np.mean(mean_squared_error(pred.cpu().numpy(), y_val))
    mae = np.mean(mean_absolute_error(pred.cpu().numpy(), y_val))

    return mse, mae


def pipeline_val_seq2y(D_val, sheet, model, seed=None, device='cuda'):
    np.random.seed(seed)
    X_val = torch.stack([torch.tensor(seq, device=device) for seq in D_val['seqs']])
    y_val = D_val[sheet]

    try:
        X_val, _ = model.pretrained_model.encoder(X_val, [None])
    except: AttributeError

    pred = model.predict_label(X_val, i=0).squeeze(1)
    mse = np.mean(mean_squared_error(pred.cpu().numpy(), y_val))
    mae = np.mean(mean_absolute_error(pred.cpu().numpy(), y_val))

    return mse, mae

def pipeline_downstream_pred(D, D_val, sheet, model, downstream_model=None, cv=10, seed=42, device='cpu'):
    np.random.seed(seed)
    assert downstream_model != None, "Please specify downstream model"

    X = torch.stack([torch.tensor(seq, device=device) for seq in D['seqs']]).long()
    y = D[sheet].values

    if model.SSVAE or model.VAE: #or model.regCVAE:
        labels,labels_train, labels_val = [None], [None], [None]
    if model.SSCVAE:
        # TODO Right now if SSCVAE and multilabel > 1 then label is not used in elbo
        if model.multilabel > 1:
            labels = torch.cat([torch.zeros(D.size(0), model.predict_label(x, i=i).size(1)).cuda().scatter_(1, torch.argmax(model.predict_label(x, i=i),1,keepdim=True), 1) for i in range(model.multilabel)], dim=1)
            labels = labels.cuda() if torch.cuda.is_available() else labels
        else:
            if len(D_val) == 0:
                logits = model.predict_label(X, i=0)
                labels = torch.zeros(D.size(0), logits.size(1)).cuda().scatter_(1, torch.argmax(logits,1,keepdim=True), 1) if torch.cuda.is_available() else torch.zeros(D.size(0), logits.size(1)).scatter_(1, torch.argmax(logits,1,keepdim=True), 1)
            else:
                X_val = torch.stack([torch.tensor(seq, device=device) for seq in D_val['seqs']]).long()
                logits_t = model.predict_label(X, i=0)
                logits_v = model.predict_label(X_val, i=0)
                labels_train = torch.zeros(X.size(0), logits_t.size(1)).cuda().scatter_(1, torch.argmax(logits_t,1,keepdim=True), 1) if torch.cuda.is_available() else torch.zeros(X.size(0), logits_t.size(1)).scatter_(1, torch.argmax(logits_v,1,keepdim=True), 1)
                labels_val = torch.zeros(X_val.size(0), logits_v.size(1)).cuda().scatter_(1, torch.argmax(logits_v,1,keepdim=True), 1) if torch.cuda.is_available() else torch.zeros(X_val.size(0), logits_v.size(1)).scatter_(1, torch.argmax(logits_t,1,keepdim=True), 1)

    if model.CVAE:
        if len(D_val) == 0:
            labels = add_unknown_filler_labels(model.conditional_data_dim, len(X)-1)
            labels = torch.tensor(labels, device=device).to(torch.float)
        else:
            labels_train = torch.tensor(add_unknown_filler_labels(model.conditional_data_dim, len(D)-1), device=device)
            labels_val = torch.tensor(add_unknown_filler_labels(model.conditional_data_dim, len(D_val)-1), device=device)


    ae = []
    se = []

    # try:
    #     D, _ = model.pretrained_model.encoder(D, labels)
    #     D_val, _ = model.pretrained_model.encoder(D_val, labels)
    # except: AttributeError

    if len(D_val) == 0:
        print('cv 10')
        kf = KFold(n_splits=cv, random_state=seed, shuffle=True)
        X, logvar = model.encoder(X, labels)#.cuda() if torch.cuda.is_available() else model.encoder(X_train, labels_training)
        for train_index, test_index in kf.split(X):
            # if type(labels_train[0]) != type(None):
            #     labels_training = labels_train[train_index]
            #     labels_test = labels_train[test_index]
            # else:
            #     labels_training = labels_train
            #     labels_test = [None]

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if model.CVAE or model.SSCVAE:
                labels_train, labels_val = labels[train_index], labels[test_index]


            downstream_model.fit(X_train.cpu().detach().numpy(), y_train)
            pred = downstream_model.predict(X_test.cpu().detach().numpy())

            se.append(mean_squared_error(pred, y_test))
            ae.append(mean_absolute_error(pred, y_test))
    else:
        X_val = torch.stack([torch.tensor(seq, device=device) for seq in D_val['seqs']]).long()
        y_val = D_val[sheet].values
        encoded_X_train, encoded_X_train_logvar = model.encoder(X, labels_train)
        encoded_X_test, encoded_X_test_logvar = model.encoder(X_val, labels_val)
        downstream_model.fit(encoded_X_train.cpu().detach().numpy(), y)
        pred = downstream_model.predict(encoded_X_test.cpu().detach().numpy())
        se.append(mean_squared_error(pred, y_val))
        ae.append(mean_absolute_error(pred, y_val))

    mse, mae = np.mean(se), np.mean(ae)
    if len(D_val) == 0:
        mse_std, mae_std = np.std(se, ddof=1), np.std(ae, ddof=1)
    else:
        mse_std, mae_std = 0, 0
    if len(D_val)==0:
        return mse, mae, mse_std, mae_std
    else:
        return mse, mae, mse_std, mae_std

def pipeline_baseline(D, sheet, val_index, cv=10):
    mae = 0
    mse = 0

    if len(val_index) == 0:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        D = D.reset_index(drop=True)
        for train_index, test_index in kf.split(D):
            y_train, y_test = D[sheet][train_index], D[sheet][test_index]

            mse += mean_squared_error([np.mean(y_train)]*len(y_test), y_test)/cv
            mae += mean_absolute_error([np.mean(y_train)]*len(y_test), y_test)/cv
    else:
        D_val = D.loc[val_index]
        D = D.drop(val_index, axis=0)
        mse += mean_squared_error([np.mean(D[sheet])]*len(D_val), D_val[sheet])
        mae += mean_absolute_error([np.mean(D[sheet])]*len(D_val), D_val[sheet])
    return mse, mae


def pipeline_ohc_baseline(D, sheet, val_index, model, downstream_model=None, cv=10, seed=42, device='cuda'):
    np.random.seed(seed)
    assert downstream_model != None, "Please specify downstream model"

    ae = []
    se = []
    try:
        X, _ = model.pretrained_model.encoder(X, labels_train)
        X_val, _ = model.pretrained_model.encoder(X_val, labels_val)
    except: AttributeError

    if len(val_index) == 0:
        X = torch.stack([torch.tensor(seq, device=device) for seq in D['seqs']]).long()
        X = F.one_hot(X.to(torch.int64), model.alphabet_size).flatten(1).to(torch.float).cuda()
        y = D.reset_index(drop=True)[sheet]
        kf = KFold(n_splits=cv)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            downstream_model.fit(X_train.cpu().detach().numpy(), y_train)
            pred = downstream_model.predict(X_test.cpu().detach().numpy())

            se.append(mean_squared_error(pred, y_test))
            ae.append(mean_absolute_error(pred, y_test))
    else:
        D_val = D.loc[val_index]
        y_val = D_val[sheet]

        D = D.drop(val_index, axis=0)
        y = D[sheet]

        X = torch.stack([torch.tensor(seq, device=device) for seq in D['seqs']]).long()
        X = F.one_hot(X.to(torch.int64), model.alphabet_size).flatten(1).to(torch.float).cuda()
        X_val = torch.stack([torch.tensor(seq, device=device) for seq in D_val['seqs']]).long()
        X_val = F.one_hot(X_val.to(torch.int64), model.alphabet_size).flatten(1).to(torch.float).cuda()
        downstream_model.fit(X.cpu().detach().numpy(), y)
        pred = downstream_model.predict(X_val.cpu().detach().numpy())
        se.append(mean_squared_error(pred, y_val))
        ae.append(mean_absolute_error(pred, y_val))

    mse, mae = np.mean(se), np.mean(ae),

    mse_std = np.std(se, ddof=1) if len(se)>1 else 0
    mae_std = np.std(ae, ddof=1) if len(ae)>1 else 0

    return mse, mae, mse_std, mae_std


def pipeline_h1decoder_pred(D, D_val, sheet, model, downstream_model=None, cv=10, seed=42, device='cpu'):
    np.random.seed(seed)
    assert downstream_model != None, "Please specify downstream model"

    X = torch.stack([torch.tensor(seq, device=device) for seq in D['seqs']]).long()
    y = D[sheet].values

    if model.SSVAE or model.VAE: #or model.regCVAE:
        labels,labels_train, labels_val = [None], [None], [None]
    if model.SSCVAE:
        # TODO Right now if SSCVAE and multilabel > 1 then label is not used in elbo
        if model.multilabel > 1:
            labels = torch.cat([torch.zeros(D.size(0), model.predict_label(x, i=i).size(1)).cuda().scatter_(1, torch.argmax(model.predict_label(x, i=i),1,keepdim=True), 1) for i in range(model.multilabel)], dim=1)
            labels = labels.cuda() if torch.cuda.is_available() else labels
        else:
            if len(D_val) == 0:
                logits = model.predict_label(X, i=0)
                labels = torch.zeros(D.size(0), logits.size(1)).cuda().scatter_(1, torch.argmax(logits,1,keepdim=True), 1) if torch.cuda.is_available() else torch.zeros(D.size(0), logits.size(1)).scatter_(1, torch.argmax(logits,1,keepdim=True), 1)
            else:
                X_val = torch.stack([torch.tensor(seq, device=device) for seq in D_val['seqs']]).long()
                logits_t = model.predict_label(X, i=0)
                logits_v = model.predict_label(X_val, i=0)
                labels_train = torch.zeros(X.size(0), logits_t.size(1)).cuda().scatter_(1, torch.argmax(logits_t,1,keepdim=True), 1) if torch.cuda.is_available() else torch.zeros(X.size(0), logits_t.size(1)).scatter_(1, torch.argmax(logits_v,1,keepdim=True), 1)
                labels_val = torch.zeros(X_val.size(0), logits_v.size(1)).cuda().scatter_(1, torch.argmax(logits_v,1,keepdim=True), 1) if torch.cuda.is_available() else torch.zeros(X_val.size(0), logits_v.size(1)).scatter_(1, torch.argmax(logits_t,1,keepdim=True), 1)

    if model.CVAE:
        if len(D_val) == 0:
            labels = add_unknown_filler_labels(model.conditional_data_dim, len(X)-1)
            labels = torch.tensor(labels, device=device).to(torch.float)
        else:
            labels_train = torch.tensor(add_unknown_filler_labels(model.conditional_data_dim, len(D)-1), device=device)
            labels_val = torch.tensor(add_unknown_filler_labels(model.conditional_data_dim, len(D_val)-1), device=device)


    ae = []
    se = []

    if len(D_val) == 0:
        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
        for train_index, test_index in kf.split(X):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if model.CVAE or model.SSCVAE:
                labels_train, labels_val = labels[train_index], labels[test_index]

            encoded_X_train, encoded_X_train_var = model.encoder(X_train, labels_train)#.cuda() if torch.cuda.is_available() else model.encoder(X_train, labels_training)
            encoded_X_test, encoded_X_test_var = model.encoder(X_test, labels_val)#.cuda() if torch.cuda.is_available() else model.encoder(X_test, labels_test)

            qz_x_train = Normal(encoded_X_train, encoded_X_train_var.mul(0.5).exp())
            qz_x_test = Normal(encoded_X_test, encoded_X_test_var.mul(0.5).exp())

            z_train = qz_x_train.rsample((1,))
            z_test = qz_x_test.rsample((1,))

            h1_train, _, _, _ = model.decoder(z_train.flatten(0, 1), labels_train)
            h1_test, _, _, _ = model.decoder(z_test.flatten(0, 1), labels_val)

            downstream_model.fit(h1_train.cpu().detach().numpy(), y_train)
            pred = downstream_model.predict(h1_test.cpu().detach().numpy())

            se.append(mean_squared_error(pred, y_test))
            ae.append(mean_absolute_error(pred, y_test))
    else:
        X_val = torch.stack([torch.tensor(seq, device=device) for seq in D_val['seqs']]).long()
        y_val = D_val[sheet].values
        encoded_X_train, encoded_X_train_var = model.encoder(X, labels_train)
        encoded_X_test, encoded_X_test_var = model.encoder(X_val, labels_val)
        qz_x_train = Normal(encoded_X_train, encoded_X_train_var.mul(0.5).exp())
        qz_x_test = Normal(encoded_X_test, encoded_X_test_var.mul(0.5).exp())

        z_train = qz_x_train.rsample((1,))
        z_test = qz_x_test.rsample((1,))

        h1_train, _, _, _ = model.decoder(z_train.flatten(0, 1), labels_train)
        h1_test, _, _, _ = model.decoder(z_test.flatten(0, 1), labels_val)

        downstream_model.fit(h1_train.cpu().detach().numpy(), y)
        pred = downstream_model.predict(h1_test.cpu().detach().numpy())
        se.append(mean_squared_error(pred, y_val))
        ae.append(mean_absolute_error(pred, y_val))

    mse, mae = np.mean(se), np.mean(ae)
    if len(D_val) == 0:
        mse_std, mae_std = np.std(se, ddof=1), np.std(ae, ddof=1)
    else:
        mse_std, mae_std = 0, 0

    return mse, mae, mse_std, mae_std



def pipeline_h2decoder_pred(D, D_val, sheet, model, downstream_model=None, cv=10, seed=42, device='cpu'):
    np.random.seed(seed)
    assert downstream_model != None, "Please specify downstream model"

    X = torch.stack([torch.tensor(seq, device=device) for seq in D['seqs']]).long()
    y = D[sheet].values

    if model.SSVAE or model.VAE: #or model.regCVAE:
        labels,labels_train, labels_val = [None], [None], [None]
    if model.SSCVAE:
        # TODO Right now if SSCVAE and multilabel > 1 then label is not used in elbo
        if model.multilabel > 1:
            labels = torch.cat([torch.zeros(D.size(0), model.predict_label(x, i=i).size(1)).cuda().scatter_(1, torch.argmax(model.predict_label(x, i=i),1,keepdim=True), 1) for i in range(model.multilabel)], dim=1)
            labels = labels.cuda() if torch.cuda.is_available() else labels
        else:
            if len(D_val) == 0:
                logits = model.predict_label(X, i=0)
                labels = torch.zeros(D.size(0), logits.size(1)).cuda().scatter_(1, torch.argmax(logits,1,keepdim=True), 1) if torch.cuda.is_available() else torch.zeros(D.size(0), logits.size(1)).scatter_(1, torch.argmax(logits,1,keepdim=True), 1)
            else:
                X_val = torch.stack([torch.tensor(seq, device=device) for seq in D_val['seqs']]).long()
                logits_t = model.predict_label(X, i=0)
                logits_v = model.predict_label(X_val, i=0)
                labels_train = torch.zeros(X.size(0), logits_t.size(1)).cuda().scatter_(1, torch.argmax(logits_t,1,keepdim=True), 1) if torch.cuda.is_available() else torch.zeros(X.size(0), logits_t.size(1)).scatter_(1, torch.argmax(logits_v,1,keepdim=True), 1)
                labels_val = torch.zeros(X_val.size(0), logits_v.size(1)).cuda().scatter_(1, torch.argmax(logits_v,1,keepdim=True), 1) if torch.cuda.is_available() else torch.zeros(X_val.size(0), logits_v.size(1)).scatter_(1, torch.argmax(logits_t,1,keepdim=True), 1)

    if model.CVAE:
        if len(D_val) == 0:
            labels = add_unknown_filler_labels(model.conditional_data_dim, len(X)-1)
            labels = torch.tensor(labels, device=device).to(torch.float)
        else:
            labels_train = torch.tensor(add_unknown_filler_labels(model.conditional_data_dim, len(D)-1), device=device)
            labels_val = torch.tensor(add_unknown_filler_labels(model.conditional_data_dim, len(D_val)-1), device=device)


    ae = []
    se = []

    if len(D_val) == 0:
        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
        for train_index, test_index in kf.split(X):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if model.CVAE or model.SSCVAE:
                labels_train, labels_val = labels[train_index], labels[test_index]

            encoded_X_train, encoded_X_train_var = model.encoder(X_train, labels_train)#.cuda() if torch.cuda.is_available() else model.encoder(X_train, labels_training)
            encoded_X_test, encoded_X_test_var = model.encoder(X_test, labels_val)#.cuda() if torch.cuda.is_available() else model.encoder(X_test, labels_test)

            qz_x_train = Normal(encoded_X_train, encoded_X_train_var.mul(0.5).exp())
            qz_x_test = Normal(encoded_X_test, encoded_X_test_var.mul(0.5).exp())

            z_train = qz_x_train.rsample((1,))
            z_test = qz_x_test.rsample((1,))

            _, h2_train, _, _ = model.decoder(z_train.flatten(0, 1), labels_train)
            _, h2_test, _, _ = model.decoder(z_test.flatten(0, 1), labels_val)

            downstream_model.fit(h2_train.cpu().detach().numpy(), y_train)
            pred = downstream_model.predict(h2_test.cpu().detach().numpy())

            se.append(mean_squared_error(pred, y_test))
            ae.append(mean_absolute_error(pred, y_test))
    else:
        X_val = torch.stack([torch.tensor(seq, device=device) for seq in D_val['seqs']]).long()
        y_val = D_val[sheet].values
        encoded_X_train, encoded_X_train_var = model.encoder(X, labels_train)
        encoded_X_test, encoded_X_test_var = model.encoder(X_val, labels_val)
        qz_x_train = Normal(encoded_X_train, encoded_X_train_var.mul(0.5).exp())
        qz_x_test = Normal(encoded_X_test, encoded_X_test_var.mul(0.5).exp())

        z_train = qz_x_train.rsample((1,))
        z_test = qz_x_test.rsample((1,))

        _, h2_train, _, _ = model.decoder(z_train.flatten(0, 1), labels_train)
        _, h2_test, _, _ = model.decoder(z_test.flatten(0, 1), labels_val)

        downstream_model.fit(h2_train.cpu().detach().numpy(), y)
        pred = downstream_model.predict(h2_test.cpu().detach().numpy())
        se.append(mean_squared_error(pred, y_val))
        ae.append(mean_absolute_error(pred, y_val))

    mse, mae = np.mean(se), np.mean(ae)
    if len(D_val) == 0:
        mse_std, mae_std = np.std(se, ddof=1), np.std(ae, ddof=1)
    else:
        mse_std, mae_std = 0, 0

    return mse, mae, mse_std, mae_std


def kfold_pred(model, X, y, cv=10, seed=42):
    mae_list = []
    mse_list = []
    kf = KFold(n_splits=cv, random_state=seed, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae_list.append(mean_absolute_error(pred, y_test))
        mse_list.append(mean_squared_error(pred, y_test))

    return np.mean(mse_list), np.mean(mae_list), np.std(mse_list, ddof=1), np.std(mae_list, ddof=1)
