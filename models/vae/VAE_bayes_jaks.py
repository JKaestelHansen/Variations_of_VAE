"""
Created on Tue Feb 18 13:13:07 2020

@author: JAKS
"""
import torch
from torch import nn
from ..utils import smooth_one_hot
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from torch.nn import functional as F
import itertools
import time


def generate_all_labels(x, y_dim):
    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return generated.float()


def multilabel_generate_permutations(x, label_dim, multilabel):
    def generate_all_labels(x, y_dim):
        def batch(batch_size, label):
            labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
            y = torch.zeros((batch_size, y_dim))
            y.scatter_(1, labels, 1)
            return y.type(torch.LongTensor)
        generated = torch.cat([batch(1, i) for i in range(y_dim)])

        if x.is_cuda:
            generated = generated.cuda()

        return generated.float()
    batch_size = x.size(0)
    permuts = [list(p) for p in itertools.product(list(range(label_dim)), repeat=multilabel)]
    labels = torch.stack([generate_all_labels(x, label_dim)[per].view(1, -1).squeeze(0) for per in permuts])
    if x.is_cuda:
        return torch.repeat_interleave(labels, batch_size, dim=0).cuda().float()
    else:
        return torch.repeat_interleave(labels, batch_size, dim=0)


def sample_gaussian(mu, logsigma): #reparametrization trick
    std = torch.exp(logsigma)
    eps = torch.randn_like(std)
    return mu + eps * std


def KLdivergence(mu, sigma):
    return 0.5*torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)


def log_standard_categorical(p):
    with torch.no_grad():
        prior = F.softmax(torch.ones_like(p), dim=1)
        cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)
    return cross_entropy


def idx_for_each_label(labels, multilabel, num_label_cols):
    keep_idx = []
    cols_per_label = num_label_cols // multilabel
    for i in range(multilabel):
        if cols_per_label > 1:
            y_val = labels[:,i*cols_per_label:i*cols_per_label+cols_per_label]
        else:
            y_val =labels[:,i]
        keep_idx.append(torch.unique(torch.where(y_val>=0)[0]))
    return keep_idx


def cross_entropy_loss(pred, y):
    return -torch.sum(y * torch.log(pred+1e-11), dim=1).mean()


class Classifier(nn.Module):
    def __init__(self, layer_size):
        super(Classifier, self).__init__()
        [input_dim, h_dim, y_dim] = layer_size
        self.dense = nn.Linear(input_dim, h_dim)
        nn.init.xavier_normal_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0.1)

        self.logits = nn.Linear(h_dim, y_dim)
        nn.init.xavier_normal_(self.logits.weight)
        nn.init.constant_(self.logits.bias, 0.1)

        self.network = nn.Sequential(self.dense,
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     self.logits,
                                     nn.Softmax(dim=1))

    def forward(self, x):
        x = self.network(x)
        return x

class Regressor(nn.Module):
    def __init__(self, layer_size):
        super(Regressor, self).__init__()
        [input_dim, h_dim, y_dim] = layer_size
        self.dense = nn.Linear(input_dim, h_dim)
        nn.init.xavier_normal_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0.1)

        self.logits = nn.Linear(h_dim, y_dim)
        nn.init.xavier_normal_(self.logits.weight)
        nn.init.constant_(self.logits.bias, 0.1)

        self.network = nn.Sequential(self.dense,
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     self.logits)

    def forward(self, x):
        x = self.network(x)
        return x


class VAE_bayes(nn.Module):
    """
    We differentiate between SSVAE and SSCVAE/CVAE bc they have discretized labels
    """
    def __init__(self,
                 layer_sizes,
                 alphabet_size,
                 z_samples = 1,
                 dropout = 0.5,
                 use_bayesian = True,
                 num_patterns = 4,
                 cw_inner_dimension = 40,
                 use_param_loss = True,
                 use_sparse_interactions = False,
                 rws = False,
                 conditional_data_dim=False,
                 SSVAE = False,
                 SSCVAE = False,
                 CVAE = False,
                 regCVAE = False,
                 VAE=True,
                 multilabel=0,
                 seq2yalpha=0,
                 z2yalpha=0,
                 label_pred_layer_sizes = None,
                 pred_from_latent=False,
                 pred_from_seq=False,
                 warm_up = 0,
                 batchnorm = False,
                 device='cpu'):

        super(VAE_bayes, self).__init__()

        self.layer_sizes = layer_sizes
        self.alphabet_size = alphabet_size
        self.max_sequence_length = self.layer_sizes[0] // self.alphabet_size
        self.cw_inner_dimension  = cw_inner_dimension
        self.device = device
        self.dropout = dropout
        self.nb_patterns = num_patterns
        self.dec_h2_dim = layer_sizes[-2]
        self.z_samples = z_samples
        self.use_bayesian = use_bayesian
        self.use_param_loss = use_param_loss
        self.use_sparse_interactions = use_sparse_interactions
        self.rws = rws
        self.warm_up = warm_up
        self.warm_up_scale = 0
        self.conditional_data_dim = conditional_data_dim
        self.multilabel = multilabel
        self.label_pred_layer_sizes = label_pred_layer_sizes
        self.VAE = VAE
        self.SSVAE = SSVAE
        self.SSCVAE = SSCVAE
        self.CVAE = CVAE
        self.regCVAE = regCVAE
        self.pred_from_latent = pred_from_latent
        self.pred_from_seq = pred_from_seq
        self.seq2yalpha = seq2yalpha
        self.z2yalpha = z2yalpha
        self.batchnorm = batchnorm
        self.device = device

        if SSVAE:
            self.criterion = nn.MSELoss()
        if SSCVAE:
            self.criterion = cross_entropy_loss

        self.latent_dim = layer_sizes.index(min(layer_sizes))

        # Encoder
        # encoder neural network prior to mu and sigma
        if CVAE or SSCVAE and multilabel<2:
            input_dim = self.max_sequence_length*self.alphabet_size+self.conditional_data_dim
        else:
            input_dim = self.max_sequence_length*alphabet_size

        self.enc_fc1 = nn.Linear(input_dim, layer_sizes[1])
        nn.init.xavier_normal_(self.enc_fc1.weight)
        nn.init.constant_(self.enc_fc1.bias, 0.1)

        self.enc_fc2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        nn.init.xavier_normal_(self.enc_fc2.weight)
        nn.init.constant_(self.enc_fc2.bias, 0.1)

        if batchnorm:
            self.encode_layers = nn.Sequential(self.enc_fc1,
                                               nn.BatchNorm1d(layer_sizes[1]),
                                               nn.ReLU(inplace = True),
                                               self.enc_fc2,
                                               nn.BatchNorm1d(layer_sizes[2]),
                                               nn.ReLU(inplace = True)
                                                    )
        else:
            self.encode_layers = nn.Sequential(self.enc_fc1,
                                               nn.ReLU(inplace = True),
                                               self.enc_fc2,
                                               nn.ReLU(inplace = True)
                                                    )

        # encode mu from h_dim to z_dim deterministically (reparameterization trick)
        self.encode_mu = nn.Linear(layer_sizes[2], layer_sizes[3])
        nn.init.xavier_normal_(self.encode_mu.weight)
        nn.init.constant_(self.encode_mu.bias, 0.1)

        # encode sigma from h_dim to z_dim deterministically (reparameterization trick)
        self.encode_logsigma = nn.Linear(layer_sizes[2], layer_sizes[3])
        nn.init.xavier_normal_(self.encode_logsigma.weight)
        nn.init.constant_(self.encode_logsigma.bias, -10)

        # Decoder
        # weights layer 1
        # TODO decide how to deal with issue of partly labelling
        if CVAE or SSCVAE and multilabel<2:
            z_dim = layer_sizes[3] + self.conditional_data_dim
        else:
            z_dim = layer_sizes[3]

        # if sparse interactions and other tricks are enabled initiate variables
        if self.use_sparse_interactions:
            # sparse interaction and dict from deepsequence paper
            self.mu_S = nn.Parameter(torch.Tensor(int(layer_sizes[5] / self.nb_patterns), self.max_sequence_length))
            nn.init.zeros_(self.mu_S)
            self.logsigma_S = nn.Parameter(torch.Tensor(int(layer_sizes[5] / self.nb_patterns), self.max_sequence_length))
            nn.init.constant_(self.logsigma_S, -10)
            self.mu_C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size))
            nn.init.xavier_normal_(self.mu_C)
            self.logsigma_C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size))
            nn.init.constant_(self.logsigma_C, -10)

            # inverse temperature term from deepsequence paper
            self.mu_l = nn.Parameter(torch.Tensor([1]))
            self.logsigma_l = nn.Parameter(torch.Tensor([-10.0]))

            # alter output shape for sparse interactions
            W3_output_shape = cw_inner_dimension * self.max_sequence_length

        if not self.use_sparse_interactions:
            W3_output_shape = alphabet_size * self.max_sequence_length

        # bayesian network
        if self.use_bayesian:
            self.mu_W1_dec = nn.Parameter(torch.Tensor(z_dim, layer_sizes[4]))
            nn.init.xavier_normal_(self.mu_W1_dec)
            self.logsigma_W1_dec = nn.Parameter(torch.Tensor(z_dim, layer_sizes[4]))
            nn.init.constant_(self.logsigma_W1_dec, -10)
            self.mu_b1_dec = nn.Parameter(torch.Tensor(layer_sizes[4]))
            nn.init.constant_(self.mu_b1_dec, 0.1)
            self.logsigma_b1_dec = nn.Parameter(torch.Tensor(layer_sizes[4]))
            nn.init.constant_(self.logsigma_b1_dec, -10)

            # weights layer 2
            self.mu_W2_dec = nn.Parameter(torch.Tensor(layer_sizes[4], layer_sizes[5]))
            nn.init.xavier_normal_(self.mu_W2_dec)
            self.logsigma_W2_dec = nn.Parameter(torch.Tensor(layer_sizes[4], layer_sizes[5]))
            nn.init.constant_(self.logsigma_W2_dec, -10)
            self.mu_b2_dec = nn.Parameter(torch.Tensor(layer_sizes[5]))
            nn.init.constant_(self.mu_b2_dec, 0.1)
            self.logsigma_b2_dec = nn.Parameter(torch.Tensor(layer_sizes[5]))
            nn.init.constant_(self.logsigma_b2_dec, -10)

            # weights layer 3
            self.mu_W3_dec = nn.Parameter(torch.Tensor(layer_sizes[5], W3_output_shape))
            nn.init.xavier_normal_(self.mu_W3_dec)
            self.logsigma_W3_dec = nn.Parameter(torch.Tensor(layer_sizes[5], W3_output_shape))
            nn.init.constant_(self.logsigma_W3_dec, -10)
            self.mu_b3_dec = nn.Parameter(torch.Tensor(alphabet_size * self.max_sequence_length))
            nn.init.constant_(self.mu_b3_dec, 0.1)
            self.logsigma_b3_dec = nn.Parameter(torch.Tensor(alphabet_size * self.max_sequence_length))
            nn.init.constant_(self.logsigma_b3_dec, -10)


        # non-bayesian network
        h1_decoder_layers = []
        h2_decoder_layers = []
        if not self.use_bayesian:
            # decoder
            self.h1_dec = nn.Linear(z_dim, layer_sizes[4])
            nn.init.xavier_normal_(self.h1_dec.weight)
            nn.init.constant_(self.h1_dec.bias, 0.1)
            h1_decoder_layers.append(self.h1_dec)
            if batchnorm:
                h1_decoder_layers.append(nn.BatchNorm1d(layer_sizes[4]))
            h1_decoder_layers.append(nn.ReLU())
            if not batchnorm:
                h1_decoder_layers.append(nn.Dropout(self.dropout))

            self.h2_dec = nn.Linear(layer_sizes[4], layer_sizes[5])
            nn.init.xavier_normal_(self.h2_dec.weight)
            nn.init.constant_(self.h2_dec.bias, 0.1)
            h2_decoder_layers.append(self.h2_dec)
            h2_decoder_layers.append(nn.Sigmoid())
            if batchnorm:
                h2_decoder_layers.append(nn.BatchNorm1d(layer_sizes[5]))
            if not batchnorm:
                h2_decoder_layers.append(nn.Dropout(self.dropout))

            self.h3_dec = nn.Linear(layer_sizes[5], W3_output_shape, bias = False)
            nn.init.xavier_normal_(self.h3_dec.weight)
            self.b3 = nn.Parameter(torch.Tensor(alphabet_size * self.max_sequence_length))
            nn.init.constant_(self.b3, -10)

        self.h1_decoder_network = nn.Sequential(*h1_decoder_layers)
        self.h2_decoder_network = nn.Sequential(*h2_decoder_layers)
        if self.pred_from_seq:
            if SSCVAE:
                self.label_pred_model = nn.ModuleList([Classifier([layer_sizes[0], label_pred_layer_sizes[0], 3]) for i in range(multilabel)])
            if SSVAE:
                self.label_pred_model = nn.ModuleList([Regressor([layer_sizes[0], label_pred_layer_sizes[0], 1]) for i in range(multilabel)])
        if pred_from_latent:
            if SSCVAE:
                self.z_label_pred = nn.ModuleList([Classifier([layer_sizes[3], label_pred_layer_sizes[0], 3]) for i in range(multilabel)])
            else:
                self.z_label_pred = nn.ModuleList([Regressor([layer_sizes[3], label_pred_layer_sizes[0], 1]) for i in range(multilabel)])

        self.to(device)

    def predict_label(self, x, i=0):
        x = F.one_hot(x.to(torch.int64), self.alphabet_size).flatten(1).to(torch.float).cuda() if self.device=='cuda' else F.one_hot(x.to(torch.int64), self.alphabet_size).flatten(1).to(torch.float)
        assert torch.isfinite(x).all() == True
        return self.label_pred_model[i](x)

    def predict_label_from_z(self, x, i=0):
        return self.z_label_pred[i](x)

    def encoder(self, x, labels, device=None):
        x = F.one_hot(x.to(torch.int64), self.alphabet_size).flatten(1).to(torch.float)
        if self.CVAE or self.SSCVAE and self.multilabel<2:
            x = torch.cat((x, labels.to(torch.float)), 1)
        h = self.encode_layers(x)
        mu = self.encode_mu(h)
        logvar = self.encode_logsigma(h)
        return mu+1e-6, logvar+1e-6

    def decoder(self, z, labels):
        if self.CVAE or self.SSCVAE and self.multilabel<2:
            if len(z.size())>=3:
                event_dim, row_dim, col1_dim = z.size(0), z.size(1), z.size(2)
                col2_dim = labels.size(2)
                z = z.flatten(0,1)
                labels = labels.flatten(0,1)
                z = torch.cat((z, labels.to(torch.float)), 1)
                z = z.view(event_dim, row_dim, col1_dim+col2_dim)
            else:
                z = torch.cat((z, labels.to(torch.float)), 1)
        # if bayesian sample new weights
        if self.use_bayesian:
            # weights and bias for layer 1
            W1 = sample_gaussian(self.mu_W1_dec, self.logsigma_W1_dec)
            b1 = sample_gaussian(self.mu_b1_dec, self.logsigma_b1_dec)
            W2 = sample_gaussian(self.mu_W2_dec, self.logsigma_W2_dec)
            b2 = sample_gaussian(self.mu_b2_dec, self.logsigma_b2_dec)
            W3 = sample_gaussian(self.mu_W3_dec, self.logsigma_W3_dec)
            b3 = sample_gaussian(self.mu_b3_dec, self.logsigma_b3_dec)
        else:
            W3 = self.h3_dec.weight
            b3 = self.b3

        # if sparse interactions perform linear operations
        if self.use_sparse_interactions:
            l = sample_gaussian(self.mu_l, self.logsigma_l)
            S = sample_gaussian(self.mu_S, self.logsigma_S)
            C = sample_gaussian(self.mu_C, self.logsigma_C)
            S = torch.sigmoid(S.repeat(self.nb_patterns, 1))
            W3 = W3.view(self.dec_h2_dim * self.max_sequence_length, -1)
            W_out = W3 @ C
            W_out = W_out.view(-1, self.max_sequence_length, self.alphabet_size)
            W_out = W_out * S.unsqueeze(2)
            W_out = W_out.view(-1, self.max_sequence_length * self.alphabet_size)
        if not self.use_sparse_interactions and not self.use_bayesian:
            W_out = W3.t()
        if not self.use_sparse_interactions and self.use_bayesian:
            W_out = W3

        if self.use_bayesian:
            h1 = nn.functional.relu(nn.functional.linear(z, W1.t(), b1))
            h2 = torch.sigmoid(nn.functional.linear(h1, W2.t(), b2))

        if not self.use_bayesian:
            h1 = self.h1_decoder_network(z)
            h2 = self.h2_decoder_network(h1)

        h3 = nn.functional.linear(h2, W_out.t(), b3)
        if self.use_sparse_interactions:
            h3 = h3 * torch.log(1 + l.exp())

        h3 = h3.view((-1, self.max_sequence_length, self.alphabet_size))
        px_z = torch.distributions.Categorical(logits=h3)
        h3 = nn.functional.log_softmax(h3, -1)

        return h1, h2, h3, px_z

    def recon_loss(self, recon_x, x):
        # How well do input x and output recon_x agree?
        recon_x = recon_x.view(self.z_samples, -1, recon_x.size(1), self.alphabet_size).permute(1, 2, 0, 3)
        x = x.unsqueeze(-1).expand(-1, -1, self.z_samples)

        smooth_target = smooth_one_hot(x, self.alphabet_size)
        loss = -(smooth_target * recon_x).sum(-1)
        loss = loss.mean(-1).sum(-1)
        return loss


    def kld_loss(self, encoded_distribution):
        prior = Normal(torch.zeros_like(encoded_distribution.mean), torch.ones_like(encoded_distribution.variance))
        kld = kl_divergence(encoded_distribution, prior).sum(dim = 1)

        return kld

    def protein_logp(self, x, labels):
        # correctly incorporate pred loss in elbo
        if self.SSVAE or self.SSCVAE:
            if self.SSVAE:
                labels = [None]
            if self.SSCVAE:
                # TODO Right now if SSCVAE and multilabel > 1 then label is not used in elbo
                if self.multilabel > 1:
                    labels = torch.cat([torch.zeros(x.size(0), self.predict_label(x, i=i).size(1)).cuda().scatter_(1, torch.argmax(self.predict_label(x, i=i),1,keepdim=True), 1) for i in range(self.multilabel)], dim=1)
                    labels = labels.cuda() if torch.cuda.is_available() else labels
                else:
                    logits = self.predict_label(x, i=0)
                    labels = torch.zeros(x.size(0), logits.size(1)).cuda().scatter_(1, torch.argmax(logits,1,keepdim=True), 1) if torch.cuda.is_available() else torch.zeros(x.size(0), logits.size(1)).scatter_(1, torch.argmax(logits,1,keepdim=True), 1)
                # prior is just a constant
                #prior = -log_standard_categorical(labels)
            mu, logvar = self.encoder(x, labels)
            encoded_distribution = Normal(mu, logvar.mul(0.5).exp())
            z = encoded_distribution.rsample((self.z_samples,))
            h1, h2, recon_x, px_z = self.decoder(z.flatten(0, 1), labels)
            recon_loss = self.recon_loss(recon_x, x)
            kld_loss = self.kld_loss(encoded_distribution)
            param_kld = torch.zeros(1) + 1e-5
            elbo = recon_loss + kld_loss
            logp = recon_loss
            kld = kld_loss

            # SSCVAE does trick with unlabelled and so will be done here
            # TODO decided not to calc entropy and all that as we decide a most probable label and therefore it is known
            # if self.SSCVAE:
            #     l = -L
            #     # Auxiliary classification loss q(y|x) in Regular cross entropy
            #     H = 0
            #     L = 0
            #     for i in range(self.multilabel):
            #         logits = self.predict_label(unlabelled_x, i=i)
            #         # Calculate entropy H(q(y|x)) and sum over all labels
            #         H += torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)
            #         L += torch.sum(torch.mul(logits, l), dim=-1)
            #     U = -torch.mean(L + H)
            #
            # # SSVAE does nothing with unlabelled and hence nothing is done
            # if self.SSVAE:
            #     elbo = L.squeeze(1)

        if self.VAE or self.CVAE:
            mu, logvar = self.encoder(x, labels)
            encoded_distribution = Normal(mu, logvar.mul(0.5).exp())
            kld = self.kld_loss(encoded_distribution)
            z = encoded_distribution.rsample()
            recon_x = self.decoder(z, labels)[2].permute(0, 2, 1)
            logp = F.nll_loss(recon_x, x, reduction = "none").mul(-1).sum(1)
            elbo = logp + kld

        # amino acid probabilities are independent conditioned on z
        return elbo, logp, kld


    def global_parameter_kld(self):
        global_kld = 0

        global_kld += -KLdivergence(self.mu_W1_dec, self.logsigma_W1_dec.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_b1_dec, self.logsigma_b1_dec.mul(0.5).exp())

        global_kld += -KLdivergence(self.mu_W2_dec, self.logsigma_W2_dec.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_b2_dec, self.logsigma_b2_dec.mul(0.5).exp())

        global_kld += -KLdivergence(self.mu_W3_dec, self.logsigma_W3_dec.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_b3_dec, self.logsigma_b3_dec.mul(0.5).exp())

        if not self.use_sparse_interactions:
            return global_kld

        global_kld += -KLdivergence(self.mu_C, self.logsigma_C.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_S, self.logsigma_S.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_l, self.logsigma_l.mul(0.5).exp())

        return global_kld


    def global_parameter_kld1(self):
        global_kld = 0

        global_kld += torch.sum(self.kld_loss(Normal(self.mu_W1_dec, self.logsigma_W1_dec.mul(0.5).exp())))
        global_kld += -KLdivergence(self.mu_b1_dec, self.logsigma_b1_dec.mul(0.5).exp())

        global_kld += -KLdivergence(self.mu_W2_dec, self.logsigma_W2_dec.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_b2_dec, self.logsigma_b2_dec.mul(0.5).exp())

        global_kld += -KLdivergence(self.mu_W3_dec, self.logsigma_W3_dec.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_b3_dec, self.logsigma_b3_dec.mul(0.5).exp())

        if not self.use_sparse_interactions:
            return global_kld

        global_kld += -KLdivergence(self.mu_C, self.logsigma_C.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_S, self.logsigma_S.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_l, self.logsigma_l.mul(0.5).exp())

        return global_kld


    def vae_loss(self, x, labels, neff, weights):
        if self.CVAE or self.VAE or self.regCVAE:
            mu, logvar = self.encoder(x, labels)
            encoded_distribution = Normal(mu, logvar.mul(0.5).exp())
            z = encoded_distribution.rsample((self.z_samples,))
            h1, h2, recon_x, px_z = self.decoder(z.flatten(0, 1), labels)
            recon_loss = self.recon_loss(recon_x, x)
            kld_loss = self.kld_loss(encoded_distribution)
            if self.rws:
                recon_kld_loss = torch.mean(weights*(recon_loss + kld_loss))
            else:
                recon_kld_loss = torch.mean(recon_loss + kld_loss)

            if self.use_bayesian and self.use_param_loss:
                param_kld = self.global_parameter_kld1() / neff
                total_loss = recon_kld_loss + param_kld
            else:
                param_kld = torch.zeros(1) + 1e-5
                total_loss = recon_kld_loss
            return total_loss, recon_loss.mean().item(), kld_loss.mean().item(), param_kld.item(), encoded_distribution, px_z

        if self.SSVAE or self.SSCVAE:
            label_cols = x.size(1)
            x = torch.cat([x.float(), labels], dim=1)
            # separate rows with and without label
            cond = torch.isnan(x[:, label_cols:]) != True
            labelled_data = x[cond.any(1), :]
            unlabelled_data = x[~cond.any(1), :]
            if self.SSVAE:
                weights_l = weights[cond.squeeze(1)]
                weights_u = weights[~cond.squeeze(1)]
            if self.SSCVAE:
                w_cond = cond.sum(1)//cond.size(1)
                weights_l = weights[w_cond==1]
                weights_u = weights[w_cond==0]

            if labelled_data.size(0) > 0:
                x_labelled = labelled_data[:,:label_cols].long()
                y = labelled_data[:,label_cols:]
                if self.SSCVAE:
                    # TODO decide how to handle this problem of partly labelled seqs produce NaNs
                    if self.multilabel>1:
                        labels = [None]
                    else:
                        labels = y
                if self.SSVAE:
                    labels = [None]
                # labelled loss
                # Elbo
                mu_l, logvar_l = self.encoder(x_labelled, labels)
                encoded_distribution_l = Normal(mu_l, logvar_l.mul(0.5).exp())
                z_l = encoded_distribution_l.rsample((self.z_samples,))
                h1_l, h2_l, recon_x_l, px_z_l = self.decoder(z_l.flatten(0, 1), labels)
                recon_loss_l = self.recon_loss(recon_x_l, x_labelled) if not self.rws else self.recon_loss(recon_x_l, x_labelled) * weights_l
                kld_loss_l = self.kld_loss(encoded_distribution_l) if not self.rws else self.kld_loss(encoded_distribution_l) * weights_l
                # prior is just a constant
                # add prior on y
                # TODO decided not to use prior calc a prior on each label and have it interchangeable between normal and log Categorical
                # for ssvae evt standardize labels and have that as prior
                prior_l = 0
                if self.SSCVAE:
                    for l in labels:
                        prior_l += log_standard_categorical(y)
                # if self.SSVAE:
                #     for l in labels:
                #         prior_l += log_standard_categorical(y)

                labelled_loss = recon_loss_l + kld_loss_l + prior_l if self.SSCVAE else recon_loss_l + kld_loss_l
                L = torch.mean(labelled_loss)
                # Auxiliary classification loss q(y|x) in Regular cross entropy
                classication_loss = 0
                label_loss_z = 0
                # WIP TODO labels have incorrect shape
                if self.multilabel>1:
                    keep_idx = idx_for_each_label(y, self.multilabel, y.size(1))
                    for i in range(self.multilabel):
                        if self.SSCVAE:
                            num_cols = self.conditional_data_dim//self.multilabel
                            y_true = y[keep_idx[i]][:,i*num_cols:i*num_cols+num_cols]
                        else:
                            y_true = y[keep_idx[i]][:,i].unsqueeze(1)
                        assert torch.isfinite(y_true).all() == True, "target is not finite"
                        if self.pred_from_seq:
                            pred = self.predict_label(x_labelled[keep_idx[i]][:,:label_cols], i=i)
                            assert torch.isfinite(pred).all() == True, "pred is not finite"
                            classication_loss += self.criterion(pred, y_true)
                        if self.pred_from_latent:
                            pred_from_z = self.predict_label_from_z(mu_l[keep_idx[i]], i=i)
                            label_loss_z += self.criterion(pred_from_z, y_true)
                else:
                    if self.pred_from_seq:
                        pred = self.predict_label(x_labelled, i=0)
                        classication_loss +=  self.criterion(pred, y)
                    if self.pred_from_latent:
                        pred_from_z = self.predict_label_from_z(mu_l, i=0)
                        label_loss_z += self.criterion(pred_from_z, y)
                L_a = L.clone()
                # add classification loss and divide by number of labels used
                if self.pred_from_seq:
                    L_a += self.seq2yalpha * classication_loss / self.multilabel#(x_labelled.size(0)*self.multilabel)
                if self.pred_from_latent:
                    L_a += self.z2yalpha * label_loss_z / self.multilabel#(x_labelled.size(0)*self.multilabel)
                #print(self.seq2yalpha * classication_loss/ (x_labelled.size(0)*self.multilabel), self.z2yalpha * label_loss_z/ (x_labelled.size(0)*self.multilabel))

            else:
                L_a = 0
                classication_loss = 0
                label_loss_z = 0
                recon_loss_l = torch.tensor([0]).float()
                kld_loss_l = torch.tensor([0]).float()
                encoded_distribution_l = 0
                px_z_l = 0
                x_labelled = None


            # unlabelled loss
            if unlabelled_data.size(0) > 0:
                # generate all labels for each data point to sum out label later
                if self.SSCVAE:
                    if self.multilabel > 1:
                         label_dim = (x.size(1)-label_cols)//self.multilabel
                         labels = multilabel_generate_permutations(unlabelled_data, label_dim, self.multilabel)
                    else:
                         labels = generate_all_labels(unlabelled_data, x.size(1)-label_cols)
                    prior_u = log_standard_categorical(labels)

                    unlabelled_x = unlabelled_data[:,:label_cols].repeat(labels.size(0)//unlabelled_data.size(0), 1).long()
                    weights_u = weights_u.repeat(labels.size(0)//unlabelled_data.size(0))
                    # add prior on y
                if self.SSVAE:
                    labels = [None]
                    unlabelled_x = unlabelled_data[:,:label_cols].long()
                    # prior_u = 0
                # Elbo
                mu_u, logvar_u = self.encoder(unlabelled_x, labels)
                encoded_distribution_u = Normal(mu_u, logvar_u.mul(0.5).exp())
                z_u = encoded_distribution_u.rsample((self.z_samples,))
                h1u, h2u, recon_x_u, px_z_u = self.decoder(z_u.flatten(0, 1), labels)
                recon_loss_u = self.recon_loss(recon_x_u, unlabelled_x) if not self.rws else self.recon_loss(recon_x_u, unlabelled_x) * weights_u
                kld_loss_u = self.kld_loss(encoded_distribution_u) if not self.rws else self.kld_loss(encoded_distribution_u) * weights_u
                L = recon_loss_u + kld_loss_u + prior_u if self.SSCVAE else recon_loss_u + kld_loss_u
                if self.SSCVAE:
                    l = -L.unsqueeze(1)
                    # Auxiliary classification loss q(y|x) in Regular cross entropy
                    H = 0
                    L = 0
                    Hz = 0
                    Lz = 0
                    for i in range(self.multilabel):
                        logits = self.predict_label(unlabelled_x, i=i)
                        # Calculate entropy H(q(y|x)) and sum over all labels
                        H += torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)
                        L += torch.sum(torch.mul(logits, l), dim=-1)

                        if self.pred_from_latent:
                            logits = self.predict_label_from_z(mu_u, i=i)
                            # Calculate entropy H(q(y|x)) and sum over all labels
                            Hz += torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)
                            Lz += torch.sum(torch.mul(logits, l), dim=-1)
                    U = -torch.mean(L + H) if not self.pred_from_latent else -torch.mean(L + H + Lz + Hz)
                if self.SSVAE:
                    U = torch.mean(L)
            else:
                U = 0
                recon_loss_u = torch.tensor([0]).float()+1e-6
                kld_loss_u = torch.tensor([0]).float()+1e-6
                encoded_distribution_u = 0
                px_z_u = 0
                unlabelled_x = None
            total_loss = U + L_a
            #print('size', labelled_data.size(0))
            if self.use_bayesian and self.use_param_loss:
                param_kld = self.global_parameter_kld() / neff
                total_loss = total_loss + param_kld
            else:
                param_kld = torch.zeros(1) + 1e-5
            # ensure everything is right shape
            recon_loss = recon_loss_l.mean().item() + recon_loss_u.mean().item()
            kld_loss = kld_loss_l.mean().item() + kld_loss_u.mean().item()
            param_kld = param_kld.item()
            # TODO put px_z_ back together properly so index matches input

            # low deltaelbo must mean one is closer to consensus or what VAE thinks evolution wanted for the
            # protein as it reconstructs better and is closer to the center
            return total_loss, recon_loss, kld_loss, param_kld, encoded_distribution_l, encoded_distribution_u, px_z_l, px_z_u, classication_loss, label_loss_z, unlabelled_x, x_labelled,  U,  L_a


    def forward(self, x, neff, labels, weights):
        # Forward pass + loss + metrics
        if self.VAE or self.CVAE:
            total_loss, nll_loss, kld_loss, param_kld, encoded_distribution, px_z = self.vae_loss(x, labels, neff, weights)

        if self.SSVAE or self.SSCVAE:
            total_loss, nll_loss, kld_loss, param_kld, encoded_distribution_l, encoded_distribution_u, px_z, px_z_u, classication_loss, label_loss_z, unlabelled_x, x_labelled, unlabelled_loss, labelled_loss = self.vae_loss(x, labels, neff, weights)
        # Metrics
        metrics_dict = {}
        with torch.no_grad():
            # Accuracy
            if self.SSCVAE:
                cond = torch.isnan(labels[:,:]) != True
                l = labels[cond.any(1), :]
                u = labels[~cond.any(1), :]
                if type(encoded_distribution_l) != type(0) and x_labelled != None:
                    acc_l = (self.decoder(encoded_distribution_l.mean, l)[2].exp().argmax(dim = -1) == x_labelled).to(torch.float).mean().item()
                    metrics_dict["labelled accuracy"] = acc_l
                if type(encoded_distribution_u) != type(0) and unlabelled_x != None:
                    if self.multilabel:
                         label_dim = (labels.size(1))//self.multilabel
                         labels = multilabel_generate_permutations(u, label_dim, self.multilabel)
                    else:
                         labels = generate_all_labels(u, label_dim)
                    acc_u = (self.decoder(encoded_distribution_u.mean, labels)[2].exp().argmax(dim = -1) == unlabelled_x).to(torch.float).mean().item()
                    metrics_dict["unlabelled accuracy"] = acc_u

            if self.CVAE:
                acc = (self.decoder(encoded_distribution.mean, labels)[2].exp().argmax(dim = -1) == x).to(torch.float).mean().item()
                metrics_dict["accuracy"] = acc

            if self.VAE:
                acc = (self.decoder(encoded_distribution.mean, [None])[2].exp().argmax(dim = -1) == x).to(torch.float).mean().item()
                metrics_dict["accuracy"] = acc

            if self.SSVAE:
                if type(encoded_distribution_l) != type(0):
                    acc_l = (self.decoder(encoded_distribution_l.mean, [None])[2].exp().argmax(dim = -1) == x_labelled).to(torch.float).mean().item()
                    metrics_dict["labelled accuracy"] = acc_l
                if type(encoded_distribution_u) != type(0):
                    acc_u = (self.decoder(encoded_distribution_u.mean, [None])[2].exp().argmax(dim = -1) == unlabelled_x).to(torch.float).mean().item()
                    metrics_dict["unlabelled accuracy"] = acc_u

            metrics_dict["nll_loss"] = nll_loss
            metrics_dict["kld_loss"] = kld_loss
            metrics_dict["param_kld"] = param_kld
            if self.SSVAE or self.SSCVAE:
                metrics_dict["seq2y_loss"] = classication_loss
                metrics_dict["z2y_loss"] = label_loss_z
                metrics_dict["labelled seqs"] = x_labelled
                metrics_dict["unlabelled seqs"] = unlabelled_x
                metrics_dict["unlabelled_loss"] = unlabelled_loss
                metrics_dict["labelled_loss"] = labelled_loss

        return total_loss, metrics_dict, px_z


    def summary(self):
        num_params = sum(p.numel() for p in self.parameters())

        return (f"Variational Auto-Encoder summary:\n"
                f"  Layer sizes: {[self.layer_sizes]}\n"
                f"  Parameters: {num_params:,}\n"
                f"  Bayesian: {self.use_bayesian}\n")


    def save(self, path):
        args_dict = {
            "layer_sizes": self.layer_sizes,
            "alphabet_size": self.alphabet_size,
            "z_samples": self.z_samples,
            "dropout": self.dropout,
            "use_bayesian": self.use_bayesian,
            "nb_patterns": self.nb_patterns,
            "inner_CW_dim": self.cw_inner_dimension,
            "use_param_loss": self.use_param_loss,
            "use_sparse_interactions": self.use_sparse_interactions,
            "warm_up": self.warm_up,
        }
        time.sleep(10)
        torch.save({
            "name": "VAE",
            "state_dict": self.state_dict(),
            "args_dict": args_dict,
        }, path)

#
#
# class VAE_for_stacked(nn.Module):
#     """
#     We differentiate between SSVAE and SSCVAE/CVAE bc they have discretized labels
#     """
#     def __init__(self,
#                  layer_sizes,
#                  alphabet_size,
#                  z_samples = 1,
#                  dropout = 0.5,
#                  use_bayesian = True,
#                  num_patterns = 4,
#                  cw_inner_dimension = 40,
#                  use_param_loss = True,
#                  use_sparse_interactions = False,
#                  conditional_data_dim=False,
#                  SSVAE = False,
#                  SSCVAE = False,
#                  CVAE = False,
#                  VAE=True,
#                  multilabel=0,
#                  seq2yalpha=0,
#                  z2yalpha=0,
#                  label_pred_layer_sizes = None,
#                  pred_from_latent=False,
#                  pred_from_seq=False,
#                  warm_up = 0,
#                  batchnorm = False,
#                  device='cpu'):
#
#         super(VAE_for_stacked, self).__init__()
#
#         self.layer_sizes = layer_sizes
#         self.alphabet_size = alphabet_size
#         self.max_sequence_length = self.layer_sizes[0] // self.alphabet_size
#         self.cw_inner_dimension  = cw_inner_dimension
#         self.device = device
#         self.dropout = dropout
#         self.nb_patterns = num_patterns
#         self.dec_h2_dim = layer_sizes[-2]
#         self.z_samples = z_samples
#         self.use_bayesian = use_bayesian
#         self.use_param_loss = use_param_loss
#         self.use_sparse_interactions = use_sparse_interactions
#         self.warm_up = warm_up
#         self.warm_up_scale = 0
#         self.conditional_data_dim = conditional_data_dim
#         self.multilabel = multilabel
#         self.label_pred_layer_sizes = label_pred_layer_sizes
#         self.VAE = VAE
#         self.SSVAE = SSVAE
#         self.SSCVAE = SSCVAE
#         self.CVAE = CVAE
#         self.pred_from_latent = pred_from_latent
#         self.pred_from_seq = pred_from_seq
#         self.seq2yalpha = seq2yalpha
#         self.z2yalpha = z2yalpha
#
#         if SSVAE:
#             self.criterion = nn.MSELoss()
#         if SSCVAE:
#             self.criterion = cross_entropy_loss
#
#         self.latent_dim = layer_sizes.index(min(layer_sizes))
#
#         # Encoder
#         # encoder neural network prior to mu and sigma
#         if CVAE or SSCVAE and multilabel<2:
#             input_dim = self.max_sequence_length*self.alphabet_size+self.conditional_data_dim
#         else:
#             input_dim = self.max_sequence_length*alphabet_size
#
#         self.enc_fc1 = nn.Linear(input_dim, layer_sizes[1])
#         nn.init.xavier_normal_(self.enc_fc1.weight)
#         nn.init.constant_(self.enc_fc1.bias, 0.1)
#
#         self.enc_fc2 = nn.Linear(layer_sizes[1], layer_sizes[2])
#         nn.init.xavier_normal_(self.enc_fc2.weight)
#         nn.init.constant_(self.enc_fc2.bias, 0.1)
#
#         self.encode_layers = nn.Sequential(self.enc_fc1,
#                                            nn.ReLU(inplace = True),
#                                            self.enc_fc2,
#                                            nn.ReLU(inplace = True)
#                                                 )
#
#         # encode mu from h_dim to z_dim deterministically (reparameterization trick)
#         self.encode_mu = nn.Linear(layer_sizes[2], layer_sizes[3])
#         nn.init.xavier_normal_(self.encode_mu.weight)
#         nn.init.constant_(self.encode_mu.bias, 0.1)
#
#         # encode sigma from h_dim to z_dim deterministically (reparameterization trick)
#         self.encode_logsigma = nn.Linear(layer_sizes[2], layer_sizes[3])
#         nn.init.xavier_normal_(self.encode_logsigma.weight)
#         nn.init.constant_(self.encode_logsigma.bias, -10)
#
#         # Decoder
#         # weights layer 1
#         # TODO decide how to deal with issue of partly labelling
#         if CVAE or SSCVAE and multilabel<2:
#             z_dim = layer_sizes[3] + self.conditional_data_dim
#         else:
#             z_dim = layer_sizes[3]
#
#         # if sparse interactions and other tricks are enabled initiate variables
#         if self.use_sparse_interactions:
#             # sparse interaction and dict from deepsequence paper
#             self.mu_S = nn.Parameter(torch.Tensor(int(layer_sizes[5] / self.nb_patterns), self.max_sequence_length))
#             nn.init.zeros_(self.mu_S)
#             self.logsigma_S = nn.Parameter(torch.Tensor(int(layer_sizes[5] / self.nb_patterns), self.max_sequence_length))
#             nn.init.constant_(self.logsigma_S, -10)
#             self.mu_C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size))
#             nn.init.xavier_normal_(self.mu_C)
#             self.logsigma_C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size))
#             nn.init.constant_(self.logsigma_C, -10)
#
#             # inverse temperature term from deepsequence paper
#             self.mu_l = nn.Parameter(torch.Tensor([1]))
#             self.logsigma_l = nn.Parameter(torch.Tensor([-10.0]))
#
#             # alter output shape for sparse interactions
#             W3_output_shape = cw_inner_dimension * self.max_sequence_length
#
#         if not self.use_sparse_interactions:
#             W3_output_shape = alphabet_size * self.max_sequence_length
#
#         # bayesian network
#         if self.use_bayesian:
#             self.mu_W1_dec = nn.Parameter(torch.Tensor(z_dim, layer_sizes[4]))
#             nn.init.xavier_normal_(self.mu_W1_dec)
#             self.logsigma_W1_dec = nn.Parameter(torch.Tensor(z_dim, layer_sizes[4]))
#             nn.init.constant_(self.logsigma_W1_dec, -10)
#             self.mu_b1_dec = nn.Parameter(torch.Tensor(layer_sizes[4]))
#             nn.init.constant_(self.mu_b1_dec, 0.1)
#             self.logsigma_b1_dec = nn.Parameter(torch.Tensor(layer_sizes[4]))
#             nn.init.constant_(self.logsigma_b1_dec, -10)
#
#             # weights layer 2
#             self.mu_W2_dec = nn.Parameter(torch.Tensor(layer_sizes[4], layer_sizes[5]))
#             nn.init.xavier_normal_(self.mu_W2_dec)
#             self.logsigma_W2_dec = nn.Parameter(torch.Tensor(layer_sizes[4], layer_sizes[5]))
#             nn.init.constant_(self.logsigma_W2_dec, -10)
#             self.mu_b2_dec = nn.Parameter(torch.Tensor(layer_sizes[5]))
#             nn.init.constant_(self.mu_b2_dec, 0.1)
#             self.logsigma_b2_dec = nn.Parameter(torch.Tensor(layer_sizes[5]))
#             nn.init.constant_(self.logsigma_b2_dec, -10)
#
#             # weights layer 3
#             self.mu_W3_dec = nn.Parameter(torch.Tensor(layer_sizes[5], W3_output_shape))
#             nn.init.xavier_normal_(self.mu_W3_dec)
#             self.logsigma_W3_dec = nn.Parameter(torch.Tensor(layer_sizes[5], W3_output_shape))
#             nn.init.constant_(self.logsigma_W3_dec, -10)
#             self.mu_b3_dec = nn.Parameter(torch.Tensor(alphabet_size * self.max_sequence_length))
#             nn.init.constant_(self.mu_b3_dec, 0.1)
#             self.logsigma_b3_dec = nn.Parameter(torch.Tensor(alphabet_size * self.max_sequence_length))
#             nn.init.constant_(self.logsigma_b3_dec, -10)
#
#
#         # non-bayesian network
#         decoder_layers = []
#         if not self.use_bayesian:
#             # decoder
#             self.h1_dec = nn.Linear(z_dim, layer_sizes[4])
#             nn.init.xavier_normal_(self.h1_dec.weight)
#             nn.init.constant_(self.h1_dec.bias, 0.1)
#             decoder_layers.append(self.h1_dec)
#             decoder_layers.append(nn.ReLU())
#             decoder_layers.append(nn.Dropout(self.dropout))
#
#             self.h2_dec = nn.Linear(layer_sizes[4], layer_sizes[5])
#             nn.init.xavier_normal_(self.h2_dec.weight)
#             nn.init.constant_(self.h2_dec.bias, 0.1)
#             decoder_layers.append(self.h2_dec)
#             decoder_layers.append(nn.Sigmoid())
#             decoder_layers.append(nn.Dropout(self.dropout))
#
#             self.h3_dec = nn.Linear(layer_sizes[5], W3_output_shape, bias = False)
#             nn.init.xavier_normal_(self.h3_dec.weight)
#             self.b3 = nn.Parameter(torch.Tensor(alphabet_size * self.max_sequence_length))
#             nn.init.constant_(self.b3, -10)
#
#         self.decoder_network = nn.Sequential(*decoder_layers)
#         if self.pred_from_seq:
#             if SSCVAE:
#                 self.label_pred_model = nn.ModuleList([Classifier([30, 3, 3]) for i in range(multilabel)])
#             if SSVAE:
#                 self.label_pred_model = nn.ModuleList([Regressor([30, 3, 1]) for i in range(multilabel)])
#         if pred_from_latent:
#             if SSCVAE:
#                 self.z_label_pred = nn.ModuleList([Classifier([layer_sizes[3], 3, 3]) for i in range(multilabel)])
#             if SSVAE:
#                 self.z_label_pred = nn.ModuleList([Regressor([layer_sizes[3], 3, 1]) for i in range(multilabel)])
#
#         self.to(device)
#
#     def predict_label(self, x, i=0):
#         return self.label_pred_model[i](x)
#
#     def predict_label_from_z(self, x, i=0):
#         return self.z_label_pred[i](x)
#
#     def encoder(self, x, labels, ohc=True):
#         if self.CVAE or self.SSCVAE and self.multilabel<2:
#             x = torch.cat((x, labels.to(torch.float)), 1)
#         h = self.encode_layers(x)
#         mu = self.encode_mu(h)
#         logvar = self.encode_logsigma(h)
#         return mu, logvar
#
#     def decoder(self, z, labels):
#         if self.CVAE or self.SSCVAE and self.multilabel<2:
#             z = torch.cat((z, labels.to(torch.float)), 1)
#         # if bayesian sample new weights
#         if self.use_bayesian:
#             # weights and bias for layer 1
#             W1 = sample_gaussian(self.mu_W1_dec, self.logsigma_W1_dec)
#             b1 = sample_gaussian(self.mu_b1_dec, self.logsigma_b1_dec)
#             W2 = sample_gaussian(self.mu_W2_dec, self.logsigma_W2_dec)
#             b2 = sample_gaussian(self.mu_b2_dec, self.logsigma_b2_dec)
#             W3 = sample_gaussian(self.mu_W3_dec, self.logsigma_W3_dec)
#             b3 = sample_gaussian(self.mu_b3_dec, self.logsigma_b3_dec)
#         else:
#             W3 = self.h3_dec.weight
#             b3 = self.b3
#
#         # if sparse interactions perform linear operations
#         if self.use_sparse_interactions:
#             l = sample_gaussian(self.mu_l, self.logsigma_l)
#             S = sample_gaussian(self.mu_S, self.logsigma_S)
#             C = sample_gaussian(self.mu_C, self.logsigma_C)
#             S = torch.sigmoid(S.repeat(self.nb_patterns, 1))
#             W3 = W3.view(self.dec_h2_dim * self.max_sequence_length, -1)
#
#             W_out = W3 @ C
#             W_out = W_out.view(-1, self.max_sequence_length, self.alphabet_size)
#             W_out = W_out * S.unsqueeze(2)
#             W_out = W_out.view(-1, self.max_sequence_length * self.alphabet_size)
#
#         if not self.use_sparse_interactions and not self.use_bayesian:
#             W_out = W3.t()
#         if not self.use_sparse_interactions and self.use_bayesian:
#             W_out = W3
#
#         if self.use_bayesian:
#             h1 = nn.functional.relu(nn.functional.linear(z, W1.t(), b1))
#             h2 = torch.sigmoid(nn.functional.linear(h1, W2.t(), b2))
#
#         if not self.use_bayesian:
#             h2 = self.decoder_network(z)
#
#         h3 = nn.functional.linear(h2, W_out.t(), b3)
#         if self.use_sparse_interactions:
#             h3 = h3 * torch.log(1 + l.exp())
#
#         h3 = h3.view((-1, self.max_sequence_length, self.alphabet_size))
#         h3 = nn.functional.log_softmax(h3, -1)
#         px_z = torch.distributions.Categorical(logits=h3)
#
#         return h3, px_z
#
#     def recon_loss(self, recon_x, x):
#         # How well do input x and output recon_x agree?
#         recon_x = recon_x.view(self.z_samples, -1, recon_x.size(1), self.alphabet_size).permute(1, 2, 0, 3)
#         x = x.unsqueeze(-1).expand(-1, -1, self.z_samples)
#
#         smooth_target = F.one_hot(x.to(torch.int64), self.alphabet_size).flatten(1).to(torch.float)
#         loss = -(smooth_target * recon_x).sum(-1)
#         loss = loss.mean(-1).sum(-1)
#         return loss
#
#
#     def kld_loss(self, encoded_distribution):
#         prior = Normal(torch.zeros_like(encoded_distribution.mean), torch.ones_like(encoded_distribution.variance))
#         kld = kl_divergence(encoded_distribution, prior).sum(dim = 1)
#
#         return kld
#
#     def protein_logp(self, x):
#         x, _ = self.pretrained_model.encoder(x, labels)
#         # correctly incorporate pred loss in elbo
#         if self.SSVAE or self.SSCVAE:
#             if self.SSVAE:
#                 labels = [None]
#             if self.SSCVAE:
#                 # TODO Right now if SSCVAE and multilabel > 1 then label is not used in elbo
#                 if self.multilabel > 1:
#                     labels = torch.cat([torch.zeros(x.size(0), self.predict_label(x, i=i).size(1)).cuda().scatter_(1, torch.argmax(self.predict_label(x, i=i),1,keepdim=True), 1) for i in range(self.multilabel)], dim=1)
#                     labels = labels.cuda() if torch.cuda.is_available() else labels
#                 else:
#                     logits = self.predict_label(x, i=0)
#                     labels = torch.zeros(x.size(0), logits.size(1)).cuda().scatter_(1, torch.argmax(logits,1,keepdim=True), 1) if torch.cuda.is_available() else torch.zeros(x.size(0), logits.size(1)).scatter_(1, torch.argmax(logits,1,keepdim=True), 1)
#                 # prior is just a constant
#                 #prior = -log_standard_categorical(labels)
#             mu, logvar = self.encoder(x, labels)
#             encoded_distribution = Normal(mu, logvar.mul(0.5).exp())
#             z = encoded_distribution.rsample((self.z_samples,))
#             recon_x, px_z = self.decoder(z.squeeze(0), labels)
#             recon_loss =  F.mse_loss(recon_x.squeeze(-1), x, reduction = "none").sum(dim = 1)
#             kld_loss = self.kld_loss(encoded_distribution)
#             param_kld = torch.zeros(1) + 1e-5
#             elbo = recon_loss + kld_loss
#             logp = recon_loss
#             kld = kld_loss
#
#         if self.VAE or self.CVAE:
#             mu, logvar = self.encoder(x, labels)
#             encoded_distribution = Normal(mu, logvar.mul(0.5).exp())
#             kld = self.kld_loss(encoded_distribution)
#             z = encoded_distribution.rsample()
#             recon_x, _ = self.decoder(z, labels)
#             logp =  F.binary_cross_entropy(recon_x.squeeze(-1), x, reduction = "none").sum(dim = 1)
#             elbo = logp + kld
#
#         # amino acid probabilities are independent conditioned on z
#         return elbo, logp, kld
#
#
#     def global_parameter_kld(self):
#         global_kld = 0
#
#         global_kld += -KLdivergence(self.mu_W1_dec, self.logsigma_W1_dec.mul(0.5).exp())
#         global_kld += -KLdivergence(self.mu_b1_dec, self.logsigma_b1_dec.mul(0.5).exp())
#
#         global_kld += -KLdivergence(self.mu_W2_dec, self.logsigma_W2_dec.mul(0.5).exp())
#         global_kld += -KLdivergence(self.mu_b2_dec, self.logsigma_b2_dec.mul(0.5).exp())
#
#         global_kld += -KLdivergence(self.mu_W3_dec, self.logsigma_W3_dec.mul(0.5).exp())
#         global_kld += -KLdivergence(self.mu_b3_dec, self.logsigma_b3_dec.mul(0.5).exp())
#
#         if not self.use_sparse_interactions:
#             return global_kld
#
#         global_kld += -KLdivergence(self.mu_C, self.logsigma_C.mul(0.5).exp())
#         global_kld += -KLdivergence(self.mu_S, self.logsigma_S.mul(0.5).exp())
#         global_kld += -KLdivergence(self.mu_l, self.logsigma_l.mul(0.5).exp())
#
#         return global_kld
#
#
#     def vae_loss(self, x, labels, neff):
#         if self.CVAE or self.VAE:
#             mu, logvar = self.encoder(x, labels)
#             encoded_distribution = Normal(mu, logvar.mul(0.5).exp())
#             z = encoded_distribution.rsample((self.z_samples,))
#             recon_x, px_z = self.decoder(z.flatten(0, 1), labels)
#             recon_loss = F.mse_loss(recon_x.squeeze(-1), x, reduction='none').sum(dim = 1)
#             kld_loss = self.kld_loss(encoded_distribution)
#             recon_kld_loss = torch.mean(recon_loss + kld_loss)
#
#             if self.use_bayesian and self.use_param_loss:
#                 param_kld = self.global_parameter_kld() / neff
#                 total_loss = recon_kld_loss + param_kld
#             else:
#                 param_kld = torch.zeros(1) + 1e-5
#                 total_loss = recon_kld_loss
#             return total_loss, recon_loss.mean().item(), kld_loss.mean().item(), param_kld.item(), encoded_distribution, px_z
#
#         if self.SSVAE or self.SSCVAE:
#             label_cols = x.size(1)
#             x = torch.cat([x.float(), labels], dim=1)
#             # separate rows with and without label
#             cond = torch.isnan(x[:, label_cols:]) != True
#             labelled_data = x[cond.any(1), :]
#             unlabelled_data = x[~cond.any(1), :]
#             if labelled_data.size(0) > 0:
#                 x_labelled = labelled_data[:,:label_cols]
#                 y = labelled_data[:,label_cols:]
#                 if self.SSCVAE:
#                     # TODO decide how to handle this problem of partly labelled seqs produce NaNs
#                     if self.multilabel>1:
#                         labels = [None]
#                     else:
#                         labels = y
#                 if self.SSVAE:
#                     labels = [None]
#                 # labelled loss
#                 # Elbo
#                 mu_l, logvar_l = self.encoder(x_labelled, labels)
#                 encoded_distribution_l = Normal(mu_l, logvar_l.mul(0.5).exp())
#                 z_l = encoded_distribution_l.rsample((self.z_samples,))
#                 recon_x_l, px_z_l = self.decoder(z_l.squeeze(0), labels)
#                 recon_loss_l =  F.mse_loss(recon_x_l.squeeze(-1), x_labelled, reduction = "none").sum(dim = 1)
#                 kld_loss_l = self.kld_loss(encoded_distribution_l)
#                 # prior is just a constant
#                 # add prior on y
#                 # TODO decided not to use prior calc a prior on each label and have it interchangeable between normal and log Categorical
#                 # for ssvae evt standardize labels and have that as prior
#                 prior_l = 0
#                 if self.SSCVAE:
#                     for l in labels:
#                         prior_l += log_standard_categorical(y)
#                 # if self.SSVAE:
#                 #     for l in labels:
#                 #         prior_l += log_standard_categorical(y)
#
#                 labelled_loss = recon_loss_l + kld_loss_l + prior_l if self.SSCVAE else recon_loss_l + kld_loss_l
#                 L = torch.mean(labelled_loss)
#                 # Auxiliary classification loss q(y|x) in Regular cross entropy
#                 classication_loss = 0
#                 label_loss_z = 0
#                 # WIP TODO labels have incorrect shape
#
#                 if self.multilabel>1:
#                     keep_idx = idx_for_each_label(y, self.multilabel, y.size(1))
#                     for i in range(self.multilabel):
#                         if self.SSCVAE:
#                             num_cols = self.conditional_data_dim//self.multilabel
#                             y_true = y[keep_idx[i]][:,i*num_cols:i*num_cols+num_cols]
#                         else:
#                             y_true = y[keep_idx[i]][:,i].unsqueeze(1)
#                         assert torch.isfinite(y_true).all() == True, "target is not finite"
#                         if self.pred_from_seq:
#                             pred = self.predict_label(x_labelled[keep_idx[i]][:,:label_cols], i=i)
#                             assert torch.isfinite(pred).all() == True, "pred is not finite"
#                             classication_loss += self.criterion(pred, y_true)
#                         if self.pred_from_latent:
#                             pred_from_z = self.predict_label_from_z(mu_l[keep_idx[i]], i=i)
#                             label_loss_z += self.criterion(pred_from_z, y_true)
#                 else:
#                     if self.pred_from_seq:
#                         pred = self.predict_label(x_labelled, i=0)
#                         classication_loss +=  self.criterion(pred, y)
#                     if self.pred_from_latent:
#                         pred_from_z = self.predict_label_from_z(mu_l, i=0)
#                         label_loss_z += self.criterion(pred_from_z, y)
#
#
#                 L_a = L
#                 if self.pred_from_seq:
#                     L_a += self.seq2yalpha * classication_loss / self.multilabel
#                 if self.pred_from_latent:
#                     L_a += self.z2yalpha * label_loss_z / self.multilabel
#             else:
#                 L_a = 0
#                 classication_loss = 0
#                 label_loss_z = 0
#                 recon_loss_l = torch.tensor([0]).float()
#                 kld_loss_l = torch.tensor([0]).float()
#                 encoded_distribution_l = 0
#                 px_z_l = 0
#                 x_labelled = None
#
#
#             # unlabelled loss
#             if unlabelled_data.size(0) > 0:
#                 # generate all labels for each data point to sum out label later
#                 if self.SSCVAE:
#                     if self.multilabel > 1:
#                          label_dim = (x.size(1)-label_cols)//self.multilabel
#                          labels = multilabel_generate_permutations(unlabelled_data, label_dim, self.multilabel)
#                     else:
#                          labels = generate_all_labels(unlabelled_data, x.size(1)-label_cols)
#                     prior_u = log_standard_categorical(labels)
#
#                     unlabelled_x = unlabelled_data[:,:label_cols].repeat(labels.size(0)//unlabelled_data.size(0), 1)
#                     # add prior on y
#                 if self.SSVAE:
#                     labels = [None]
#                     unlabelled_x = unlabelled_data[:,:label_cols]
#                     # prior_u = 0
#                 # Elbo
#                 mu_u, logvar_u = self.encoder(unlabelled_x, labels)
#                 encoded_distribution_u = Normal(mu_u, logvar_u.mul(0.5).exp())
#                 z_u = encoded_distribution_u.rsample((self.z_samples,))
#                 recon_x_u, px_z_u = self.decoder(z_u.squeeze(0), labels)
#                 recon_loss_u = F.mse_loss(recon_x_u.squeeze(-1), unlabelled_x, reduction = "none").sum(dim = 1)
#                 kld_loss_u = self.kld_loss(encoded_distribution_u)
#                 L = recon_loss_u + kld_loss_u + prior_u if self.SSCVAE else recon_loss_u + kld_loss_u
#                 if self.SSCVAE:
#                     l = -L.unsqueeze(1)
#                     # Auxiliary classification loss q(y|x) in Regular cross entropy
#                     H = 0
#                     L = 0
#                     Hz = 0
#                     Lz = 0
#                     for i in range(self.multilabel):
#                         logits = self.predict_label(unlabelled_x, i=i)
#                         # Calculate entropy H(q(y|x)) and sum over all labels
#                         H += torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)
#                         L += torch.sum(torch.mul(logits, l), dim=-1)
#                         if self.pred_from_latent:
#                             logits = self.predict_label_from_z(mu_u, i=i)
#                             # Calculate entropy H(q(y|x)) and sum over all labels
#                             Hz += torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)
#                             Lz += torch.sum(torch.mul(logits, l), dim=-1)
#
#                     U = -torch.mean(L + H) if not self.pred_from_latent else -torch.mean(L + H + Lz + Hz)
#                 if self.SSVAE:
#                     U = torch.mean(L)
#             else:
#                 U = 0
#                 recon_loss_u = torch.tensor([0]).float()
#                 kld_loss_u = torch.tensor([0]).float()
#                 encoded_distribution_u = 0
#                 px_z_u = 0
#                 unlabelled_x = None
#
#             total_loss = U + L_a
#             if self.use_bayesian and self.use_param_loss:
#                 param_kld = self.global_parameter_kld() / neff
#                 total_loss = total_loss + param_kld
#             else:
#                 param_kld = torch.zeros(1) + 1e-5
#             # ensure everything is right shape
#             recon_loss = recon_loss_l.mean().item() + recon_loss_u.mean().item()
#             kld_loss = kld_loss_l.mean().item() + kld_loss_u.mean().item()
#             param_kld = param_kld.item()
#
#             # TODO put px_z_ back together properly so index matches input
#
#             # low deltaelbo must mean one is closer to consensus or what VAE thinks evolution wanted for the
#             # protein as it reconstructs better and is closer to the center
#             return total_loss, recon_loss, kld_loss, param_kld, encoded_distribution_l, encoded_distribution_u, px_z_l, px_z_u, classication_loss/self.multilabel, label_loss_z/self.multilabel, unlabelled_x, x_labelled
#
#
#     def forward(self, x, neff, labels):
#         # Forward pass + loss + metrics
#         if self.VAE or self.CVAE:
#             total_loss, nll_loss, kld_loss, param_kld, encoded_distribution, px_z = self.vae_loss(x, labels, neff)
#
#         if self.SSVAE or self.SSCVAE:
#             total_loss, nll_loss, kld_loss, param_kld, encoded_distribution_l, encoded_distribution_u, px_z, px_z_u, classication_loss, label_loss_z, unlabelled_x, x_labelled = self.vae_loss(x, labels, neff)
#         # Metrics
#         metrics_dict = {}
#         with torch.no_grad():
#             # Accuracy
#             if self.SSCVAE:
#                 cond = torch.isnan(labels[:,:]) != True
#                 l = labels[cond.any(1), :]
#                 u = labels[~cond.any(1), :]
#                 if type(encoded_distribution_l) != type(0) and x_labelled != None:
#                     acc_l = (self.decoder(encoded_distribution_l.mean, l)[0].exp().argmax(dim = -1) == x_labelled).to(torch.float).mean().item()
#                     metrics_dict["labelled accuracy"] = acc_l
#                 if type(encoded_distribution_u) != type(0) and unlabelled_x != None:
#                     if self.multilabel:
#                          label_dim = (labels.size(1))//self.multilabel
#                          labels = multilabel_generate_permutations(u, label_dim, self.multilabel)
#                     else:
#                          labels = generate_all_labels(u, label_dim)
#                     acc_u = (self.decoder(encoded_distribution_u.mean, labels)[0].exp().argmax(dim = -1) == unlabelled_x).to(torch.float).mean().item()
#                     metrics_dict["unlabelled accuracy"] = acc_u
#
#             if self.CVAE:
#                 acc = (self.decoder(encoded_distribution.mean, labels)[0].exp().argmax(dim = -1) == x).to(torch.float).mean().item()
#                 metrics_dict["accuracy"] = acc
#
#             if self.VAE:
#                 acc = (self.decoder(encoded_distribution.mean, [None])[0].exp().argmax(dim = -1) == x).to(torch.float).mean().item()
#                 metrics_dict["accuracy"] = acc
#
#             if self.SSVAE:
#                 if type(encoded_distribution_l) != type(0):
#                     acc_l = (self.decoder(encoded_distribution_l.mean, [None])[0].exp().argmax(dim = -1) == x_labelled).to(torch.float).mean().item()
#                     metrics_dict["labelled accuracy"] = acc_l
#                 if type(encoded_distribution_u) != type(0):
#                     acc_u = (self.decoder(encoded_distribution_u.mean, [None])[0].exp().argmax(dim = -1) == unlabelled_x).to(torch.float).mean().item()
#                     metrics_dict["unlabelled accuracy"] = acc_u
#
#             metrics_dict["nll_loss"] = nll_loss
#             metrics_dict["kld_loss"] = kld_loss
#             metrics_dict["param_kld"] = param_kld
#             if self.SSVAE or self.SSCVAE:
#                 metrics_dict["seq2y_loss"] = classication_loss
#                 metrics_dict["z2y_loss"] = label_loss_z
#
#         return total_loss, metrics_dict, px_z
#
#
#     def summary(self):
#         num_params = sum(p.numel() for p in self.parameters())
#
#         return (f"Variational Auto-Encoder summary:\n"
#                 f"  Layer sizes: {[self.max_sequence_length*self.alphabet_size+self.conditional_data_dim] + self.layer_sizes[1:-1] + [self.max_sequence_length*self.alphabet_size]}\n"
#                 f"  Parameters: {num_params:,}\n"
#                 f"  Bayesian: {self.use_bayesian}\n")
#
#
#     def save(self, path):
#         args_dict = {
#             "layer_sizes": self.layer_sizes,
#             "alphabet_size": self.alphabet_size,
#             "z_samples": self.z_samples,
#             "dropout": self.dropout,
#             "use_bayesian": self.use_bayesian,
#             "nb_patterns": self.nb_patterns,
#             "inner_CW_dim": self.cw_inner_dimension,
#             "use_param_loss": self.use_param_loss,
#             "use_sparse_interactions": self.use_sparse_interactions,
#             "warm_up": self.warm_up,
#         }
#
#         torch.save({
#             "name": "VAE",
#             "state_dict": self.state_dict(),
#             "args_dict": args_dict,
#         }, path)
#
#
# class Stacked_SSVAE(VAE_for_stacked):
#     def __init__(self,
#                 pretrained_model,
#                 layer_sizes,
#                 alphabet_size,
#                 z_samples = 1,
#                 dropout = 0,
#                 use_bayesian = True,
#                 num_patterns = 40,
#                 cw_inner_dimension = 4,
#                 use_param_loss = True,
#                 use_sparse_interactions = True,
#                 conditional_data_dim=0,
#                 SSVAE = False,
#                 SSCVAE = False,
#                 CVAE = False,
#                 VAE=True,
#                 multilabel=0,
#                 seq2yalpha=0,
#                 z2yalpha=0,
#                 label_pred_layer_sizes = [],
#                 pred_from_latent=False,
#                 pred_from_seq=False,
#                 warm_up = 0,
#                 batchnorm = False,
#                 device='cpu'):
#         """
#         M1+M2 model as described in [Kingma 2014].
#         """
#         super(Stacked_SSVAE, self).__init__(layer_sizes,
#                                             alphabet_size,
#                                             z_samples,
#                                             dropout,
#                                             use_bayesian,
#                                             num_patterns,
#                                             cw_inner_dimension,
#                                             use_param_loss,
#                                             use_sparse_interactions,
#                                             conditional_data_dim,
#                                             SSVAE,
#                                             SSCVAE,
#                                             CVAE,
#                                             VAE,
#                                             multilabel,
#                                             seq2yalpha,
#                                             z2yalpha,
#                                             label_pred_layer_sizes,
#                                             pred_from_latent,
#                                             pred_from_seq,
#                                             warm_up,
#                                             batchnorm)
#
#         # Make vae feature model untrainable by freezing parameters
#         self.pretrained_model = pretrained_model
#         self.pretrained_model.train(False)
#         self.layer_sizes = layer_sizes
#         self.alphabet_size = alphabet_size
#         for param in self.pretrained_model.parameters():
#             param.requires_grad = False
#         self.to(device)
#
#     def forward(self, x, neff, labels):
#         # Sample a new latent x from the M1 model
#         mu, logvar = self.pretrained_model.encoder(x, [None])
#         # Use the sample as new input to M2
#         return super(Stacked_SSVAE, self).forward(mu, neff, labels)
