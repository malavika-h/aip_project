"""
Copyright to DeYO Authors, ICLR 2024 Spotlight (top-5% of the submissions)
built upon on Tent code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torchvision
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from skimage.filters import threshold_otsu
from kneed import KneeLocator
from sklearn.mixture import GaussianMixture

class DeYO(nn.Module):
    """DeYO online adapts a model by entropy minimization with entropy and PLPD filtering & reweighting during testing.
    Once DeYOed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, args, optimizer, steps=1, episodic=False, deyo_margin=0.5*math.log(1000), margin_e0=0.4*math.log(1000)):
        super().__init__()
        self.model = model
        self.alpha = nn.Parameter(torch.tensor(float(args.reweight_ent))) if args.reweight_ent else nn.Parameter(torch.tensor(1.0))  # weight for entropy
        self.beta = nn.Parameter(torch.tensor(float(args.reweight_plpd))) if args.reweight_plpd else nn.Parameter(torch.tensor(1.0))   # weight for PLPD
        self.register_parameter("alpha", self.alpha)
        self.register_parameter("beta", self.beta)
        params = list(optimizer.param_groups[0]['params']) + [self.alpha, self.beta]
        optimizer.param_groups[0]['params'] = params
        self.optimizer = optimizer
        self.args = args
        if args.wandb_log:
            import wandb
        self.steps = steps
        self.episodic = episodic
        args.counts = [1e-6,1e-6,1e-6,1e-6]
        args.correct_counts = [0,0,0,0]
        
        self.deyo_margin = deyo_margin
        self.margin_e0 = margin_e0

    def update_csid_probs(self, train_loader, val_loader, device, num_classes):
        train_feats, train_labels = self.extract_features_from_loader(self, train_loader, device)
        prototypes = self.compute_prototypes(train_feats, train_labels, num_classes)

        test_feats, _ = self.extract_features_from_loader(self, val_loader, device)
        sim_scores = self.compute_similarity_scores(test_feats, prototypes)
        self.csid_probs = self.fit_gmm(sim_scores)

    def forward(self, x, iter_, targets=None, flag=True, group=None):
        if self.episodic:
            self.reset()
        
        if targets is None:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward = forward_and_adapt_deyo(x, iter_, self.model, self.args,
                                                                              self.optimizer, self.deyo_margin,
                                                                              self.margin_e0, self.alpha, self.beta,
                                                                              targets, flag, group, self)
                else:
                    outputs = forward_and_adapt_deyo(x, iter_, self.model, self.args,
                                                    self.optimizer, self.deyo_margin,
                                                    self.margin_e0, self.alpha, self.beta, 
                                                    targets, flag, group, self)
        else:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward, corr_pl_1, corr_pl_2 = forward_and_adapt_deyo(x, iter_, self.model, 
                                                                                                    self.args, 
                                                                                                    self.optimizer, 
                                                                                                    self.deyo_margin,
                                                                                                    self.margin_e0,
                                                                                                    self.alpha,
                                                                                                    self.beta,
                                                                                                    targets, flag, group, self)
                else:
                    outputs = forward_and_adapt_deyo(x, iter_, self.model, 
                                                    self.args, self.optimizer, 
                                                    self.deyo_margin,
                                                    self.margin_e0,
                                                    self.alpha,
                                                    self.beta,
                                                    targets, flag, group, self)
        if targets is None:
            if flag:
                return outputs, backward, final_backward
            else:
                return outputs
        else:
            if flag:
                return outputs, backward, final_backward, corr_pl_1, corr_pl_2
            else:
                return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state, self.alpha, self.beta)
        self.ema = None

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    #temprature = 1.1 #0.9 #1.2
    #x = x ** temprature #torch.unsqueeze(temprature, dim=-1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_deyo(x, iter_, model, args, optimizer, deyo_margin, margin, alpha, beta, targets=None, flag=True, group=None, deyo=None):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    outputs = model(x)
    if not flag:
        return outputs
    
    optimizer.zero_grad()
    entropys = softmax_entropy(outputs)
    # csid_weights = None

    # Optional: weight PLPD and entropy using csID confidence scores BEFORE filtering
    # if hasattr(args, "use_csid_weighting") and args.use_csid_weighting and hasattr(model, "csid_probs"):
    prob_slice = deyo.csid_probs[iter_ * args.test_batch_size : iter_ * args.test_batch_size + len(entropys)]
    csid_weights = torch.tensor(prob_slice, device=entropys.device, dtype=entropys.dtype)

        # Apply weighting before thresholding
        # plpd = plpd * csid_weights
        # entropys = entropys * csid_weights

    if args.filter_ent:
        # --- ORIGINAL THRESHOLDING ---
        # filter_ids_1 = torch.where((entropys < deyo_margin))
        # --- OTSU THRESHOLDING ---
        # entropy_scores = entropys.detach().cpu().numpy()
        # ent_thresh = threshold_otsu(entropy_scores)
        # filter_ids_1 = torch.where((entropys < ent_thresh))
        # filter_ids_1 = torch.where((entropys < args.global_ent_thresh))
        # --- ELBOW THRESHOLDING ---
        # entropy_scores = np.sort(entropys.detach().cpu().numpy())
        # kneedle = KneeLocator(range(len(entropy_scores)), entropy_scores, curve='convex', direction='increasing')
        # ent_thresh = entropy_scores[kneedle.knee]
        # filter_ids_1 = torch.where((entropys < ent_thresh))
        # --- UNIENT THRESHOLDING ---
        filter_ids_1 = torch.where((csid_weights > 0.5))        
        
    else:    
        filter_ids_1 = torch.where((entropys <= math.log(1000)))
    entropys = entropys[filter_ids_1]
    backward = len(entropys)
    if backward==0:
        if targets is not None:
            return outputs, 0, 0, 0, 0
        return outputs, 0, 0

    x_prime = x[filter_ids_1]
    x_prime = x_prime.detach()
    if args.aug_type=='occ':
        first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
        final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
        occlusion_window = final_mean.expand(-1, -1, args.occlusion_size, args.occlusion_size)
        x_prime[:, :, args.row_start:args.row_start+args.occlusion_size,args.column_start:args.column_start+args.occlusion_size] = occlusion_window
    elif args.aug_type=='patch':
        resize_t = torchvision.transforms.Resize(((x.shape[-1]//args.patch_len)*args.patch_len,(x.shape[-1]//args.patch_len)*args.patch_len))
        resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
        x_prime = resize_t(x_prime)
        x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=args.patch_len, ps2=args.patch_len)
        perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
        x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
        x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=args.patch_len, ps2=args.patch_len)
        x_prime = resize_o(x_prime)
    elif args.aug_type=='pixel':
        x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
        x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
        x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
    with torch.no_grad():
        outputs_prime = model(x_prime)
    
    prob_outputs = outputs[filter_ids_1].softmax(1)
    prob_outputs_prime = outputs_prime.softmax(1)

    cls1 = prob_outputs.argmax(dim=1)

    plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
    plpd = plpd.reshape(-1)
    
    if args.filter_plpd:
        # --- ORIGINAL THRESHOLDING ---
        filter_ids_2 = torch.where(plpd > args.plpd_threshold)
        # --- OTSU THRESHOLDING ---
        # plpd_scores = plpd.detach().cpu().numpy()
        # plpd_thresh = threshold_otsu(plpd_scores)
        # filter_ids_2 = torch.where((plpd > plpd_thresh))
        # --- ELBOW THRESHOLDING ---
        # plpd_scores = np.sort(plpd.detach().cpu().numpy())
        # kneedle = KneeLocator(range(len(plpd_scores)), plpd_scores, curve='convex', direction='increasing')
        # plpd_thresh = plpd_scores[kneedle.knee]
        # filter_ids_2 = torch.where((plpd > plpd_thresh))

    else:
        filter_ids_2 = torch.where(plpd >= -2.0)
    entropys = entropys[filter_ids_2]
    final_backward = len(entropys)
    
    if targets is not None:
        corr_pl_1 = (targets[filter_ids_1] == prob_outputs.argmax(dim=1)).sum().item()
        
    if final_backward==0:
        del x_prime
        del plpd
        
        if targets is not None:
            return outputs, backward, 0, corr_pl_1, 0
        return outputs, backward, 0
        
    plpd = plpd[filter_ids_2]
    
    if targets is not None:
        corr_pl_2 = (targets[filter_ids_1][filter_ids_2] == prob_outputs[filter_ids_2].argmax(dim=1)).sum().item()

    if args.reweight_ent or args.reweight_plpd:

        # --- SOFTMIN SOFTMAX IMPLEMENTATION ---
        # T = 0.3
        # entropy_weights = F.softmin(entropys / T, dim=0)
        # plpd_weights = F.softmax(plpd / T, dim=0)
        # coeff = args.reweight_ent * entropy_weights + args.reweight_plpd * plpd_weights

        # --- MULTIPLICATIVE INVERSE IMPLEMENTATION ---
        # plpd_norm = plpd - plpd.min(0)[0]
        # plpd_norm /= plpd_norm.max(0)[0]

        # entropys_norm = entropys - entropys.min(0)[0]
        # entropys_norm /= entropys_norm.max(0)[0]

        # coeff = (plpd_norm / (1 + entropys_norm))

        # --- LEARNABLE PARAMETER IMPLEMENTATION ---
        # coeff = F.softmax(alpha * plpd, dim=0) + F.softmax(-beta * (entropys - margin), dim=0)

        # coeff = (alpha * (1 / (torch.exp(((entropys.clone().detach()) - margin)))) +
        #          beta * (1 / (torch.exp(-1. * plpd.clone().detach())))
        #         ) 

        # --- UNIENT WEIGHTING ---
        # csid_weights = torch.ones(len(plpd), device = plpd.device)
        # if hasattr(args, "use_csid_weighting") and args.use_csid_weighting and hasattr(model, "csid_probs"):
        #     prob_slice = model.csid_probs[iter_ * args.test_batch_size : iter_ * args.test_batch_size + len(plpd)]
        #     csid_weights = torch.tensor(prob_slice, device=plpd.device, dtype=plpd.dtype)
        # coeff = csid_weights

        # --- ORIGINAL IMPLEMENTATION ---
        coeff = (args.reweight_ent * (1 / (torch.exp(((entropys.clone().detach()) - margin)))) +
                 args.reweight_plpd * (1 / (torch.exp(-1. * plpd.clone().detach())))
                )            
        entropys = entropys.mul(coeff)
    loss = entropys.mean(0)

    if final_backward != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()

    del x_prime
    del plpd
    
    if targets is not None:
        return outputs, backward, final_backward, corr_pl_1, corr_pl_2
    return outputs, backward, final_backward

def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state, alpha, beta):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state + [alpha, beta])


def configure_model(model):
    """Configure model for use with DeYO."""
    # train mode, because DeYO optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what DeYO updates
    model.requires_grad_(False)
    # configure norm for DeYO updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model

@torch.no_grad()
def extract_deyo_features(model, x):
    """Extract features from a ResNet-like model (excluding final FC)."""
    layers = list(model.children())[:-1]  # remove FC layer
    backbone = nn.Sequential(*layers).to(x.device)
    feats = backbone(x)
    return feats.view(feats.size(0), -1)

def extract_features_from_loader(model, dataloader, device):
    model.eval()
    all_feats, all_labels = [], []
    for imgs, labels, _ in dataloader:
        imgs = imgs.to(device)
        feats = extract_deyo_features(model, imgs)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)

def compute_prototypes(features, labels, num_classes):
    prototypes = []
    for c in range(num_classes):
        feats_c = features[labels == c]
        if len(feats_c) == 0:
            prototypes.append(np.zeros(features.shape[1]))
        else:
            prototypes.append(np.mean(feats_c, axis=0))
    return np.stack(prototypes)

def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.dot(a_norm, b_norm.T)

def compute_similarity_scores(test_features, prototypes):
    sims = cosine_similarity(test_features, prototypes)
    max_sims = np.max(sims, axis=1)
    return (max_sims - max_sims.min()) / (max_sims.max() - max_sims.min())  # normalize to [0, 1]

def fit_gmm(sim_scores):
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm.fit(sim_scores.reshape(-1, 1))
    high_conf = np.argmax(gmm.means_)
    probs = gmm.predict_proba(sim_scores.reshape(-1, 1))
        # Ensure component 0 is csID (high similarity)
    if gmm.means_[0, 0] < gmm.means_[1, 0]:
        csid_probs = probs[:, 1]  # higher mean → csID
    else:
        csid_probs = probs[:, 0]
    return probs[:, high_conf]  # probability of being csID