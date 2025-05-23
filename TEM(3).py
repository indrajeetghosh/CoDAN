import torch
import torch.nn.functional as F

#Here is the loss function for our TEM loss, a variant to entropy minimization loss - It helps us to generate robust psudeo labels and particularly for pUDA where the traget label space is subset of source domain and this causes uncertainity on the prediction of labels. - For more information please take a look at the paper methodology section 
 
def entropy_tem_loss(logits,
                     tau: float = 0.1,
                     gamma: float = 0.95,
                     reduce_over_confident: bool = False):
#     """
#     Temperature-based Entropy Minimisation (TEM).

#     Args
#     ----
#     logits : (B, C) tensor – raw network outputs on target samples
#     tau    : temperature scaling factor (τ)
#     gamma  : confidence threshold (γ)
#     reduce_over_confident : if True, normalise by #confident samples only

#     Returns
#     -------
#     loss 𝓛_TEM.
#     """
    probs = F.softmax(logits / tau, dim=1)                
    max_probs, _ = probs.max(dim=1)
    mask = (max_probs > gamma).float()                   

    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
    entropy = entropy * mask                            
    if reduce_over_confident:
        denom = mask.sum().clamp_min(1.0)                 
    else:
        denom = probs.size(0)                             
    return entropy.sum() / denom
