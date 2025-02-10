import torch
import torch.nn.functional as F

#Here is the loss function for our TEM loss, a variant to entropy minimization loss - It helps us to generate robust psudeo labels and particularly for pUDA where the traget label space is subset of source domain and this causes uncertainity on the prediction of labels. - For more information please take a look at the paper methodology section 
    
def entropy_tem_loss(predictions, temperature=0.1, threshold=0.95):
    predictions = F.softmax(predictions / temperature, dim=1)
    
    max_probs, _ = torch.max(predictions, dim=1)
    mask = max_probs > threshold  

    entropy = -torch.sum(predictions * torch.log(predictions + 1e-9), dim=1)
    entropy = entropy * mask.float()
    return torch.sum(entropy) / predictions.size(0)