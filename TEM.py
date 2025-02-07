import torch
import torch.nn.functional as F

#Here is the loss function for our TEM loss, a variant to entropy minimization loss - It helps us to generate robust psudeo labels and particularly for pUDA where the traget label space is subset of source domain and this causes uncertainity on the prediction of labels. - For more information please take a look at the paper methodology section 
    
def entropy_tem_loss(predictions, temperature=0.1, threshold=0.99):
    scaled_predictions = predictions / temperature
    probabilities = F.softmax(scaled_predictions, dim=1)
    
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=1)
    max_probs, _ = torch.max(probabilities, dim=1)

    confident_mask = max_probs > threshold
    if confident_mask.any():
        return torch.mean(entropy[confident_mask])
    else:
        return torch.mean(entropy)