#Regularization introduced in PkCC loss:- Kernel (regularization added directly to the kernel matrix, impacting the kernelized feature space - motivation is prevent overfitting and to enhance the stability - singularity or ill-conditioning during during matrix inversion or eigenvalue decomposition.

import torch
import torch.nn.functional as F

def polynomial_kernel(X1, X2, degree=4, coef0=0.8):
    return (X1 @ X2.T + coef0) ** degree

def centre(K):
    n = K.size(0)
    H = torch.eye(n, device=K.device, dtype=K.dtype) - torch.ones(n, n, device=K.device, dtype=K.dtype) / n
    return H @ K @ H

def centre_rect(K):
    n, m = K.shape
    Hn = torch.eye(n, device=K.device, dtype=K.dtype) - torch.ones(n, n, device=K.device, dtype=K.dtype) / n
    Hm = torch.eye(m, device=K.device, dtype=K.dtype) - torch.ones(m, m, device=K.device, dtype=K.dtype) / m
    return Hn @ K @ Hm

def pkcc(source_embeddings, source_labels,target_embeddings, target_pseudo_labels,num_classes, degree=4, coef0=0.8, regularization=0.3, eps=1e-8, jitter=1e-6):
    device = source_embeddings.device
    dtype  = source_embeddings.dtype
    Ns, Nt = source_embeddings.size(0), target_embeddings.size(0)
    if source_labels.dim() == 1:
        source_probs = F.one_hot(source_labels.long(), num_classes=num_classes).to(device=device, dtype=dtype)
    else:
        source_probs = source_labels.to(device=device, dtype=dtype)

    if target_pseudo_labels.dim() == 1:
        target_probs = F.one_hot(target_pseudo_labels.long(), num_classes=num_classes).to(device=device, dtype=dtype)
    else:
        target_probs = target_pseudo_labels.to(device=device, dtype=dtype)

    K_s_full  = polynomial_kernel(source_embeddings, source_embeddings, degree, coef0)
    K_t_full  = polynomial_kernel(target_embeddings, target_embeddings, degree, coef0)
    K_st_full = polynomial_kernel(source_embeddings, target_embeddings, degree, coef0)
    I_s = torch.eye(Ns, device=device, dtype=dtype)
    I_t = torch.eye(Nt, device=device, dtype=dtype)
    loss = torch.zeros((), device=device, dtype=dtype)
    classes = 0
    for c in range(num_classes):
        ps = source_probs[:, c:c+1]
        pt = target_probs[:, c:c+1]
        if ps.sum() < eps or pt.sum() < eps:
            continue

        W_ss = ps @ ps.T
        W_tt = pt @ pt.T
        W_st = ps @ pt.T

        K_s_w  = K_s_full  * W_ss
        K_t_w  = K_t_full  * W_tt
        K_st_w = K_st_full * W_st
        K_s_c  = centre(K_s_w)
        K_t_c  = centre(K_t_w)
        K_st_c = centre_rect(K_st_w)
        L_s  = K_s_c  + regularization * I_s
        L_t  = K_t_c  + regularization * I_t
        L_st = K_st_c + regularization * I_s
        chol_s  = torch.linalg.cholesky(L_s + jitter * I_s)
        chol_t  = torch.linalg.cholesky(L_t + jitter * I_t)
        chol_st = torch.linalg.cholesky(L_st + jitter * I_s)
        Cs  = torch.cholesky_solve(K_s_c.T,  chol_s).T
        Ct  = torch.cholesky_solve(K_t_c.T,  chol_t).T
        Cst = torch.cholesky_solve(K_st_c.T, chol_st).T
        diff = Cs + Ct - 2 * Cst
        loss = loss + torch.norm(diff, p='fro').pow(2)
        classes += 1
    if classes == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)
    return loss / classes
