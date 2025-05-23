#Regularization introduced in PkCC loss:- Kernel (regularization added directly to the kernel matrix, impacting the kernelized feature space - motivation is prevent overfitting and to enhance the stability - singularity or ill-conditioning during during matrix inversion or eigenvalue decomposition.

import torch
import torch.nn.functional as F

def polynomial_kernel(X1, X2, degree=4, coef0=1.0):
    return (torch.mm(X1, X2.T) + coef0) ** degree

def centre(K):
    n = K.size(0)
    H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
    return H @ K @ H

def centre_rect(K):
    n, m = K.shape
    Hn = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
    Hm = torch.eye(m, device=K.device) - torch.ones(m, m, device=K.device) / m
    return Hn @ K @ Hm

def pkcc(source_embeddings, source_labels,
         target_embeddings, target_pseudo_labels,
         num_classes, degree=4, coef0=0.8, regularization=0.3):

    device = source_embeddings.device
    loss = 0.0
    classes = 0
    eye_cache = {}

    def eye(n):
        if n not in eye_cache or eye_cache[n].device != device:
            eye_cache[n] = torch.eye(n, device=device)
        return eye_cache[n]

    for class_idx in range(num_classes):
        # Extract class-specific features
        source_class_features = source_embeddings[source_labels == class_idx]
        target_class_features = target_embeddings[target_pseudo_labels == class_idx]
        if len(source_class_features) < 100 or len(target_class_features) < 100:
            continue

        # Compute kernel matrices
        K_s = polynomial_kernel(source_class_features, source_class_features, degree, coef0)
        K_t = polynomial_kernel(target_class_features, target_class_features, degree, coef0)
        K_st = polynomial_kernel(source_class_features, target_class_features, degree, coef0)

        # Center kernel matrices
        K_s_centered = centre(K_s)
        K_t_centered = centre(K_t)
        K_st_centered = centre_rect(K_st)

        # Regularize
        L_s = K_s_centered + regularization * eye(K_s.size(0))
        L_t = K_t_centered + regularization * eye(K_t.size(0))
        L_st = K_st_centered + regularization * eye(K_s.size(0))  # square matrix on source side

        # Cholesky solve for covariance operators
        Cs  = torch.cholesky_solve(K_s_centered.T, torch.linalg.cholesky(L_s)).T
        Ct  = torch.cholesky_solve(K_t_centered.T, torch.linalg.cholesky(L_t)).T
        Cst = torch.cholesky_solve(K_st_centered.T, torch.linalg.cholesky(L_st)).T

        # Difference operator and loss
        difference = Cs + Ct - 2 * Cst
        class_loss = torch.norm(difference, p='fro') ** 2

        loss += class_loss
        classes += 1

    if classes == 0:
        return torch.tensor(0.0).to(device)

    return loss / classes
