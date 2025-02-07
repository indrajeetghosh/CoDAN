#Regularization introduced in PkCC loss:- Kernel (regularization added directly to the kernel matrix, impacting the kernelized feature space - motivation is prevent overfitting and to enhance the stability - singularity or ill-conditioning during during matrix inversion or eigenvalue decomposition.

import torch
import torch.nn.functional as F

def polynomial_kernel(X1, X2, degree=4, coef0=1.0):
    return (torch.mm(X1, X2.T) + coef0) ** degree

def pkcc(source_embeddings, source_labels, target_embeddings, target_pseudo_labels, num_classes, degree=4, coef0=0.8, regularization=0.3):
    device = source_embeddings.device
    loss = 0.0
    classes = 0

    for class_idx in range(num_classes):
        # Extract class-specific features
        source_class_features = source_embeddings[source_labels == class_idx]
        target_class_features = target_embeddings[target_pseudo_labels == class_idx]
        
        # Ensure there are enough samples for both source and target
        if len(source_class_features) < 10 or len(target_class_features) < 10:
            continue

        # Calculate the polynomial kernels
        K_s = polynomial_kernel(source_class_features, source_class_features, degree, coef0)
        K_t = polynomial_kernel(target_class_features, target_class_features, degree, coef0)
        K_st = polynomial_kernel(source_class_features, target_class_features, degree, coef0)

        # Regularization for numerical stability
        n_s, n_t = K_s.size(0), K_t.size(0)
        reg_eye_s = regularization * torch.eye(n_s, device=device)
        reg_eye_t = regularization * torch.eye(n_t, device=device)

        # Center the kernel matrices
        one_s = torch.ones((n_s, n_s), device=device) / n_s
        one_t = torch.ones((n_t, n_t), device=device) / n_t

        K_s_centered = K_s - one_s @ K_s - K_s @ one_s + one_s @ K_s @ one_s
        K_t_centered = K_t - one_t @ K_t - K_t @ one_t + one_t @ K_t @ one_t

        # Cross-centering K_st
        one_st_s = torch.ones((n_s, n_t), device=device) / n_s
        one_st_t = torch.ones((n_t, n_s), device=device) / n_t
        K_st_centered = K_st - one_st_s @ K_t - K_s @ one_st_t + one_s.mean()

        # Adding regularization for stability
        L_s = K_s_centered + reg_eye_s
        L_t = K_t_centered + reg_eye_t

        # Compute cross-covariance operators
        C_s = torch.linalg.pinv(L_s) @ K_s_centered
        C_t = torch.linalg.pinv(L_t) @ K_t_centered
        C_st = torch.linalg.pinv(L_s + L_t) @ K_st_centered  # Cross-covariance regularized

        # Compute the difference operator
        difference = C_s + C_t - 2 * C_st

        # Calculate the squared Frobenius norm
        class_loss = torch.norm(difference, p='fro') ** 2

        loss += class_loss
        classes += 1

    if classes == 0:
        return torch.tensor(0.0, device=device)

    return loss / classes
