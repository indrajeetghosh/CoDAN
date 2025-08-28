#Regularization introduced in PkCC loss:- Kernel (regularization added directly to the kernel matrix, impacting the kernelized feature space - motivation is prevent overfitting and to enhance the stability - singularity or ill-conditioning during during matrix inversion or eigenvalue decomposition.

import torch
import torch.nn.functional as F

def polynomial_kernel(X1, X2, degree=4, coef0=0.8):
    return (torch.mm(X1, X2.T) + coef0) ** degree

def center_square(K, w=None, eps=1e-6):
    n = K.size(0)
    if w is None:
        H = torch.eye(n, device=K.device, dtype=K.dtype) - torch.ones(n, n, device=K.device, dtype=K.dtype) / n
        return H @ K @ H
    w = w / (w.sum() + eps)
    H = torch.eye(n, device=K.device, dtype=K.dtype) - torch.outer(torch.ones(n, device=K.device, dtype=K.dtype), w)
    D = torch.diag(torch.sqrt(w + eps))
    return H @ (D @ K @ D) @ H

def center_rect(K, ws=None, wt=None, eps=1e-6):
    n, m = K.shape
    if ws is None or wt is None:
        Hn = torch.eye(n, device=K.device, dtype=K.dtype) - torch.ones(n, n, device=K.device, dtype=K.dtype) / n
        Hm = torch.eye(m, device=K.device, dtype=K.dtype) - torch.ones(m, m, device=K.device, dtype=K.dtype) / m
        return Hn @ K @ Hm
    ws = ws / (ws.sum() + eps)
    wt = wt / (wt.sum() + eps)
    Hn = torch.eye(n, device=K.device, dtype=K.dtype) - torch.outer(torch.ones(n, device=K.device, dtype=K.dtype), ws)
    Hm = torch.eye(m, device=K.device, dtype=K.dtype) - torch.outer(torch.ones(m, device=K.device, dtype=K.dtype), wt)
    Ds = torch.diag(torch.sqrt(ws + eps))
    Dt = torch.diag(torch.sqrt(wt + eps))
    return Hn @ (Ds @ K @ Dt) @ Hm

def pkcc(source_embeddings, source_labels, target_embeddings, target_pseudo_labels, num_classes, degree=4, coef0=0.8, regularization=0.3):
    Xs, Xt = source_embeddings, target_embeddings
    device, dtype = Xs.device, Xs.dtype
    Ns, Nt = Xs.size(0), Xt.size(0)
    eps = 1e-6 if dtype == torch.float32 else 1e-12
    if source_labels.dim() == 1:
        Ps = F.one_hot(source_labels.long(), num_classes=num_classes).to(device=device, dtype=dtype)
    else:
        Ps = source_labels.to(device=device, dtype=dtype)
    if target_pseudo_labels.dim() == 1:
        Pt = F.one_hot(target_pseudo_labels.long(), num_classes=num_classes).to(device=device, dtype=dtype)
    else:
        Pt = target_pseudo_labels.to(device=device, dtype=dtype)
    Kss = polynomial_kernel(Xs, Xs, degree, coef0)
    Ktt = polynomial_kernel(Xt, Xt, degree, coef0)
    Kst = polynomial_kernel(Xs, Xt, degree, coef0)
    Kts = Kst.T
    Is = torch.eye(Ns, device=device, dtype=dtype)
    It = torch.eye(Nt, device=device, dtype=dtype)
    loss = Xs.new_tensor(0.)
    used = 0
    for c in range(num_classes):
        ps = Ps[:, c:c+1]
        pt = Pt[:, c:c+1]
        if ps.sum() <= eps or pt.sum() <= eps:
            continue
        ws = ps.squeeze(-1)
        wt = pt.squeeze(-1)
        Wss = ps @ ps.T
        Wtt = pt @ pt.T
        Wst = ps @ pt.T
        Kss_c = center_square(Kss * Wss, ws, eps)
        Ktt_c = center_square(Ktt * Wtt, wt, eps)
        Kst_c = center_rect(Kst * Wst, ws, wt, eps)
        Kts_c = Kst_c.T
        Kss_c = 0.5 * (Kss_c + Kss_c.T)
        Ktt_c = 0.5 * (Ktt_c + Ktt_c.T)
        As = Kss_c + regularization * Is
        At = Ktt_c + regularization * It
        chol_As = torch.linalg.cholesky(As)
        chol_At = torch.linalg.cholesky(At)
        tmp_s  = torch.cholesky_solve(Kss_c, chol_As)
        Cs     = Kss_c @ tmp_s
        tmp_t  = torch.cholesky_solve(Kts_c, chol_At)
        Ct_s   = Kst_c @ tmp_t
        tmp_t2 = torch.cholesky_solve(Ktt_c, chol_At)
        Ct     = Ktt_c @ tmp_t2
        tmp_s2 = torch.cholesky_solve(Kst_c, chol_As)
        Cs_t   = Kts_c @ tmp_s2
        diff_s = Cs - Ct_s
        diff_t = Ct - Cs_t
        loss  += (diff_s.square().sum() + diff_t.square().sum()) * 0.5
        used  += 1
    return loss / max(used, 1)

