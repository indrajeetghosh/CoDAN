# KDD 2025
Official Implementation of CoDAN Framework


**Abstract:-** We propose a novel deep adaptation approach, *CoDAN*, that addresses two challenges: (i) generating robust pseudo labels for the unlabeled target domain and (ii) minimizing the conditional shift between domains. To address the first challenge, we employ a temperature-based threshold entropy minimization loss, which generates high-confidence pseudo labels, adjusting the confidence levels of the model's predictions by scaling the logits with a temperature parameter. Following the second challenge, we introduce a novel conditional domain-invariant loss inspired by a high-order statistics technique, polynomial kernel-based cross-covariance (PkCC). The PkCC loss enables the capture of conditional feature embedding by transforming the feature space into the high-dimensional reproducing kernel Hilbert space, subsequently reducing the conditional shift between the source and target domains. We also showcase *CoDAN* can be employed to address a particular scenario, partial UDA (pUDA), where the target domain label space is a subset of the source domain label space. 


**Overall Pipeline:-**

![CoDAN Framework: comprises two main modules: AdLR and conditional embedding alignment. The principle function of the (i) AdLR module is to generate robust pseudo labels for target domains and (2) conditional embedding alignment to minimize the conditional embedding discrepancy between the domains.](DoCAN.png)

We exhibit and discuss the overall architecture of the CoDAN Framework and each module of the framework. We categorize the CoDAN framework into two main modules: (i) adaptive label refinement to generate high-quality pseudo labels and extract discriminant information via temperature-based threshold entropy minimization (TEM) loss and (ii) conditional embedding alignment module to minimize domain discrepancy between the source and target domain features using polynomial kernel-based cross-covariance (PkCC).


