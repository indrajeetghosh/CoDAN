import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import label_binarize

model.eval()

true_labels = []
pred_probs = []

# Evaluation loop
with torch.no_grad():  
    for target_batch in eval_target_dataloader:
   
        target_inputs, target_labels = target_batch
        target_inputs = target_inputs.to(device)
        target_labels = target_labels.to(device)
        
   
        outputs, _ = model(target_inputs)  
        

        outputs = torch.nn.functional.softmax(outputs, dim=1)
        
    
        true_labels.append(target_labels.cpu())
        pred_probs.append(outputs.cpu())  
        

true_labels = torch.cat(true_labels, dim=0)
pred_probs = torch.cat(pred_probs, dim=0)

#We used only Macro F1 score but we also calculated other metrics for reference,

_, pred_labels = torch.max(pred_probs, 1)
accuracy = accuracy_score(true_labels, pred_labels)
macro_f1 = f1_score(true_labels, pred_labels, average='macro')
precision = precision_score(true_labels, pred_labels, average='macro')
recall = recall_score(true_labels, pred_labels, average='macro')


print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")


data = {
    'Macro F1-Score': [macro_f1],
    'Precision': [precision],
    'Recall': [recall],
    'Accuracy': [accuracy]
}
df = pd.DataFrame(data)


df.to_csv('model_metrics_only_UDA.csv', index=False)

print("Metrics saved to model_metrics.csv")


num_classes = len(np.unique(true_labels))


y_true_bin = label_binarize(true_labels.numpy(), classes=np.arange(num_classes))


fpr = dict()
tpr = dict()
roc_auc = dict()


for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], pred_probs[:, i].numpy())
    roc_auc[i] = auc(fpr[i], tpr[i])


fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), pred_probs.numpy().ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
mean_tpr = np.zeros_like(all_fpr)

for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= num_classes

fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


plt.figure(figsize=(12, 10))


colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown', 'pink', 'gray'])


for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=4,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))


plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)


plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=6)


plt.xlabel('False Positive Rate', fontweight='semibold', fontsize=40)
plt.ylabel('True Positive Rate', fontweight='semibold', fontsize=40)
plt.legend(loc='lower right', prop={'weight': 'semibold', 'size': 18}, ncol=1)

#plt.savefig("Images/CoDAN_BAR.png", format='png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()
