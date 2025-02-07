import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

        
best_model_state_dict = None
best_accuracy = 0.0
num_epochs = 200
entropy_loss_weight = 0.2
temperature = 0.2
confidence_threshold = 0.85
epoch_losses = []
total_acc_list = []

num_classes = 12

save_folder = 'Model_checkpoints'
os.makedirs(save_folder, exist_ok=True)


for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    epoch_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for (source_data, source_labels), (target_data, _) in zip(source_dataloader, target_dataloader):
        source_data, source_labels = source_data.to(device), source_labels.to(device)
        target_data = target_data.to(device)

        optimizer.zero_grad()

    
        source_outputs, source_embeddings = model(source_data)
        source_loss = F.cross_entropy(source_outputs, source_labels)

        target_outputs, target_embeddings = model(target_data)

        # Apply the temperature-threshold-based entropy minimization loss
        target_entropy_loss = entropy_tem_loss(target_outputs, temperature=temperature, threshold=confidence_threshold)

        # Combine source loss with entropy loss
        total_loss = source_loss + entropy_loss_weight * target_entropy_loss

  
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

        _, preds = torch.max(source_outputs, 1)
        correct_predictions += torch.sum(preds == source_labels).item()
        total_predictions += source_labels.size(0)


    avg_epoch_loss = epoch_loss / len(source_dataloader)
    epoch_losses.append(avg_epoch_loss)

    epoch_time = time.time() - start_time
    epoch_accuracy = correct_predictions / total_predictions
    total_acc_list.append(epoch_accuracy)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')#, Time: {epoch_time:.2f} seconds')
    

    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        best_model_state_dict = model.state_dict()

      
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"best_model_{timestamp}_accuracy{best_accuracy:.4f}.pt"
        save_path = os.path.join(save_folder, filename)
        torch.save(best_model_state_dict, save_path)
    

#average_time_per_epoch = np.mean(epoch_times)
#print(f'Average Time per Epoch: {average_time_per_epoch:.2f} seconds')
#epoch_times = np.array(epoch_times)
#np.save('epoch_times.npy', epoch_times)
