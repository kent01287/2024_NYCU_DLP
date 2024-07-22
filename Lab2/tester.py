# implement your testing script here
from Dataloader import MIBCI2aDataset
from torch.utils.data import Dataset ,DataLoader
from model.SCCNet import SCCNet ,SquareLayer,LogLayer
import torch
import torch.nn as nn

def test(mode,method,batch_size,model_path):
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ############## testing set###########################
    
    #load the testing set
    test_set = MIBCI2aDataset(mode=mode,method=method)
    testloader = DataLoader(test_set,batch_size=batch_size ,shuffle = False)
    
    # loss function
    criterion = nn.CrossEntropyLoss()
    #load the moedel
    model = torch.load(model_path)
    #define softmax
    softmax = nn.Softmax(dim=-1)
    # turn the model into eval mode
    model.eval()
    
    test_loss = 0
    correct = 0
    predictions = []
    true_labels = []
    total=0
    
    with torch.no_grad():
        for features, labels in testloader:
            # transfer to GPU
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            logits = model(features.to(torch.float32))
            preds =softmax(logits)
            # Calculate loss
            loss = criterion(logits, labels.to(torch.int64))
            test_loss += loss.item()  # Accumulate the total test loss
            
            
            # Get predicted labels
            preds = logits.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())  # Store predicted labels
            true_labels.extend(labels.cpu().numpy())  # Store true labels for comparison
            total+=labels.size(0)
            # Calculate accuracy
            correct += (preds == labels).sum().item()

    # Average test loss
    avg_test_loss = test_loss / len(testloader) #8

    # Accuracy calculation
    accuracy = correct / total
    
    
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Accuracy: {accuracy *100:.2f}%")

if __name__ == '__main__': 
    print("The method is SD")
    test(mode='test',method='SD',batch_size=288,model_path='model_weight/model_new_SD_60.pt')
    print("The method is LOSO")
    test(mode='test',method='LOSO',batch_size=288,model_path='model_weight/model_new_loso_63.19.pt')
    print("The method is LOSO with finetune")
    test(mode='test',method='LOSO',batch_size=288,model_path='model_weight/model_new_losoft_73.9.pt')