# implement your training script here
from torch.optim.lr_scheduler import StepLR
from Dataloader import MIBCI2aDataset
from torch.utils.data import  DataLoader
from utils import plot_learning
from model.SCCNet import SCCNet,LogLayer,SquareLayer
import torch
import torch.nn as nn
import matplotlib.pyplot as  plt


def train(n_epochs ,models,lr,mode,method ,batch_size,model_path) :
    #################Prepare Data ###################
    #DataSet
    train_set = MIBCI2aDataset(mode=mode ,method=method)
    #DataLoader
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle =True)
    
    
    
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize a model, and put it on the device specified.
    model = models.to(device)

    #loss function
    criterion = nn.CrossEntropyLoss()

    # L2 regulation 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) #0.0001
    #scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
    
    
    # The number of training epochs.
    n_epochs = n_epochs

    softmax = nn.Softmax(dim=-1)

    train_losses = []

    for epoch in range(n_epochs):
        # These are used to record information in training.
        train_loss = []
        train_accs = []
        
        
        ###################training#############################
        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()
            # Forward the data. (Make sure data and model are on the same device.)
            
            logits = model(features.to(torch.float32))

            # Obtain the probability distributions by applying softmax on logits.
            probs = softmax(logits)
            
            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(torch.int64))
            

            # Compute the gradients for parameters.
            loss.backward()

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
            
            
        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        
        train_losses.append(loss.item())
        
        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        #update lr
        #scheduler.step()
        
        
    
    #Save the model
    torch.save(model, model_path)
    print(f'model save to {model_path}')
    
    plot_learning(n_epochs,train_losses)
    return train_losses

if __name__ == '__main__':
    model =  SCCNet()
    loss1 =train(n_epochs=200 ,models=model,lr=0.001,mode='train',method='SD' ,batch_size=288,model_path='model_weight/model1.pt')
    loss2 =train(n_epochs=200 ,models=model,lr=0.001,mode='train',method='LOSO' ,batch_size=288,model_path='model_weight/model2.pt')
    model = torch.load('model_weight/model_new_loso_63.19.pt')
    loss3 =train(n_epochs=200 ,models=model,lr=0.001,mode='train',method='SD' ,batch_size=288,model_path='model_weight/model3.pt')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.plot(range(1,201), loss1, label='SD',color='blue')
    plt.plot(range(1,201), loss2, label='LOSO',color='red')
    plt.plot(range(1,201), loss3, label='LOSO with finetune',color='green')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
