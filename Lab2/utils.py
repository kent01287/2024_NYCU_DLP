# script for drawing figures, and more if needed
import matplotlib.pyplot as  plt
import sys
import matplotlib
import numpy as np
def plot_learning(n_epochs, train_losses):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.plot(range(1, n_epochs + 1), train_losses, label='Training Loss')
    plt.show()
    
if __name__ == '__main__': 
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Test Plot')
    plt.show()
    
