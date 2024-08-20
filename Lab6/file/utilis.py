import matplotlib.pyplot as plt
import numpy as np
def plot_loss(np1,save_path):
    data1 = np.load(np1)
    plt.figure()

    plt.plot(data1,label='train_loss')
    plt.title('loss curve with consine')
    #plt.yscale('log') 
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path, format='jpg')  # Save as JPG with high quality
    plt.show()
    plt.close()

if __name__=='__main__':
    #plot_loss("./numpy_file/loss_co.npy",'./numpy_file/loss.jpg')
    data = np.load('./numpy_file/lr_co.npy')
    

    # 繪製圖表
    plt.plot( data,label="lr")
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.title('lr curve')
    plt.legend()
    plt.savefig('./numpy_file/lr.jpg', format='jpg')
    print(data)