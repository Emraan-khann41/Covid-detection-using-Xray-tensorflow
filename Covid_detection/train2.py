import numpy as np
import matplotlib.pyplot as plt


H=np.load('my_history_new.npy',allow_pickle='TRUE').item()
plt.plot(H['loss'],label="train_loss")
plt.plot(H['val_loss'],label="test_loss")
plt.plot(H['accuracy'],label="train_accuracy")
plt.plot(H['val_accuracy'],label="test_accuracy")

plt.title("Training Accuracy/Loss on COVID-19 Dataset")
plt.xlabel("Epoch = 10")
plt.ylabel("Accuracy/Loss")
plt.legend(loc="lower left")
#plt.subplot(2,2,1)

plt.savefig('plot_accuracy_loss.png')
plt.show()



N = 10
