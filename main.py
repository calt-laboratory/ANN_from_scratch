# modules
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from neural_network import NeuralNetwork

# prepare dataset
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# instantiate a NeuralNetwork object and train the network
nn = NeuralNetwork(X, y, learning_rate = 0.001, epochs = 5000)
hidden_weights, outer_weights = nn.train()

# plot the loss over epochs and the accuracy
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15, 8))
ax1.plot(nn.loss_values)
ax1.set_ylabel('Loss Value')
ax1.set_title('LOSS OVER TRAINING EPOCHS')
ax2.plot(nn.acc)
ax2.set_xlabel('Number of Epochs')
ax2.set_ylabel('Accuracy')
ax2.set_title('ACCURACY OVER TRAINING EPOCHS')
plt.show()