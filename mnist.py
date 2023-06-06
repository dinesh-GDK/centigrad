import idx2numpy
import numpy as np
from tqdm import tqdm

from centigrad.tensor import Tensor
from centigrad.layer import Flatten, FullyConnected
from centigrad.optimizer import GradientDescent
from centigrad.loss import cross_entropy
from centigrad.activation import relu, softmax
from centigrad.model import Model

dataset = ['train-images.idx3-ubyte', 'train-labels.idx1-ubyte',
           't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte']

def label_to_onehot(index, num_of_classes=10):
    label = np.zeros((index.shape[0], num_of_classes))
    for i in range(index.shape[0]):
        label[i, index[i]] = 1
    return label

train_images = idx2numpy.convert_from_file("../data/"+dataset[0]) / 255.0
train_labels = idx2numpy.convert_from_file("../data/"+dataset[1])
test_images  = idx2numpy.convert_from_file("../data/"+dataset[2]) / 255.0
test_labels  = idx2numpy.convert_from_file("../data/"+dataset[3])

train_labels = label_to_onehot(train_labels)
test_labels = label_to_onehot(test_labels)

class MnistNet(Model):

    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.layer1 = FullyConnected(28*28, 128)
        self.layer2 = FullyConnected(128, 16)
        self.layer3 = FullyConnected(16, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = softmax(self.layer3(x))
        return x

if __name__ == "__main__":

    net = MnistNet()
    optimizer = GradientDescent(net.parameters())

    epochs = 5
    batch_size = 32

    for e in range(epochs):

        r_loss = 0
        pbar = tqdm(total=len(train_images))

        for i, (image, label) in enumerate(zip(train_images, train_labels), 1):
            
            image, label = Tensor(image), Tensor(label)
            x = net(image)
            loss = cross_entropy(x, label)
            loss.backward()

            r_loss += loss.item()
            
            if i % batch_size == 0:
                optimizer.step(batch_size)
                optimizer.zero_grad()

            if i % 500 == 0:
                pbar.update(500)

        pbar.close()
        print(f"Epoch: {e+1} | Loss: {(r_loss / len(test_images)):.4f}")

    acc = 0
    for image, label in zip(test_images, test_labels):

        image, label = Tensor(image), Tensor(label)
        x = net(image)
        
        if np.argmax(x.data) == np.argmax(label.data):
            acc += 1

    print(f"Accuracy: {(acc/len(test_images)):.4f}")
