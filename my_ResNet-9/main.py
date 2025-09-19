import function as f
import module as m
import numpy as np
import torchvision

def test():
    # load dataset
    train_loader = torchvision.datasets.MNIST('mnist_data/', train=True, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                              ]))

    test_loader = torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                                             transform=torchvision.transforms.Compose([
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                             ]))

    print(train_loader.data.shape)

    X = np.array(train_loader.data[:])
    X = X[:, np.newaxis, ...]
    y = np.array(train_loader.targets[:])


    for X, y in f.data_iter(100, X, y):
        print(X, '\n', y)

    # 创建模型实例
    model = m.NumPyResNet9(y, input_shape=(1, 28, 28), num_classes=10)

    # 前向传播
    model.forward(X)
    loss = model.loss()
    print("Loss:", loss)


if __name__ == '__main__':
    test()
