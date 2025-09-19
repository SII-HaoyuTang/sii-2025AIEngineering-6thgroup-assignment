import numpy as np


def get_fashion_mnist_labels(labels): #@save
    """返回数据集的文本标签"""
    text_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return [text_labels[int(i)] for i in labels]


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    np.random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(
        indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]