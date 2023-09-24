import numpy as np
import matplotlib.pyplot as plt
import numpy.random
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
data=mnist['data']
labels=mnist['target']

idx = numpy.random.RandomState(0).choice(70000, 11000)
train = np.array(data[idx[:10000], :])
train_labels = np.array(labels[idx[:10000]]).astype(int)
test = np.array(data[idx[10000:], :])
test_labels = np.array(labels[idx[10000:]]).astype(int)

#Q1a
def knn(images, labels, query_image, k: int):
    distances = np.linalg.norm(images-query_image, axis=1)
    uniques, counts = np.unique(labels[np.argsort(distances)][:k], return_counts=True)
    ret = uniques[np.argmax(counts)]
    return ret

def test_n_images(n: int, k: int):
    print(f'n: {n}, k: {k}')
    knn_labeler = lambda image: knn(train[:n], train_labels[:n], image, k)
    knn_labels = np.fromiter(map(knn_labeler, test), dtype=int)
    return np.sum(knn_labels == test_labels) / len(test)


def Q1b():
    print('Q1b - the prediction accuracy is', test_n_images(n=1000, k=10))
    
def Q1c():
    q1c_lambda = lambda k: test_n_images(1000, k)
    accuracies = np.fromiter(map(q1c_lambda, range(1, 101)), dtype=float)
    plt.plot(accuracies)
    plt.show()
    
def Q1d():
    q1d_lambda = lambda n: test_n_images(n=n, k=1)
    accuracies = np.fromiter(map(q1d_lambda, range(100, 5001, 100)), dtype=float)
    plt.plot(accuracies)
    plt.show()


def main():
    #Q1b()
    #Q1c()
    Q1d()

main()