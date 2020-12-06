import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import sys

import cvxopt
from cvxopt import matrix, solvers

class_pair_to_data = {}
class_pair_to_aplha = {}
num_classes = 10

def _give_image_label_split(x, y, images, labels):
    """Gives the image, label split according to the 
    2 classes given in the args of the func.

    It also gives -1 label to x, and 1 to y.
    """
    mask = (labels == x) + (labels == y)
    mask = mask > 0

    new_labels = labels[mask]
    new_images = images[mask]

    mask = (new_labels == x)
    new_labels[mask] = -1
    
    mask = (new_labels == y)
    new_labels[mask] = 1

    return new_images, new_labels

def _load_mnist(path_train="fashion_mnist/train.csv", path_test="fashion_mnist/test.csv",split='train', x=4, y=5):
    """Load the data and normalize values b/w 0 and 1
    It also takes in x and y, for 2 classes.
    assigns -1 to x and 1 to y.
    """
    data = None
    if split == "train":
        data = pd.read_csv(path_train, header=None)
    elif split == "test":
        data = pd.read_csv(path_test, header=None)
    data = np.array(data)

    images = data[:, :-1]
    labels = data[:, -1]
    images /= 255
    new_images, new_labels = _give_image_label_split(x, y, images, labels)

    return new_images, new_labels

def _load_all_mnist(path_train="fashion_mnist/train.csv", path_test="fashion_mnist/test.csv",split='train'):
    data = None
    if split == "train":
        data = pd.read_csv(path_train, header=None)
    elif split == "test":
        data = pd.read_csv(path_test, header=None)
    data = np.array(data)

    images = data[:, :-1]
    labels = data[:, -1]
    images /= 255

    return images, labels

# TODO: Optimize this two nested loops
def _give_kernel_matrix(X, Y, kernel_func):
    """make sure Y is only {-1 or 1}
    This is actually not kernel matrix, it actually gives the P matrix
    """
    m = X.shape[0]
    K = np.zeros([m, m])

    print("generating P matrix...")

    for i in tqdm(range(m)):
        for j in range(m):
            K[i][j] = kernel_func(X[i], X[j]) * Y[i] * Y[j]

    return K

def _dot_product(x, z):
    assert x.shape[0] == z.shape[0]
    return np.sum(x * z)

def _gaussian_kernel(x, z, gamma=0.05):
    assert x.shape[0] == z.shape[0]
    diff = (x - z)**2
    diff = np.sum(diff) * (-gamma)
    return np.exp(diff)

def _solve_svm(images, labels, kernel_func):
    """
    Solves the constraint optimization problem of SVM

    Returns:
        alpha : the final params
    """
    m = labels.shape[0]

    P = _give_kernel_matrix(images, labels, kernel_func=kernel_func)
    q = np.zeros(m) - 1

    I = np.eye(m)
    minus_I = -np.eye(m)
    G = np.concatenate([minus_I, I]) # Swtich

    C = 1.0
    h_c = np.zeros(m) + C
    h_zero = np.zeros(m)
    h = np.concatenate([h_zero, h_c]) # swtich

    A = labels.copy()
    A = np.reshape(A, [1, -1])

    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix([[0.0]])
    sol = solvers.qp(P,q,G,h,A,b)
    aplha = np.array(sol['x']).ravel()

    return aplha

def _find_b(images, labels, alpha, kernel_func):
    idx = -1
    m = images.shape[0]
    for i in range(m):
        if alpha[i] > 0 and alpha[i] < 1.0:
            idx = i
            break
    
    assert idx >= 0 and alpha[idx] > 0 and alpha[idx] < 1.0

    pred = 0
    for i in range(m):
        pred += kernel_func(images[idx], images[i]) * labels[i] * alpha[i]
    
    b = labels[idx] - pred
    return b

def predict(image, images, labels, alpha, b, kernel_func=_gaussian_kernel, return_score=False):
    prediction = 0
    for i in range(images.shape[0]):
        prod = kernel_func(image, images[i]) * labels[i] * alpha[i]
        prediction += prod
    prediction += b

    # always return the pos prediction
    if return_score:
        if prediction >= 0:
            return 1, prediction
        else:
            return -1, -prediction
    
    if prediction >= 0:
        return 1
    else:
        return -1

def _find_acc(images_test, labels_test, images, labels, alpha, b, kernel_func):
    predictions = []
    print("Evaluating now...")
    for i in tqdm(range(images_test.shape[0])):
        pred = predict(images_test[i], images, labels, alpha, b, kernel_func)
        predictions.append(pred)
    predictions = np.array(predictions)

    correct = np.sum(predictions == labels_test)
    total = labels_test.shape[0]

    return correct / total

#####################################################################################

def generate_choose_2(N : int):
    choose2 = []
    for i in range(N):
        for j in range(i+1,N):
            choose2.append((i, j))
    return choose2

def _train_all_svms(path_train, kernel_func):
    print("Begining training.")
    for x, y in generate_choose_2(num_classes):
        print("Now training {} & {}".format(x, y))
        file_name = "SVM-guassian-{}-{}.npy".format(x, y)
        path_name = os.path.join("weights", file_name)

        if os.path.exists(path_name):
            print("Skipping {}-{} as weights file found in cache".format(x, y))
            continue
    
        images, labels = _load_mnist(path_train=path_train, split='train', x=x, y=y)
        kernel = kernel_func
        alpha = _solve_svm(images, labels, kernel)

        f = open(path_name,"wb")
        np.save(f, alpha)
        f.close()

    print("Training of all svms complete")

def predict_multiclass(test_image, kernel_func):
    scores = np.zeros(10)
    num = np.zeros(10)
    for x, y  in generate_choose_2(num_classes):
        # load the model first
        assert (x, y) in class_pair_to_aplha

        alpha = class_pair_to_aplha[(x, y)]
        train_images, train_labels = class_pair_to_data[(x, y)]

        b = _find_b(train_images, train_labels, alpha, kernel_func=kernel_func)

        output, score = predict(test_image, train_images, train_labels, alpha, b, kernel_func=kernel_func, return_score=True)

        if output == 1:
            scores[y] += 1
            num[y] += score
        else:
            scores[x] += 1
            num[x] += score
    
    idx = np.argmax(scores)
    idxs = []

    for i in range(scores.shape[0]):
        if scores[i] >= scores[idx]:
            idxs.append(i)
    
    mx = -1e16
    final_idx = -1
    for i in range(scores.shape[0]):
        if i in idxs:
            if mx < num[i]:
                final_idx = i
                mx = num[i]
    
    assert final_idx >= 0
    return final_idx

def _setup_data(path_train):
    """ Load trained model weights and load train data in respective dict
    """
    print("Loading trained model weights and train data for predicting test data.")
    for x, y in tqdm(generate_choose_2(num_classes)):
        file_name = "SVM-guassian-{}-{}.npy".format(x, y)
        path_name = os.path.join("weights", file_name)
        f = open(path_name, "rb")
        alpha = np.load(f)
        f.close()
        class_pair_to_aplha[(x, y)] = alpha
        class_pair_to_data[(x, y)] = _load_mnist(path_train=path_train, split="train", x=x, y=y)


def calc_acc_multiclass(path_train, path_test, kernel_func):

    # Load test images
    test_images, test_labels = _load_all_mnist(path_test=path_test, split="test")
    test_labels = test_labels.astype(np.int8)

    predictions = []
    print("Running prediction on test images")

    for i in tqdm(range(test_images.shape[0])):
        test_image = test_images[i]
        output = predict_multiclass(test_image, kernel_func)
        predictions.append(output)

        # Running acc
        # if i%10 == 0 and i > 0:
        #     preds = np.array(predictions[:i])
        #     mask = preds == test_labels[:i]
        #     print(sum(mask) / len(mask))
    
    predictions = np.array(predictions).astype(np.int8)
    return predictions

#####################################################################################

if __name__ == "__main__":

    train_path = "fashion_mnist/train.csv"
    test_path = "fashion_mnist/val.csv"
    output_path = "output.txt"

    # train_path, test_path, output_path = sys.argv[1:]
    kernel = _gaussian_kernel

    if not os.path.exists("weights"):
        os.makedirs("weights")
    
    # Train all pairs of SVM
    _train_all_svms(train_path, kernel)

    # Load trained model weights
    _setup_data(train_path)

    predictions = calc_acc_multiclass(train_path, test_path, kernel)

    f = open(output_path, "w")
    for i in range(len(predictions)):
        print(predictions[i], file=f)
    f.close()

    # test_images, test_labels = _load_all_mnist(path_test=path_test, split="test")
    # mask = test_labels == predictions
    # print("accuracy on test is : {:.3f}".format(sum(mask) / len(mask)))