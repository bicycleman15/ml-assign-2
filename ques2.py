import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from cvxopt import matrix, solvers

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

def _load_mnist(path="fashion_mnist", split='train', x=4, y=5):
    """Load the data and normalize values b/w 0 and 1
    It also takes in x and y, for 2 classes.
    assigns -1 to x and 1 to y.
    """
    path = os.path.join(path, split + '.csv')
    data = pd.read_csv(path, header=None)
    data = np.array(data)

    images = data[:, :-1]
    labels = data[:, -1]
    
    images /= 255

    return _give_image_label_split(x, y, images, labels)

def _load_all_mnist(path="fashion_mnist", split='val'):
    path = os.path.join(path, split + '.csv')
    data = pd.read_csv(path, header=None)
    data = np.array(data)

    images = data[:, :-1]
    labels = data[:, -1]

    # NOTE : delete this later
    mask = labels <= 4
    images = images[mask]
    labels = labels[mask]
    # ---------------------
    
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
    G = np.concatenate([I, minus_I])

    C = 1.0
    h_c = np.zeros(m) + C
    h_zero = np.zeros(m)
    h = np.concatenate([h_c, h_zero])

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

def _find_weight_matrix(images, labels, alpha):
    """Will only work for dot prodcut features."""
    m, n = images.shape
    w = np.zeros(n)
    for i in range(m):
        w += images[i] * labels[i] * alpha[i]
    return w

def _find_b_dot(images, labels, alpha, kernel_func=_dot_product):
    W = _find_weight_matrix(images, labels, alpha)
    mx = -1e16
    mn = 1e16

    for i in range(images.shape[0]):
        prediction = np.dot(images[i], W)
        if labels[i] == 1:
            mn = min(mn, prediction)
        else:
            mx = max(mx, prediction)
    
    return -(mx + mn)/2

def _find_b_gauss(images, labels, alpha, kernel_func=_gaussian_kernel):
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

    if return_score:
        if prediction >= 0:
            return 1, prediction
        else:
            return -1, prediction
    
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

class_pair_to_data = {}
class_pair_to_aplha = {}

def generate_choose_2(N : int):
    for i in range(N):
        for j in range(i+1,N):
            yield (i, j)

def _train_all_svms():
    print("Begining training.")
    for x, y in generate_choose_2(10):
        print("Now training {} & {}".format(x, y))
        file_name = "SVM-guassian-{}-{}.npy".format(x, y)
        path_name = os.path.join("weights", file_name)

        if os.path.exists(path_name):
            print("Skipping {}-{} as weights file found in cache".format(x, y))
            continue
    
        images, labels = _load_mnist(split='train', x=x, y=y)
        kernel = _gaussian_kernel

        alpha = _solve_svm(images, labels, kernel)

        f = open(path_name,"wb")
        np.save(f, alpha)
        f.close()

    print("Training of all svms complete")

def predict_multiclass(image, kernel_func=_gaussian_kernel):
    scores = np.zeros(10)
    num = np.zeros(10)
    for x, y  in tqdm(generate_choose_2(10)):
        # load the model first
        file_name = "SVM-guassian-{}-{}.npy".format(x, y)
        path_name = os.path.join("weights", file_name)

        assert os.path.exists(path_name)

        f = open(path_name,"rb")
        alpha = np.load(f)
        f.close()

        images, labels = _load_mnist(x=x, y=y)

        b = 0.
        if kernel_func == _gaussian_kernel:
            b = _find_b_gauss(images, labels, alpha)
        else:
            b = _find_b_dot(images, labels, alpha)

        output, score = predict(image, images, labels, alpha, b, kernel_func=kernel_func,return_score=True)

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

def _setup_data(split='val'):
    print("Loading weights and data.")

def calc_acc_multiclass(split='val', kernel_func=_gaussian_kernel):
    _setup_data(split=split)
    images, labels = _load_all_mnist(split=split)
    predictions = []

    print("Now running validation on split {}.".format(split))
    for i in tqdm(range(images.shape[0])):
        image = images[i]
        output = predict_multiclass(image, kernel_func)
        predictions.append(output)
    
    predictions = np.array(predictions)
    mask = predictions == labels

    correct = np.sum(mask)
    total = len(predictions)

    return correct / total

#####################################################################################

# images, labels = _load_mnist(split='train', x=2, y=3)

# print("loaded training data")
# print(images.shape)

# kernel = _dot_product

# alpha = _solve_svm(images, labels, kernel)

# # b = _find_b_gauss(images, labels, alpha)
# b = _find_b_dot(images, labels, alpha)

# X_test, Y_test = _load_mnist(split='test', x=2, y=3)

# print("loaded val data")
# print(X_test.shape)

# acc = _find_acc(X_test, Y_test, images, labels, alpha, b, kernel)
# print(acc * 100)

# _train_all_svms()

print(calc_acc_multiclass())