import autograd.numpy as np
import copy

def check_gradient(loss_function, grad_function, x, indices):
    """
    This function checks the accuracy of a loss_function w.r.t. gradient computation 
    using finite differences
    """

    print("Checking gradient accuracy")
    is_valid = True
    h = 1e-7
    f_x = loss_function(x)
    dj = grad_function(x)
    print("Index i  expect. grad[i] grad[i]   rel. diff.")
    for index in indices:
        x_new = copy.copy(x)
        x_new[index] += h
        dj_expected = (loss_function(x_new) - f_x) / h
        rel_error = (dj[index] - dj_expected)/dj[index]
        print("%7d   %9f   %9f   %8.2g" % (index, dj_expected, dj[index], rel_error))

        if np.abs(rel_error) > 1e-5:
            is_valid = False

    if not is_valid:
        print("The gradient seems to be wrong")
    else:
        print("The gradient looks correct :)")
    return is_valid

def encode_one_hot(y, n_labels):
    """
    Perform one hot encoding
    """
    return np.eye(n_labels)[y]
