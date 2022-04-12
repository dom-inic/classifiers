import torch
import numpy as np
# QUESTION 1


def question1():
    # Define A
    A = torch.tensor([[12,-2,23], [-8,0,1]])
    # Define B
    B = torch.tensor([[1,9], [-1, 2], [0, 2]])

    number_of_the_rows = B.numel() # change the value
    element_wise_A_B = A.mul(B.T)  # change the value
    dot_product_A_B = A.mm(B) # change the value
    transpose_value = B.t()  # change the value

    print(number_of_the_rows, element_wise_A_B,
          dot_product_A_B, transpose_value)
    return number_of_the_rows, element_wise_A_B, dot_product_A_B, transpose_value


# QUESTION 2
w = torch.randn(2, 3, requires_grad=True)  # total weight
b = torch.randn(2, requires_grad=True)  # total bias


def model(x):
    return x @ w.t() + b


inputs = np.array([[73, 67, 43],
                  [91, 88, 64],
                  [87, 134, 58],
                  [102, 43, 37],
                  [69, 96, 70]], dtype='float32')
preds = model(torch.from_numpy(inputs))
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')


def mse(t1, t2):
    loss = None

    # Replace "pass" statement with your code
    diff = t1 - t2
    loss = torch.sum(diff * diff) / diff.numel()

    return loss

# Uncomment the following lines to test
loss = mse(preds, targets)
print(loss)


# QUESTION 3
def create_sample_tensor():
    """
    Return a torch Tensor of shape (4, 2) which is filled with ones, except for
    element (0, 1) which is set to 10 and element (1, 0) which is set to 100.

    Inputs: None

    Returns:
    - Tensor of shape (4, 2) as described above.
    """
    x = None

    # Replace "pass" statement with your code
    x = torch.ones(4, 2)
    x[0, 1] = 10
    x[1, 0] = 100

    return x


def reverse_rows(x):
    """
    Reverse the rows of the input tensor.

    Your implementation should construct the output tensor using a single integer
    array indexing operation. The input tensor should not be modified.

    Input:
    - x: A tensor of shape (M, N)

    Returns: A tensor y of shape (M, N) which is the same as x but with the rows
             reversed; that is the first row of y is equal to the last row of x,
             the second row of y is equal to the second to last row of x, etc.
    """
    y = None

    # Replace "pass" statement with your code
    tx = torch.arange(-x.shape[0], 0).__reversed__()
    y = x[tx, :]

    return y


def make_one_hot(x):
    """
    Construct a tensor of one-hot-vectors from a list of Python integers.

    Input:
    - x: A list of N integers

    Returns:
    - y: A tensor of shape (N, C) and where C = 1 + max(x) is one more than the max
         value in x. The nth row of y is a one-hot-vector representation of x[n];
         In other words, if x[n] = c then y[n, c] = 1; all other elements of y are
         zeros. The dtype of y should be torch.float32.
    """
    y = None

    # Replace "pass" statement with your code
    y = torch.zeros([len(x), max(x)+1], dtype=torch.float32)
    y[torch.arange(len(x)), x] = 1

    return y


def normalize_columns(x):
    """
    Normalize the columns of the matrix x by subtracting the mean and dividing
    by standard deviation of each column. You should return a new tensor; the
    input should not be modified.

    More concretely, given an input tensor x of shape (M, N), produce an output
    tensor y of shape (M, N) where y[i, j] = (x[i, j] - mu_j) / sigma_j, where
    mu_j is the mean of the column x[:, j].

    Input:
    - x: Tensor of shape (M, N).

    Returns:
    - y: Tensor of shape (M, N) as described above. It should have the same dtype
      as the input x.
    """
    y = None

    # Replace "pass" statement with your code
    mu = x.sum(dim=0) / x.shape[0]
    sigma = torch.sqrt((((x-mu)**2).sum(dim=0)) / (x.shape[0]-1))

    print('\nMean\n', mu)
    y = ((x-mu.view(1, -1)))/sigma.view(1, -1)

    return y
