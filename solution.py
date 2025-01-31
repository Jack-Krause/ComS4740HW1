import numpy as np
import sys
import matplotlib.pyplot as plt
from helper import *


def show_images(data):
    """Show the input images and save them.

    Args:
        data: A stack of two images from train data with shape (2, 16, 16).
              Each of the image has the shape (16, 16)

    Returns:
        Do not return any arguments. Save the plots to 'image_1.*' and 'image_2.*' and
        include them in your report
    """
    # YOUR CODE HERE
    for i in range(data.shape[0]):
        plt.figure(figsize=(4, 4))
        plt.imshow(data[i], cmap='gray')
        plt.axis('off')
        plt.savefig(f"image_{i + 1}.png", bbox_inches='tight')
        plt.close()

    print("images saved")

    # END YOUR CODE


def show_features(X, y, save=True):
    """Plot a 2-D scatter plot in the feature space and save it. 

    Args:
        X: An array of shape [n_samples, n_features].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        save: Boolean. The function will save the figure only if save is True.

    Returns:
        Do not return any arguments. Save the plot to 'train_features.*' and include it
        in your report.
    """
    # YOUR CODE HERE
    c1 = (y == 1)
    c2 = (y == -1)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(X[c1, 0], X[c1, 1], color = 'blue', marker = 'o', label = "Class 1")
    plt.scatter(X[c2, 0], X[c2, 1], color = 'red', marker = 'x', label = "Class -1")
    
    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    
    if save:
        plt.savefig("train_features.png", bbox_inches='tight')
        
        
    plt.show()
    

    # END YOUR CODE


class Perceptron(object):

    def __init__(self, max_iter):
        self.max_iter = max_iter
        

    def fit(self, X, y):
        """Train perceptron model on data (X,y).
        (Implement the Perceptron Learning Algorithm (PLA))

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        # YOUR CODE HERE

        # After implementation, assign your weights w to self as below:
        w = np.zeros(X.shape[1]) #initialize the w vector to 0's
        
        for _ in range(self.max_iter):
            for i in range(X.shape[0]):
                y_pred = np.sign(np.dot(w, X[i]))
                
                if y_pred != y[i]:
                    w = w + y[i] * X[i]
                    
             
        self.W = w
        # END YOUR CODE

        return self

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W
    

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
        # YOUR CODE HERE
        if self.W is None:
            print("Error: Model has not been trained yet")
            return None
        
        y_hat = np.sign(np.dot(X, self.W))
        return y_hat

        # END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
        # YOUR CODE HERE
        y_pred = self.predict(X)
        
        correct_preds = (y_pred == y).sum()        
        accuracy = correct_preds / len(y)

        return accuracy
        # END YOUR CODE


def show_result(X, y, W):
    """Plot the linear model after training. 
       You can call show_features with 'save' being False for convenience.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].
    
    Returns:
        Do not return any arguments. Save the plot to 'result.*' and include it
        in your report.
    """
    # YOUR CODE HERE

    # END YOUR CODE


def test_perceptron(max_iter, X_train, y_train, X_test, y_test):

    # train perceptron
    model = Perceptron(max_iter)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    W = model.get_params()
    
    # test perceptron model
    test_acc = model.score(X_test, y_test)

    return W, train_acc, test_acc
