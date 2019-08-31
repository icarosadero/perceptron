import numpy as np

class Perceptron:
    """
    Simple, one-node perceptron neural network
    """
    def __init__(self,y,X,w0=0):
        self.X = np.hstack([X,np.ones((X.shape[0],1))]) #adding bias as one of the parameters
        self.T = y
        try:
            w0.shape
            self.W = w0*np.ones(self.X.shape)
        except:
            self.W = np.zeros(self.X.shape)
        self.theta = 0
    def compute_activation(self,x,w):
        return np.sign(x@w)
    def run(self):
        while(self.W.var(axis=0).sum()!=0.0 or self.W.sum()==0.0):
            for i in range(self.T.shape[0]):
                if self.T[i] != self.compute_activation(self.X[i],self.W[i-1]):
                    self.W[i] = self.W[i-1] + self.T[i]*self.X[i]
                else:
                    self.W[i] = self.W[i-1]
