import sys
from PIL import Image
import pandas as pd
import numpy as np
sys.path.append('../..')
from perceptron import Perceptron

data = pd.read_csv('data.csv')

def convertBW(path):
    image = Image.open(path)
    u=1-np.array(image.convert('1'))
    image.close()
    return u.reshape(u.shape[0]*u.shape[1])

X = np.array(list(map(convertBW,data['file'])))
y = data['y']

p = Perceptron(y,X)
p.run()
w = pd.DataFrame(p.W[0])
w.to_csv('weights.csv')

control = ['control{}.bmp'.format(i) for i in [0,1,2,3,4,5,6]]
C = np.array(list(map(convertBW,control)))
C = np.hstack([C,1*np.ones((C.shape[0],1))])
print(p.compute_activation(C,p.W[0]))
print(p.compute_activation(p.X,p.W[0]))
