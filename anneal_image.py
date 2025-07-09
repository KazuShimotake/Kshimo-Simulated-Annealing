from skimage.io import imread
from scipy.spatial import Delaunay
from skimage.draw import polygon

import numpy as np

import matplotlib.pyplot as plt

def get_rendered_im(im, x, plot=False):
    xpr = np.concatenate([x, np.array([[0,0],[1,0], [0,1],[1,1]]) ], 0)
    xpr = x * im.shape[:2]
    tri = Delaunay(xpr)
    simpl = tri.simplices
    setup = np.zeros_like(im)
    for s in simpl:
        r, c = xpr[s].T
        center = np.round(xpr[s].mean(0)).astype('int')
        #print(im[center[0], center[1]])

        rr, cc = polygon(r, c)
        rr = np.minimum(np.maximum(rr,0), im.shape[0]-1)
        cc = np.minimum(np.maximum(cc,0), im.shape[1]-1)
        setup[rr, cc] = im[min(max(center[0],0), im.shape[0]-1),
                           min(max(center[1],0), im.shape[1]-1)]

    if plot:
        plt.imshow(setup)
        plt.show()
    return setup

def fitness(im, x):
    setup = get_rendered_im(im, x)
    weights = np.array([0.299, 0.587, 0.114]).reshape(1,1,3) / 255.0
    # return np.power(weights * (im - setup), 2.0 ).sum()
    return np.power(weights * (im - setup), 2.0 ).mean()

# pick a random point, move it somewhere else randomly
def modify_student1(x):
    idx = np.random.randint(x.shape[0])
    xpr = x.copy()
    xpr[idx] = np.random.random((2,))
    return xpr

def modify_prof(x):
    idx = np.random.randint(x.shape[0])
    xpr = x.copy()
    xpr[idx] += (.1 - .2*np.random.random((2,)))
    return xpr

# shift all the points by a small random amount (different per point)
def modify_student2(x):
    return x

def modify_kazu(x):
    idx = np.random.randint(x.shape[0]-5)
    xpr = x.copy()
    for i in range(x.shape[0]):
        change = (xpr[i] - xpr[idx])*1e-4
        xpr[i]+=change
    return xpr

def temp(k, kmax):
    pass

def sim_anneal(im):
    # start with a random state; generating points between 0 and 1 and resizing them later, no reason for this
    x = np.random.random((100,2))
    # need to know the fitness of x
    fx = fitness(im,x)

    K = 100
    tmax = 1e3
    tmin = 1e-8
    log_tmax = np.log(tmax)
    log_tmin = np.log(tmin)
    log_tempvals = np.linspace(log_tmax,log_tmin,K)
    tempvals = np.exp(log_tempvals)

    better = 0
    worse = 0
    
    # loop over temperature values
    for T in tempvals:
        xpr = modify_prof(x)
        # xpr = modify_kazu(x)
        fpr = fitness(im,xpr)
        diffE = fpr - fx
        print(T,diffE)
        if diffE < 0:
            better+=1
            x = xpr
            fx = fpr
        else:
            worse+=1
            p = np.exp(-diffE/T)
            if np.random.random() < p:
                x = xpr
                fx = fpr
        print(T,fx)
    print("better:",better,", worse:",worse)
    return x

if __name__ == "__main__":
    x = np.random.random((100,2))
    im = imread("buckaroo.png")
    # im = imread("cover sleep well cores.jpg")
    get_rendered_im(im, x, plot=True)
    bestx = sim_anneal(im)
    get_rendered_im(im,bestx,plot=True)
