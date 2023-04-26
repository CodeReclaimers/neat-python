import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import sys
import cv2
import math

class ClassifyEnv(gym.Env):
  """Classification as an unsupervised OpenAI Gym RL problem.
  Includes scikit-learn digits dataset, MNIST dataset
  """

  def __init__(self, trainSet, target):
    """
    Data set is a tuple of 
    [0] input data: [nSamples x nInputs]
    [1] labels:     [nSamples x 1]

    Example data sets are given at the end of this file
    """

    self.t = 0          # Current batch number
    self.t_limit = 0    # Number of batches if you need them
    self.batch   = 1000 # Number of images per batch
    self.seed()
    self.viewer = None

    self.trainSet = trainSet
    self.target   = target

    nInputs = np.shape(trainSet)[1]
    high = np.array([1.0]*nInputs)
    self.action_space = spaces.Box(np.array(0,dtype=np.float32), \
                                   np.array(1,dtype=np.float32), dtype=np.float32)
    self.observation_space = spaces.Box(np.array(0,dtype=np.float32), \
                                   np.array(1,dtype=np.float32), dtype=np.float32)

    self.state = None
    self.trainOrder = None
    self.currIndx = None

  def seed(self, seed=None):
    ''' Randomly select from training set'''
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
  
  def reset(self):
    ''' Initialize State'''    
    #print('Lucky number', np.random.randint(10)) # same randomness?
    self.trainOrder = np.random.permutation(len(self.target))
    self.t = 0 # timestep
    self.currIndx = self.trainOrder[self.t:self.t+self.batch]
    self.state = self.trainSet[self.currIndx,:]
    return self.state
  
  def step(self, action):
    ''' 
    Judge Classification, increment to next batch
    action - [batch x output] - softmax output
    '''
    y = self.target[self.currIndx]
    m = y.shape[0]

    log_likelihood = -np.log(action[range(m),y])
    reward = log_likelihood

    if self.t_limit > 0: # We are doing batches
      reward *= (1/self.t_limit) # average
      self.t += 1
      done = False
      if self.t >= self.t_limit:
        done = True
      self.currIndx = self.trainOrder[(self.t*self.batch):\
                                      (self.t*self.batch + self.batch)]

      self.state = self.trainSet[self.currIndx,:]
    else:
      done = True

    obs = self.state
    return obs, reward, done, {}


# -- Data Sets ----------------------------------------------------------- -- #

def digit_raw():
  ''' 
  Converts 8x8 scikit digits to 
  [samples x pixels]  ([N X 64])
  '''  
  from sklearn import datasets
  digits = datasets.load_digits()
  z = (digits.images/16)
  z = z.reshape(-1, (64))
  return z, digits.target

def mnist_784():
  ''' 
  Converts 28x28 mnist digits to 
  [samples x pixels]  ([N X 784])
  '''  
  import mnist
  z = (mnist.train_images()/255)
  z = preprocess(z,(28,28))
  z = z.reshape(-1, (784))
  return z, mnist.train_labels()

def mnist_256():
  ''' 
  Converts 28x28 mnist digits to [16x16] 
  [samples x pixels]  ([N X 256])
  '''  
  import mnist
  z = (mnist.train_images()/255)
  z = preprocess(z,(16,16))

  z = z.reshape(-1, (256))
  return z, mnist.train_labels()

def mnist_64():
  ''' 
  Converts 28x28 mnist digits to [8x8] 
  [samples x pixels]  ([N X 64])
  '''  
  import mnist
  z = (mnist.train_images()/63)
  z = preprocess(z,(8,8))

  z = z.reshape(-1, (64))
  return z, mnist.train_labels()


def preprocess(img,size, patchCorner=(0,0), patchDim=None, unskew=True):
  """
  Resizes, crops, and unskewes images

  """
  if patchDim == None: patchDim = size
  nImg = np.shape(img)[0]
  procImg  = np.empty((nImg,size[0],size[1]))

  # Unskew and Resize
  if unskew == True:    
    for i in range(nImg):
      procImg[i,:,:] = deskew(cv2.resize(img[i,:,:],size),size)

  # Crop
  cropImg  = np.empty((nImg,patchDim[0],patchDim[1]))
  for i in range(nImg):
    cropImg[i,:,:] = procImg[i,patchCorner[0]:patchCorner[0]+patchDim[0],\
                               patchCorner[1]:patchCorner[1]+patchDim[1]]
  procImg = cropImg

  return procImg

def deskew(image, image_shape, negated=True):
  """
  This method deskwes an image using moments
  :param image: a numpy nd array input image
  :param image_shape: a tuple denoting the image`s shape
  :param negated: a boolean flag telling whether the input image is negated

  :returns: a numpy nd array deskewd image

  source: https://github.com/vsvinayak/mnist-helper
  """
  
  # negate the image
  if not negated:
      image = 255-image
  # calculate the moments of the image
  m = cv2.moments(image)
  if abs(m['mu02']) < 1e-2:
      return image.copy()
  # caclulating the skew
  skew = m['mu11']/m['mu02']
  M = np.float32([[1, skew, -0.5*image_shape[0]*skew], [0,1,0]])
  img = cv2.warpAffine(image, M, image_shape, \
    flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)  
  return img



 
