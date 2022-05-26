import numpy as np
import os
import pandas as pd
from scipy import special
import matplotlib.pyplot as plt
import csv
import h5py
import cvxpy as cp
import torchvision.datasets as datasets
import torch
from torchvision import transforms
from kymatio.sklearn import Scattering2D

def preprocess_data_mnist(dataset):
    X = torch.clone(dataset.data).float()
    data = X.numpy()
    data = np.array(data)
    data -= data.mean(axis=0)
    data /= data.std()
    return np.array(data)

def preprocess_data_fashionmnist(dataset):
    X = torch.clone(dataset.data).float()
    data = X.numpy()
    data = np.array(data)
    data -= data.mean(axis=0)
    data /= data.std()
    return np.array(data)

def preprocess_data_cifar10(dataset):
    n_samples, _, _, _ = dataset.data.shape
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=n_samples, shuffle=True, num_workers=0)
    data = []
    for i, batch in enumerate(trainloader):
        for k,l in enumerate(batch[1]):
            data.append(batch[0][k].numpy())
    data = np.array(data)
    data -= data.mean(axis=0)
    data /= data.std()
    return np.array(data)[:,0,:,:]

def relu(x):
    return x * (x > 0.) + 0 * (x <= 0.)

def get_real_data(p, which_dataset = "MNIST", which_transform = "no_transform", which_non_lin = "none"):
    if which_dataset == "MNIST":
        if which_transform == 'no_transform':
            mnist = datasets.MNIST(root='data', train=True, download=True, transform=None)
            X = preprocess_data_mnist(mnist)
            ntot, dx, dy = X.shape
            X = X.reshape(ntot, -1)
        elif which_transform == 'wavelet_scattering':
            mnist = datasets.MNIST(root='data', train=True, download=True, transform=None)
            X = preprocess_data_mnist(mnist)
            ntot, dx, dy = X.shape
            S = Scattering2D(shape=(dx,dy), J=3, L=8)
            X = S(X).reshape(ntot, -1)
        elif which_transform == 'random_gaussian_features':
            if which_non_lin == 'erf':
                mnist = datasets.MNIST(root='data', train=True, download=True, transform=None)
                X = preprocess_data_mnist(mnist)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1)
                ntot, d = X.shape
                F = np.random.normal(0., 1., size = (d, p))
                X = special.erf(X @ F)
            elif which_non_lin == 'tanh':
                mnist = datasets.MNIST(root='data', train=True, download=True, transform=None)
                X = preprocess_data_mnist(mnist)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1)
                ntot, d = X.shape
                F = np.random.normal(0., 1., size = (d, p))
                X = np.tanh(X @ F)
            elif which_non_lin == 'sign':
                mnist = datasets.MNIST(root='data', train=True, download=True, transform=None)
                X = preprocess_data_mnist(mnist)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1)
                ntot, d = X.shape
                F = np.random.normal(0., 1., size = (d, p))
                X = np.sign(X @ F)
            elif which_non_lin == 'relu':
                mnist = datasets.MNIST(root='data', train=True, download=True, transform=None)
                X = preprocess_data_mnist(mnist)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1)
                ntot, d = X.shape
                F = np.random.normal(0., 1., size = (d, p))
                X = relu(X @ F)
    elif which_dataset == "fashion-MNIST":
        if which_transform == 'no_transform':
            fashion_mnist = datasets.FashionMNIST(root='data', train=True, download=True, transform=None)
            X = preprocess_data_fashionmnist(fashion_mnist)
            ntot, dx, dy = X.shape
            X = X.reshape(ntot, -1)
        elif which_transform == 'wavelet_scattering':
            fashion_mnist = datasets.FashionMNIST(root='data', train=True, download=True, transform=None)
            X = preprocess_data_fashionmnist(fashion_mnist)
            ntot, dx, dy = X.shape
            S = Scattering2D(shape=(dx,dy), J=3, L=8)
            X = S(X).reshape(ntot, -1)
        elif which_transform == 'random_gaussian_features':
            if which_non_lin == 'erf':
                fashion_mnist = datasets.FashionMNIST(root='data', train=True, download=True, transform=None)
                X = preprocess_data_fashionmnist(fashion_mnist)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1)
                ntot, d = X.shape
                F = np.random.normal(0., 1., size = (d, p))
                X = special.erf(X @ F)
            elif which_non_lin == 'tanh':
                fashion_mnist = datasets.FashionMNIST(root='data', train=True, download=True, transform=None)
                X = preprocess_data_fashionmnist(fashion_mnist)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1)
                ntot, d = X.shape
                F = np.random.normal(0., 1., size = (d, p))
                X = np.tanh(X @ F)
            elif which_non_lin == 'sign':
                fashion_mnist = datasets.FashionMNIST(root='data', train=True, download=True, transform=None)
                X = preprocess_data_fashionmnist(fashion_mnist)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1)
                ntot, d = X.shape
                F = np.random.normal(0., 1., size = (d, p))
                X = np.sign(X @ F)
            elif which_non_lin == 'relu':
                fashion_mnist = datasets.FashionMNIST(root='data', train=True, download=True, transform=None)
                X = preprocess_data_fashionmnist(fashion_mnist)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1)
                ntot, d = X.shape
                F = np.random.normal(0., 1., size = (d, p))
                X = relu(X @ F)
    elif which_dataset == "cifar10":
        if which_transform == 'no_transform':
            transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1), transforms.ToTensor()])
            cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            X = preprocess_data_cifar10(cifar10)
            ntot, dx, dy = X.shape
            X = X.reshape(ntot, -1)
        elif which_transform == 'wavelet_scattering':
            transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1), transforms.ToTensor()])
            cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            X = preprocess_data_cifar10(cifar10)
            ntot, dx, dy = X.shape
            S = Scattering2D(shape=(dx,dy), J=3, L=8)
            X = S(X).reshape(ntot, -1)
        elif which_transform == 'random_gaussian_features':
            if which_non_lin == 'erf':
                transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1), transforms.ToTensor()])
                cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
                X = preprocess_data_cifar10(cifar10)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1)
                ntot, d = X.shape
                F = np.random.normal(0., 1., size = (d, p))
                X = special.erf(X @ F)
            elif which_non_lin == 'tanh':
                transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1), transforms.ToTensor()])
                cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
                X = preprocess_data_cifar10(cifar10)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1)
                ntot, d = X.shape
                F = np.random.normal(0., 1., size = (d, p))
                X = np.tanh(X @ F)
            elif which_non_lin == 'sign':
                transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1), transforms.ToTensor()])
                cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
                X = preprocess_data_cifar10(cifar10)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1)
                ntot, d = X.shape
                F = np.random.normal(0., 1., size = (d, p))
                X = np.sign(X @ F)
            elif which_non_lin == 'relu':
                transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1), transforms.ToTensor()])
                cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
                X = preprocess_data_cifar10(cifar10)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1)
                ntot, d = X.shape
                F = np.random.normal(0., 1., size = (d, p))
                X = relu(X @ F)

    hf = h5py.File('./data/X_%s_%s_%s.hdf5'%(which_dataset, which_transform, which_non_lin), 'w')
    hf.create_dataset('X_%s_%s_%s'%(which_dataset, which_transform, which_non_lin), data=X)
    hf.close()
    return
