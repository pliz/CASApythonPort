# packages

from scipy import signal
from scipy.io.wavfile import read
from sklearn import preprocessing
from numpy.ma import masked_array as maska
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse import csr_matrix
import pylab as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.cluster import KMeans
from ica import pca_whiten
from sklearn.mixture import GaussianMixture

import ipdb
import scipy
import numpy as np
import pandas as pd
import seaborn as sb
import math


def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c


def smooth2d(mat2d, sigma=3, order=0):
    return gaussian_filter(mat2d, sigma, order)


def aspectrogram(x, fs=1, wlen=1024,
                 step=512,
                 freq_bins=1024):
    if freq_bins < wlen: freq_bins = wlen
    w, t, s = signal.spectrogram(x,
                                 fs=fs,
                                 window=np.hamming(wlen),
                                 nperseg=wlen,
                                 noverlap=step,
                                 nfft=2*freq_bins-1,
                                 mode='complex')
    return w, t, s


def time2specgrams(x1, x2, fs=1, wlen=1024, step=512, freq=1024):
    w, t, tf1 = aspectrogram(x1, fs=fs, wlen=wlen, step=step, freq_bins=freq)
    w, t, tf2 = aspectrogram(x2, fs=fs, wlen=wlen, step=step, freq_bins=freq)
    return w*2*np.pi, t, tf1, tf2 # converting to cyclic frequency


def alphadelta(tf1, tf2, w):
    """From spectrograms of 2 microphones compute symetric attenuation
    and delay at each time-frequency bin. DC component needs to be
    removed in all three.

    Arguments:
    - `tf1`: spectrogram from microphone 1 (omega by time)
    - `tf2`: spectrogram from microphone 2 (omega by time)
    - `w`: cyclic frequency
    """
    eps = np.spacing(1)
    R21 = (tf2+eps)/(tf1+eps)
    a = np.absolute(R21)
    alpha = a - 1/a
    delta = - np.angle(R21) / w[:,None]
    return alpha, delta


def tfweights(tf1, tf2, w):
    h1 = (np.absolute(tf1)*np.absolute(tf2))**p
    h2 = np.absolute(w)**q
    tfweight = h1*h2[:,None]
    return tfweight


def buildhist(alpha, delta, tfweight,
              maxa=3, maxd=3,
              abins=100, dbins=100,
              smooth=True,
              smoothstd=2.0):
    # only consider time-freq points yielding estimates in bounds
    amask = ~((np.absolute(alpha) < maxa) & (np.absolute(delta) < maxd))
    alphavec = maska(alpha, mask=amask).T.compressed()
    deltavec = maska(delta, mask=amask).T.compressed()
    tfweight = maska(tfweight, mask=amask).T.compressed()
    alphaind = np.around((abins-1)*(alphavec+maxa)/(2*maxa))
    deltaind = np.around((dbins-1)*(deltavec+maxd)/(2*maxd))

    A = csr_matrix((tfweight, (alphaind, deltaind)),
                   shape=(abins, dbins)).todense()
    if smooth: A = smooth2d(A, smoothstd)
    return A


def plot_hist(A, maxa=3, maxd=1, dbins=100, abins=100):
     fig = plt.figure()
     ax = fig.add_subplot(111, projection='3d')
     X=np.linspace(-maxd, maxd, dbins)
     Y=np.linspace(-maxa, maxa, abins)
     X, Y = np.meshgrid(X, Y)
     ax.plot_wireframe(X,Y,A)
     plt.show()

     
def gmm_clustering(points, num_clusters=2):
    gmm = GaussianMixture(num_clusters, 'full')
    gmm.fit(points)
    return gmm.means_


def get_mask(tf1, tf2, ratio = 0.1):
    t = np.absolute(tf1)*np.absolute(tf2)
    eps = ratio * t.max()
    amask = np.abs(t) > eps
    return amask


def peaks_gmm(alpha, delta, num_clusters, amask, **kwargs):
    points = np.c_[alpha[amask].flatten(), delta[amask].flatten()]
    return gmm_clustering(points, num_clusters)


def get_peaks(tf1, tf2, w, num_peaks, thresh=0.1):
    alpha, delta = alphadelta(tf1[1:,:], tf2[1:,:], w[1:])
    amask = get_mask(tf1[1:,:], tf2[1:,:], ratio=thresh)
    return peaks_gmm(alpha, delta, num_peaks, amask)


def get_a_d(tf1, tf2, w, num_peaks, **kwargs):
    pp = get_peaks(tf1, tf2, w, num_peaks, **kwargs)
    for i in range(pp.shape[0]):
        pp[i,0] = (pp[i,0] + np.sqrt(pp[i,0]**2 +4))/2.
    return pp


def segmentation(peaks, tf1, tf2, w):
    J = []
    for i in range(peaks.shape[0]):
        coef = peaks[i][0]*np.exp(-1j*peaks[i][1]*w)
        J.append(np.absolute(tf1*coef[:,None] - tf2)**2/(1+peaks[i][0]**2))
    return np.argmax(np.asarray(J),axis=0)


def sources(tf1, tf2, peaks, mask, w):
    s = []
    num_sources = np.unique(mask).shape[0]
    for i in range(num_sources):
        coef = peaks[i][0]*np.exp(1j*peaks[i][1]*w)
        s.append((mask == i) * (tf1 + coef[:,None]*tf2)/(1+peaks[i][0]**2))
    return s


def duet(x1, x2, fs, num_sources=2):
    w, t, tf1, tf2 = time2specgrams(x1, x2, fs=fs)
    peaks = get_a_d(tf1, tf2, w, 2)
    mask = segmentation(peaks, tf1, tf2, w)
    s = sources(tf1, tf2, peaks, mask, w)
    return s


def get_histogram(x1, x2, fs=1,
                  maxa=5, maxd=5,
                  abins=100, dbins=100,
                  p=1, q=0,
                  wlen=1024, timestep=512, numfreq=1024,
                  smooth=True, std=2.):

    if numfreq < wlen: numfreq = wlen
    # computing the spectrograms
    w, t, tf1, tf2 = time2specgrams(x1, x2, fs=fs,
                                    wlen=wlen, step=timestep,
                                    freq=numfreq)
    # removing DC component
    tf1 = tf1[1:, :]
    tf2 = tf2[1:, :]
    # attennuation and delay
    alpha, delta = alphadelta(tf1, tf2, w)
    # weights for the histogram
    tfweight = tfweights(tf1, tf2, w) # tf - time frequency
    # the histogram
    A = buildhist(alpha, delta, tfweight)

    return A



maxd = 3
dbins = 100
maxa = 10
abins = 100
p = 1
q = 0

# reading input files

fs, x1 = read('two_speakers.1.wav')
fs, x2 = read('two_speakers.2.wav')


# A = get_histogram(x1, x2, wlen=1024, timestep=512,
#                   maxa=maxa, maxd=maxd,
#                   abins=abins, dbins=dbins,
#                   p=p, q=q,
#                   smooth=True, std=2.0)
