import numpy as np
import numpy
import matplotlib.pyplot as plt
from scipy.signal import get_window
import os, sys
sys.path.append(os.path.join('../models/'))
import utilFunctions as UF
import inspect

import ipywidgets as widget
from traitlets import link, HasTraits
from IPython.display import display
from traitlets import *
import math
import ipywidgets as widgets
import dftModel as DFT
import stft as STFT
import sineModel as SM
import stochasticModel as STM
import harmonicModel as HM
import sprModel as SPR
import spsModel as SPS
import hprModel as HPR
import hpsModel as HPS
import json
from collections import defaultdict
import pathlib
from ipywidgets import Layout
import pydub
from pydub.playback import play


def dft(window, M, N, time ,inputFile):
    fs, x_org = UF.wavread(inputFile)
    # compute analysis window
    w = get_window(window, M)

    # get a fragment of the input sound of size M
    sample = int(time*fs)
    if (sample+M >= x_org.size or sample < 0):                          # raise error if time outside of sound
        raise ValueError("Time outside sound boundaries")
    x = x_org[sample:sample+M]

    # compute the dft of the sound fragment
    mX, pX = DFT.dftAnal(x, w, N)

    # compute the inverse dft of the spectrum
    y = DFT.dftSynth(mX, pX, w.size)*sum(w)
    
    return fs, x, mX, pX, y


def stft(window, M, N, H, inputFile):
    """
    analysis/synthesis using the STFT
    inputFile: input sound file (monophonic with sampling rate of 44100)
    window: analysis window type (choice of rectangular, hanning, hamming, blackman, blackmanharris)
    M: analysis window size
    N: fft size (power of two, bigger or equal than M)
    H: hop size (at least 1/2 of analysis window size to have good overlap-add)
    """

    # read input sound (monophonic with sampling rate of 44100)
    fs, x = UF.wavread(inputFile)

    # compute analysis window
    w = get_window(window, M)

    # compute the magnitude and phase spectrogram

    mX, pX = STFT.stftAnal(x, w, N, H)

    # perform the inverse stft
    y = STFT.stftSynth(mX, pX, M, H)
    
    return x,fs, mX, pX, y

def harmonic(window, M, N, t,
minSineDur, nH, minf0, maxf0, f0et, harmDevSlope, inputFile):

    # size of fft used in synthesis
    Ns = 512

    # hop size (has to be 1/4 of Ns)
    H = 128

    # read input sound
    (fs, x) = UF.wavread(inputFile)

    # compute analysis window
    w = get_window(window, M)

    # detect harmonics of input sound
    freq, mag, phase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)

    # synthesize the harmonics
    y = SM.sineModelSynth(freq, mag, phase, Ns, H, fs)

    return fs, x, freq, y, H


def hpr(window, M, N, t, minSineDur, nH, minf0, maxf0, f0et, harmDevSlope, inputFile):
        # size of fft used in synthesis
    Ns = 512

    # hop size (has to be 1/4 of Ns)
    H = 128

    # read input sound
    (fs, x) = UF.wavread(inputFile)

    # compute analysis window
    w = get_window(window, M)

    # find harmonics and residual
    freq, mag, phase, xr = HPR.hprModelAnal(x, fs, w, N, H, t, minSineDur, nH, minf0, maxf0, f0et, harmDevSlope)

    # compute spectrogram of residual
    mX, pX = STFT.stftAnal(xr, w, N, H)

    # synthesize hpr model
    y, ys = HPR.hprModelSynth(freq, mag, phase, xr, Ns, H, fs)

    return fs, x, mX, freq, y, ys, xr, H

def hps(window, M, N, t, minSineDur, nH, minf0, maxf0, f0et, harmDevSlope, stocf,inputFile):
        # size of fft used in synthesis
    Ns = 512

    # hop size (has to be 1/4 of Ns)
    H = 128

    # read input sound
    (fs, x) = UF.wavread(inputFile)

    # compute analysis window
    w = get_window(window, M)

    # compute the harmonic plus stochastic model of the whole sound
    freq, mag, phase, stocEnv = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf)

    # synthesize a sound from the harmonic plus stochastic representation
    y, ys, yst = HPS.hpsModelSynth(freq, mag, phase, stocEnv, Ns, H, fs)

    return fs, x, stocEnv, freq, y, ys, yst, H

def sine(window, M, N, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope, inputFile):
    # size of fft used in synthesis
    Ns = 512

    # hop size (has to be 1/4 of Ns)
    H = 128

    # read input sound
    fs, x = UF.wavread(inputFile)

    # compute analysis window
    w = get_window(window, M)

    # analyze the sound with the sinusoidal model
    freq, mag, phase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)

    # synthesize the output sound from the sinusoidal representation
    y = SM.sineModelSynth(freq, mag, phase, Ns, H, fs)
    
    return fs, x, freq, y, H


def spr(window, M, N, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope, inputFile):
    """
    inputFile: input sound file (monophonic with sampling rate of 44100)
    window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)
    M: analysis window size
    N: fft size (power of two, bigger or equal than M)
    t: magnitude threshold of spectral peaks
    minSineDur: minimum duration of sinusoidal tracks
    maxnSines: maximum number of parallel sinusoids
    freqDevOffset: frequency deviation allowed in the sinusoids from frame to frame at frequency 0
    freqDevSlope: slope of the frequency deviation, higher frequencies have bigger deviation
    """

    # size of fft used in synthesis
    Ns = 512

    # hop size (has to be 1/4 of Ns)
    H = 128

    # read input sound
    (fs, x) = UF.wavread(inputFile)

    # compute analysis window
    w = get_window(window, M)

    # perform sinusoidal plus residual analysis
    freq, mag, phase, xr = SPR.sprModelAnal(x, fs, w, N, H, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope)

    # compute spectrogram of residual
    mX, pX = STFT.stftAnal(xr, w, N, H)

    # sum sinusoids and residual
    y, ys = SPR.sprModelSynth(freq, mag, phase, xr, Ns, H, fs)

    return fs, x, freq, mX, pX, y, ys, H, xr

def sps(window, M, N, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope, stocf, inputFile):
    """
    inputFile: input sound file (monophonic with sampling rate of 44100)
    window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)
    M: analysis window size; N: fft size (power of two, bigger or equal than M)
    t: magnitude threshold of spectral peaks; minSineDur: minimum duration of sinusoidal tracks
    maxnSines: maximum number of parallel sinusoids
    freqDevOffset: frequency deviation allowed in the sinusoids from frame to frame at frequency 0
    freqDevSlope: slope of the frequency deviation, higher frequencies have bigger deviation
    stocf: decimation factor used for the stochastic approximation
    """

    # size of fft used in synthesis
    Ns = 512

    # hop size (has to be 1/4 of Ns)
    H = 128

    # read input sound
    (fs, x) = UF.wavread(inputFile)

    # compute analysis window
    w = get_window(window, M)

    # perform sinusoidal+sotchastic analysis
    freq, mag, phase, stocEnv = SPS.spsModelAnal(x, fs, w, N, H, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope, stocf)

    # synthesize sinusoidal+stochastic model
    y, ys, yst = SPS.spsModelSynth(freq, mag, phase, stocEnv, Ns, H, fs)
    
    return fs, x, freq, stocEnv, y, ys, yst, H


def stochastic(H, N, stocf,inputFile):
    """
    inputFile: input sound file (monophonic with sampling rate of 44100)
    H: hop size, N: fft size
    stocf: decimation factor used for the stochastic approximation (bigger than 0, maximum 1)
    """

    # read input sound
    (fs, x) = UF.wavread(inputFile)

    # compute stochastic model
    stocEnv = STM.stochasticModelAnal(x, H, N, stocf)

    # synthesize sound from stochastic model
    y = STM.stochasticModelSynth(stocEnv, H, N)

    return fs, x, stocEnv, y


def plot_sound(x, fs, M, inputFile,time=0):
    """
    Plot a fragment of given sound.

    Parameters
    ----------
    x:

    fs:

    M:

    audio_filename:

    time:

    """
    x = np.array(x)

    if time > 0:

        plt.figure(figsize=(10,5))
        plt.plot(time + np.arange(M)/float(fs), x)
        plt.axis([time, time + M/float(fs), min(x), max(x)])
        plt.ylabel('Amplitude')
        plt.xlabel('Time (s)')
        plt.title('Input Sound: {}'.format(inputFile))
        plt.show()
    else:
        plt.plot(np.arange(x.size)/float(fs), x)
        plt.axis([0, x.size/float(fs), min(x), max(x)])
        plt.ylabel('Amplitude')
        plt.xlabel('Time (s)')
        plt.title('Input Sound: {}'.format(inputFile))
        plt.show()

def plot_magnitude_spectrum(fs, mX, N, filename):
    """
    Plot magnitude or phase spectrum.

    Parameters
    ----------
    fs:

    mX:

    """
    mX = np.array(mX)
    plt.figure(figsize=(10,5))
    plt.plot(float(fs)*np.arange(mX.size)/float(N), mX, 'r')
    plt.axis([0, fs/2.0, min(mX), max(mX)])
    plt.ylabel("Magnitude")
    plt.xlabel('Frequency (Hz)')
    plt.title("Magnitude Spectrum")
    plt.suptitle(filename, fontsize=18)
    plt.show()
    
    
def plot_phase_spectrum(fs, pX, N, filename):
    """
    Plot magnitude or phase spectrum.

    Parameters
    ----------
    fs:

    mX:

    """
    
    pX = np.array(pX)
    plt.figure(figsize=(10,5))
    plt.plot(float(fs)*np.arange(pX.size)/float(N), pX, 'r')
    plt.axis([0, fs/2.0, min(pX), max(pX)])
    plt.ylabel("Phase (radians)")
    plt.xlabel('Frequency (Hz)')
    plt.title("Phase Spectrum")
    plt.suptitle(filename, fontsize=18)
    plt.show()


def plot_magnitude_spectogram(fs, N, H, mX, filename, model ,freq=None,max_plot_freq=4000.0):
    mX = np.array(mX)
    freq = np.array(freq)
    max_plot_bin = int(N * max_plot_freq / fs)
    num_frames = int(mX[:,0].size)
    frame_time = H * np.arange(num_frames) / float(fs)
    bin_freq = np.arange(max_plot_bin + 1) * float(fs) / N
    plt.pcolormesh(frame_time, bin_freq, np.transpose(mX[:,:max_plot_bin+1]))
        
    title = "Magnitude Spectogram"
    if model == "spr":
        if freq.shape[1] > 0:
            tracks = freq * np.less(freq, max_plot_freq)
            tracks[tracks <= 0] = np.nan
            plt.plot(frame_time, tracks, color="k")
            title="Sinusoidal Tracks + Residual Spectrogram"
            plt.autoscale(tight=True)

    elif model == "hpr":

        if freq.shape[1] > 0:
            harms = freq * np.less(freq, max_plot_freq)
            harms[harms == 0] = np.nan
            num_frames = int(harms[:, 0].size)
            frame_time = H * np.arange(num_frames) / float(fs)
            plt.plot(frame_time, harms, color="k", ms=3, alpha=1)
            plt.autoscale(tight=True)
            title = "Harmonics + Residual Spectogram"
            plt.title("Harmonics + Residual Spectogram")
            
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.suptitle(filename, fontsize=18)
    plt.autoscale(tight=True)


    plt.show()
    
def plot_phase_spectogram(fs, N, H, pX, filename, max_plot_freq=4000.0):
    pX = np.array(pX)
    numFrames = int(pX[:,0].size)
    frmTime = H*np.arange(numFrames)/float(fs)
    binFreq = fs*np.arange(N*max_plot_freq/fs)/N
    plt.pcolormesh(frmTime, binFreq, np.transpose(np.diff(pX[:,:int(N*max_plot_freq/fs+1)],axis=1)))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Phase Spectrogram (derivative)')
    plt.suptitle(filename, fontsize=18)
    plt.autoscale(tight=True)
    plt.show()


def plot_stochastic_spectogram(stocEnv, freq,fs, H, model,max_plot_freq=4000.0):
    stocEnv = np.array(stocEnv)
    freq = np.array(freq)
    num_frames = int(stocEnv[:,0].size)
    size_env = int(stocEnv[0,:].size)
    frame_time = H * np.arange(num_frames) / float(fs)
    ##EXPLAIN AND GIVE A MEANINGFUL NAME TO VAR -- VARIABLE
    var = .5*fs
    var_2 = size_env * max_plot_freq / var
    bin_freq = var * np.arange(var_2) / size_env
    plt.pcolormesh(frame_time, bin_freq, np.transpose(stocEnv[:, :int(var_2)+1]))
    plt.autoscale(tight=True)
    plt.show()
    
    if model == "sps":

        sines = freq*np.less(freq,max_plot_freq)
        sines[sines==0] = np.nan
        num_frames = int(sines[:,0].size)
        frame_time = H*np.arange(num_frames)/float(fs)
        plt.plot(frame_time, sines, color='k', ms=3, alpha=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency(Hz)')
        plt.autoscale(tight=True)
        plt.title('Sinusoidal + Stochastic Spectrogram')
        
    elif model == "hps":
        if freq.shape[1] > 0:
            harms = freq*np.less(freq, max_plot_freq)
            harms[harms == 0] = np.nan
            num_frames = int(harms[:,0].size)
            frame_time = H * np.arange(num_frames) / float(fs)
            plt.plot(frame_time, harms,color="k", ms=3, alpha=1)
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.autoscale(tight=True)
            plt.title('Harmonics + Stochastic Spectogram')



def plot_stochastic_representation(stocEnv, fs, H, N, stocf, model):
    stocEnv = np.array(stocEnv)
    num_frames= int(stocEnv[:,0].size)
    frame_time = H*np.arange(num_frames)/float(fs)                             
    bin_freq= np.arange(stocf*(N/2+1))*float(fs)/(stocf*N)                      
    plt.pcolormesh(frame_time, bin_freq, np.transpose(stocEnv))
    plt.autoscale(tight=True)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Stochastic Approximation')


def plot_harmonic_sinusoidal_freqs(x, fs, freq, H,model,max_plot_freq=4000.0):
    """
    Plot harmonic or sinusoidal tracks.
    """
    freq = np.array(freq)
    x = np.array(x)
    if freq.shape[1] > 0:
        num_frames = freq.shape[0]
        frame_time = H * np.arange(num_frames) / float(fs)
        freq[freq <= 0] = np.nan
        plt.plot(frame_time, freq)
        plt.axis([0, x.size/float(fs), 0, max_plot_freq])
        plt.title('Frequencies of {} Tracks'.format(model.capitalize()))
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.show()


def play_sound(y,fs,filename):
    y = np.array(y)
    filename = "_".join(filename.split("/"))
    UF.wavwrite(y, fs, filename)
    music = pydub.AudioSegment.from_file(filename)
    play(music)