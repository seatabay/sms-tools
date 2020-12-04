import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import os, sys
sys.path.append(os.path.join('../models/'))
import utilFunctions as UF
import dftModel as DFT
import hprModel as HPR
import stft as STFT

class Models:

    def __init__():
        pass

    def dft_model(inputFile, window, M, N, time):
        fs, x = UF.wavread(inputFile)

        # compute analysis window
        w = get_window(window, M)

        # get a fragment of the input sound of size M
        sample = int(time*fs)
        if (sample+M >= x.size or sample < 0):                          # raise error if time outside of sound
            raise ValueError("Time outside sound boundaries")
        x1 = x[sample:sample+M]

        # compute the dft of the sound fragment
        mX, pX = DFT.dftAnal(x1, w, N)

        # compute the inverse dft of the spectrum
        y = DFT.dftSynth(mX, pX, w.size)*sum(w)

        return fs, x1, mX, pX, y

    def harmonic_model(inputFile='../../sounds/vignesh.wav', window='blackman', M=1201, N=2048, t=-90,
    minSineDur=0.1, nH=100, minf0=130, maxf0=300, f0et=7, harmDevSlope=0.01):

        # size of fft used in synthesis
        Ns = 512

        # hop size (has to be 1/4 of Ns)
        H = 128

        # read input sound
        (fs, x) = UF.wavread(inputFile)

        # compute analysis window
        w = get_window(window, M)

        # detect harmonics of input sound
        hfreq, hmag, hphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)

        # synthesize the harmonics
        y = SM.sineModelSynth(hfreq, hmag, hphase, Ns, H, fs)

        # output sound file (monophonic with sampling rate of 44100)
        outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_harmonicModel.wav'

        # write the sound resulting from harmonic analysis
        UF.wavwrite(y, fs, outputFile)

        return fs, x, hfreq, y


    def hpr_model(inputFile='../../sounds/sax-phrase-short.wav', window='blackman', M=601, N=1024, t=-100,
    minSineDur=0.1, nH=100, minf0=350, maxf0=700, f0et=5, harmDevSlope=0.01):
            # size of fft used in synthesis
        Ns = 512

        # hop size (has to be 1/4 of Ns)
        H = 128

        # read input sound
        (fs, x) = UF.wavread(inputFile)

        # compute analysis window
        w = get_window(window, M)

        # find harmonics and residual
        hfreq, hmag, hphase, xr = HPR.hprModelAnal(x, fs, w, N, H, t, minSineDur, nH, minf0, maxf0, f0et, harmDevSlope)

        # compute spectrogram of residual
        mXr, pXr = STFT.stftAnal(xr, w, N, H)

        # synthesize hpr model
        y, yh = HPR.hprModelSynth(hfreq, hmag, hphase, xr, Ns, H, fs)

        # output sound file (monophonic with sampling rate of 44100)
        outputFileSines = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hprModel_sines.wav'
        outputFileResidual = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hprModel_residual.wav'
        outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hprModel.wav'

        # write sounds files for harmonics, residual, and the sum
        UF.wavwrite(yh, fs, outputFileSines)
        UF.wavwrite(xr, fs, outputFileResidual)
        UF.wavwrite(y, fs, outputFile)

        return fs, x, mXr, hfreq

    def hps_model():
            # size of fft used in synthesis
        Ns = 512

        # hop size (has to be 1/4 of Ns)
        H = 128

        # read input sound
        (fs, x) = UF.wavread(inputFile)

        # compute analysis window
        w = get_window(window, M)

        # compute the harmonic plus stochastic model of the whole sound
        hfreq, hmag, hphase, stocEnv = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf)

        # synthesize a sound from the harmonic plus stochastic representation
        y, yh, yst = HPS.hpsModelSynth(hfreq, hmag, hphase, stocEnv, Ns, H, fs)

        # output sound file (monophonic with sampling rate of 44100)
        outputFileSines = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hpsModel_sines.wav'
        outputFileStochastic = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hpsModel_stochastic.wav'
        outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hpsModel.wav'

        # write sounds files for harmonics, stochastic, and the sum
        UF.wavwrite(yh, fs, outputFileSines)
        UF.wavwrite(yst, fs, outputFileStochastic)
        UF.wavwrite(y, fs, outputFile)

        return fs, x, y, stocEnv

    def sine_model():
        # size of fft used in synthesis
        Ns = 512

        # hop size (has to be 1/4 of Ns)
        H = 128

        # read input sound
        fs, x = UF.wavread(inputFile)

        # compute analysis window
        w = get_window(window, M)

        # analyze the sound with the sinusoidal model
        tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)

        # synthesize the output sound from the sinusoidal representation
        y = SM.sineModelSynth(tfreq, tmag, tphase, Ns, H, fs)

        # output sound file name
        outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_sineModel.wav'

        # write the synthesized sound obtained from the sinusoidal synthesis
        UF.wavwrite(y, fs, outputFile)


    def spr_model(inputFile='../../sounds/bendir.wav', window='hamming', M=2001, N=2048, t=-80,
        minSineDur=0.02, maxnSines=150, freqDevOffset=10, freqDevSlope=0.001):
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
        tfreq, tmag, tphase, xr = SPR.sprModelAnal(x, fs, w, N, H, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope)

        # compute spectrogram of residual
        mXr, pXr = STFT.stftAnal(xr, w, N, H)

        # sum sinusoids and residual
        y, ys = SPR.sprModelSynth(tfreq, tmag, tphase, xr, Ns, H, fs)

        # output sound file (monophonic with sampling rate of 44100)
        outputFileSines = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_sprModel_sines.wav'
        outputFileResidual = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_sprModel_residual.wav'
        outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_sprModel.wav'

        # write sounds files for sinusoidal, residual, and the sum
        UF.wavwrite(ys, fs, outputFileSines)
        UF.wavwrite(xr, fs, outputFileResidual)
        UF.wavwrite(y, fs, outputFile)

        return None

    def sps_model(inputFile='../../sounds/bendir.wav', window='hamming', M=2001, N=2048, t=-80, minSineDur=0.02,
    maxnSines=150, freqDevOffset=10, freqDevSlope=0.001, stocf=0.2):
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
        tfreq, tmag, tphase, stocEnv = SPS.spsModelAnal(x, fs, w, N, H, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope, stocf)

        # synthesize sinusoidal+stochastic model
        y, ys, yst = SPS.spsModelSynth(tfreq, tmag, tphase, stocEnv, Ns, H, fs)

        # output sound file (monophonic with sampling rate of 44100)
        outputFileSines = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_spsModel_sines.wav'
        outputFileStochastic = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_spsModel_stochastic.wav'
        outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_spsModel.wav'

        # write sounds files for sinusoidal, residual, and the sum
        UF.wavwrite(ys, fs, outputFileSines)
        UF.wavwrite(yst, fs, outputFileStochastic)
        UF.wavwrite(y, fs, outputFile)

        return None

    def stft_model(inputFile = '../../sounds/piano.wav', window = 'hamming', M = 1024, N = 1024, H = 512):
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

        # output sound file (monophonic with sampling rate of 44100)
        outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_stft.wav'

        # write the sound resulting from the inverse stft
        UF.wavwrite(y, fs, outputFile)

        return None
    def stochastic_model(inputFile='../../sounds/ocean.wav', H=256, N=512, stocf=.1):
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

        outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_stochasticModel.wav'

        # write output sound
        UF.wavwrite(y, fs, outputFile)

        return None


    def plot_sound(x, fs, M, audio_filename, time=0):
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
        plt.plot(time + np.arange(M)/float(fs), x1)
        plt.axis([time, time + M/float(fs), min(x1), max(x1)])
        plt.ylabel('Amplitude')
        plt.xlabel('Time (s)')
        plt.title('Input Sound: {}'.format(audio_filename))
        plt.show()


    def plot_magnitude_phase_spectrum(fs, mag_phase, N,spec_type="Magnitude"):
        """
        Plot magnitude or phase spectrum.

        Parameters
        ----------
        fs:

        mag_phase:

        """

        if spec_type.lower() == "magnitude":
            y_label = "Amplitude (dB)"
        elif spec_type.lower() == "phase":
            y_label = "Phase (radians)"

        plt.plot(float(fs)*np.arange(mag_phase.size)/float(N), mag_phase, 'r')
        plt.axis([0, fs/2.0, min(mag_phase), max(mag_phase)])
        plt.ylabel(y_label)
        plt.xlabel('Frequency (Hz)')
        plt.title("{} Spectrum".format(spec_type))
        plt.show()


    def plot_magnitude_spectogram(fs, N, H, mXr, max_plot_freq=4000, figsize=(15,5), freq=None, plot_type=None):
        max_plot_bin = int(N * max_plot_freq / fs)
        maxplotbin = int(N*max_plot_freq/fs)
        num_frames = int(mXr[:,0].size)
        frame_time = H * np.arange(num_frames) / float(fs)
        bin_freq = np.arange(max_plot_bin + 1) * float(fs) / N
        plt.figure(figsize=figsize)
        plt.pcolormesh(frame_time, bin_freq, np.transpose(mXr[:,:maxplotbin+1]))
        plt.autoscale(tight=True)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        
        if plot_type == "H":
            Models.plot_harmonic_on_top(freq,fs,H,freq_type="Residual")
        elif plot_type == "S":
            Models.plot_sinusoidal_on_top(freq,frame_time,max_plot_freq,plot_type="Residual")
        else:
            plt.title("Magnitude Spectogram")
        plt.show()

    def plot_stochastic_spectogram(stocEnv, fs, H, max_plot_freq=5000):
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


    def plot_stochastic_representation(stocEnv, fs, H, N, stocf):
        num_frames= int(stocEnv[:,0].size)
        frame_time = H*np.arange(numFrames)/float(fs)                             
        bin_freq= np.arange(stocf*(N/2+1))*float(fs)/(stocf*N)                      
        plt.pcolormesh(frame_time, bin_freq, np.transpose(stocEnv))
        plt.autoscale(tight=True)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Stochastic Approximation')


    def plot_harmonic_sinusoidal_freqs(x, fs, freqs, H, max_plot_freq=4000,freq_type='harmonic'):
        """
        Plot harmonic or sinusoidal tracks.
        """
        if freqs.shape[1] > 0:
            num_frames = freqs.shape[0]
            frame_time = H * np.arange(num_frames) / float(fs)
            hfreq[hfreq <= 0] = np.nan
            plt.plot(frame_time, freqs)
            plt.axis([0, x.size/float(fs), 0, max_plot_freq])
            plt.title('Frequencies of {} Tracks'.format(freq_type.capitalize()))

            

    def plot_harmonic_on_top(hfreq, fs, H, color='k', alpha=1, max_plot_freq=4000, freq_type='Stochastic'):
            if hfreq.shape[1] > 0:
                harms = hfreq*np.less(hfreq, max_plot_freq)
                harms[harms == 0] = np.nan
                num_frames = int(harms[:,0].size)
                frame_time = H * np.arange(num_frames) / float(fs)
                plt.plot(frame_time, harms, color=color, ms=3, alpha=alpha)
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.autoscale(tight=True)
                plt.title('Harmonics + {} Spectogram'.format(freq_type))

    def plot_sinusoidal_on_top(tfreq,frame_time,max_plot_freq=5000,plot_type="Stochastic"):

        if (tfreq.shape[1] > 0):
            sines_tracks = tfreq*np.less(tfreq,max_plot_freq)

            if plot_type == "Stochastic":
                sines_tracks[sines_tracks==0] = np.nan
                num_frames = int(sines[:,0].size)
                frame_time = H*np.arange(num_frames)/float(fs)
            
            if plot_type == "Residual":
                sines_tracks = tfreq*np.less(tfreq, max_plot_freq)
                sines_tracks[sines_tracks<=0] = np.nan
            
            plt.plot(frame_time, sines_tracks, color='k', ms=3, alpha=1)
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency(Hz)')
            plt.autoscale(tight=True)
            plt.title('Sinusoidal + {} Spectrogram'.format(plot_type))
            plt.show()