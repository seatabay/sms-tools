import numpy as np
import numpy
import matplotlib.pyplot as plt
from scipy.signal import get_window
import os, sys
sys.path.append(os.path.join('../models/'))
import utilFunctions as UF

import ipywidgets as widget
from traitlets import link, HasTraits
from IPython.display import display
from traitlets import *
import math
import ipywidgets as widgets
import dftModel as DFT
import stft as STFT
import json
from collections import defaultdict
import pathlib
from ipywidgets import Layout

class GuisModels:
    
    def dft(self, inputFile, window, M, N, time=0):
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
    
    
    def stft(self,inputFile = '../../sounds/piano.wav', window = 'hamming', M = 1024, N = 1024, H = 512):
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
        
        return fs, mX, pX, y
    
    def harmonic(inputFile, window, M, N, t,
    minSineDur, nH, minf0, maxf0, f0et, harmDevSlope):

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

        return fs, x, hfreq, y


    def hpr(inputFile, window, M, N, t,
    minSineDur, nH, minf0, maxf0, f0et, harmDevSlope):
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

        return fs, x, mXr, hfreq

    def hps():
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

        return fs, x, y, stocEnv

    def sine():
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
        
        return fs, x, tfreq, y


    def spr(inputFile, window, M, N, t,
        minSineDur, maxnSines, freqDevOffset, freqDevSlope):
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

        return fs, x, tfreq, mXr, pXr, y, ys

    def sps(inputFile, window, M, N, t, minSineDur,
    maxnSines, freqDevOffset, freqDevSlope, stocf):
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

        return fs, x, tfreq, y, ys, yst

    def stochastic(inputFile, H, N, stocf=.1):
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


    def plot_magnitude_phase_spectrum(self,fs, mag_phase, N, spec_type="Magnitude"):
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
        mag_phase = np.array(mag_phase)
        plt.plot(float(fs)*np.arange(mag_phase.size)/float(N), mag_phase, 'r')
        plt.axis([0, fs/2.0, min(mag_phase), max(mag_phase)])
        plt.ylabel(y_label)
        plt.xlabel('Frequency (Hz)')
        plt.title("{} Spectrum".format(spec_type))
        plt.show()


    def plot_magnitude_spectogram(fs, N, mXr, max_plot_freq=4000.0,figsize=(15,5)):
        max_plot_bin = int(N * max_plot_freq / fs)
        num_frames = int(mXr[:,0].size)
        frame_time = H * np.arange(num_frames) / float(fs)
        bin_freq = np.arange(max_plot_bin + 1) * float(fs) / N
        plt.figure(figsize=figsize)
        plt.pcolormesh(frame_time, bin_freq, np.transpose(mXr[:,:maxplotbin+1]))
        plt.autoscale(tight=True)
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
                harms = hfreq*np.less(freq, max_plot_freq)
                harms[harms == 0] = np.nan
                num_frames = int(harms[:,0].size)
                frame_time = H * np.arange(num_frames) / float(fs)
                plt.plot(frame_time, harms, color=color, ms=3, alpha=alpha)
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.autoscale(tight=True)
                plt.title('Harmonics + {} Spectogram'.format(freq_type))

    def plot_sinusoidal_stochastic():

        if (tfreq.shape[1] > 0):
            sines = tfreq*np.less(tfreq,maxplotfreq)
            sines[sines==0] = np.nan
            numFrames = int(sines[:,0].size)
            frmTime = H*np.arange(numFrames)/float(fs)
            plt.plot(frmTime, sines, color='k', ms=3, alpha=1)
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency(Hz)')
            plt.autoscale(tight=True)
            plt.title('Sinusoidal + Stochastic Spectrogram')


    def plot_sinusoidal_residual():
        if (tfreq.shape[1] > 0):
            tracks = tfreq*np.less(tfreq, maxplotfreq)
            tracks[tracks<=0] = np.nan
            plt.plot(frmTime, tracks, color='k')
            plt.title('sinusoidal tracks + residual spectrogram')
            plt.autoscale(tight=True)

            


class GuiController:
    
    def read_json(self,filename):
        f = open(filename, "r") 
        data = json.loads(f.read()) 
        return data
    
    def limit_computation_number(self,json_file,data,key):
        json_key = [key for key in json_file.keys()]
        if len(json_file.keys()) < GuiView.KEEP_TRACK:
            json_file[key] = data
        else:
            del json_file[json_key[0]]
            json_file[key] = data
        return json_file
    
    def update_json(self,data, key):
        with open(f"{GuiView.DIR_NAME}/results.json".format(GuiView.DIR_NAME), 'r+') as f:
            json_data = json.load(f)
            json_data = self.limit_computation_number(json_data,data,key)
            f.seek(0)
            f.write(json.dumps(json_data))
            f.truncate()
            
    def compute_model_results(self,json, result_dict, model_dict,filename,call_model):
        #If model results are already computed, return it
        if len(json.keys()) != 0 and filename in json.keys():
            result_dict = json[filename]
        #Otherwise compute the results
        else:
            results = call_model(*tuple(result_dict.values()))
            for result,key in zip(results,model_dict.keys()):
                print(key)
                if isinstance(result,numpy.ndarray):
                    result_dict[key] = result.tolist()
                else:
                    result_dict[key] = result                
            #Keep the number of computations stable
            self.update_json(result_dict, filename)
        return result_dict

    def create_json(self,json_filepath):
        if not pathlib.Path(json_filepath).exists():
            #If no JSON is created, create one
            dump_json(defaultdict(dict),json_filepath)
            json = self.read_json(f"{GuiView.DIR_NAME}/results.json")
        else:
            #Otherwise read the json file
            json = self.read_json(json_filepath)
        return json

    def get_selected_audio_filename(self,widget):
        audio_filename = [audio for audio in widget.value.keys()]
        return audio_filename[0]

    def dump_json(self,data, filename):
         with open(filename, 'w') as fp:
            json.dump(data, fp, indent=4)

    def create_dir(self,dir_name):
        dir_name = "{}".format(dir_name)
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)


class GuiView:
    
    DFT = dict.fromkeys(["fs","x1","mX","pX","y"])
    STFT = dict.fromkeys(["fs","mX","pX","y"])
    HARMONIC = dict.fromkeys(["fs","x","hfreq","y"])
    HPR = dict.fromkeys(["fs","x","mXr","hfreq"])
    HPS = dict.fromkeys(["fs","x","y","stocenv"])
    SINE = dict.fromkeys(["fs","x","tfreq","y"])
    SPR = dict.fromkeys(["fs","x","tfreq","mXr","pXr","y","ys"])
    STOCHASTIC = dict.fromkeys(["fs","x","stocenv","y"])
    DIR_NAME = "Results"
    KEEP_TRACK = 3
    
    
    def __init__(self, model, controller):
        
        self.model = model
        self.controller = controller
        
        self.fft_value = [2**n for n in range(2,20)]
        self.style = {'description_width': 'initial'}
        self.window_type = widget.Dropdown(options=["Rectangular",
                                 "Hanning",
                                 "Hamming",
                                 "Blackman",
                                 "Blackmanharris"],
                                disabled=False,
                                 description="Window Type",
                                 style=self.style)
        self.window_size = widget.IntText(value=501
                                      ,disabled=False,description="Window Size (M)", style=self.style)
        self.fft_size = widget.Dropdown(options=self.fft_value,
                                   disabled=False, description="FFT-Size (N)", style=self.style)
        self.time = widget.FloatText(value=0.2,
                                 disabled=False, description="Time in sound (s)", style=self.style)
        self.hop_size = widget.IntText(value=512, description="Hop Size (H)", style=self.style)
    
        self.magnitude_threshold = widget.IntText(value=-80, description="Magnitude Threshold (t)", style=self.style)
    
        self.max_par_sinusoid = widget.IntText(value=150, 
                                      description="Maximum number of parallel sinusoid",
                                     style=self.style)
    
        self.min_dur_sinusoid = widget.FloatText(value=0.02, description="Minimum duration of sinusoidal track",
                                       style=self.style)
    
        self.max_freq_dev = widget.IntText(value=10,description="Max frequency deviation in sinusoidal tracks",style=self.style)
        
        self.slope_freq_dev = widget.FloatText(value=0.0001,description="Slope of frequency deviation", style=self.style)
        
        self.min_harmonic_dur = widget.FloatText(value=0.1,
                                           description="Minimum duration of harmonic tracks", style=self.style)
        
        self.max_harmonic_num = widget.IntText(value=100, description="Minimum number of harmonics", style=self.style)
        
        self.min_fundamental_freq = widget.IntText(value=130, description="Minimum fundamental frequency", style=self.style)
        
        self.max_fundamental_freq = widget.IntText(value=300, description="Maximum fundamental frequency", style=self.style)
        
        self.max_error_f0 = widget.IntText(value=7,description="Maximum error in f0 detection algorithm",style=self.style)
        
        self.max_freq_dev_harmonic = widget.FloatText(value=0.02, description="Maximum frequency deviation in harmonic tracks",
                                                style=self.style)
        
        self.dec_factor = widget.FloatRangeSlider(value=(0,1), min=0.1, max=1,step=0.001, description="Decimation factor(bigger than 0)",
                                            style=self.style)
        
        self.stochastic_app_fact = widget.FloatText(value=0.2, description="Stochastic approximation fact", style=self.style)
                
        
        self.select_audio = widgets.FileUpload(
            description="Select Audio File",
            accept='.wav',  
            multiple=False  
        )
        
        compute = widget.Button(description="Compute", button_style="warning", layout=widgets.Layout(width='50%', height='40px'))
        mag_spectrum = widget.Button(description="Magnitude Spectrum", button_style="warning", layout=widgets.Layout(width='50%', height='40px'))
        mag_spectogram = widget.Button(description="Magnitude Spectogram", button_style="warning", layout=widgets.Layout(width='50%', height='40px'))


        stochastic_rep = widget.Button(description="Stochastic Representation", button_style="info",
                                    layout=Layout(width='50%', height='40px'))
        
        input_sound = widget.Button(description="Play Input Sound", button_style="info",
                                    layout=Layout(width='50%', height='40px'))
        
        
        idft_res = widget.Button(description="IDFT Synthesized Sound",button_style="warning",
                              layout=Layout(width='50%', height='40px'))
        
        input_sound = widget.Button(description="Input Sound", button_style="success",
                                   layout=Layout(width='50%', height='40px'))
        output_sound = widget.Button(description="Output Sound", button_style="info",
                                    layout=Layout(width='50%', height='40px'))
        
        phase_spectogram = widget.Button(description="Phase Spectogram", button_style="info",
                                    layout=Layout(width='50%', height='40px'))
    

        self.dft = widget.VBox(children=[self.window_type,
                                    self.window_size,
                                    self.fft_size,
                                    self.time,
                                    self.select_audio,
                                    compute,
                                    mag_spectrum
                                    ])
        self.stft = widget.GridBox([self.window_type,
                                self.window_size,
                                self.fft_size,
                                self.hop_size,
                                self.select_audio,
                                compute,
                                mag_spectogram
                                ],layouts=widgets.Layout(grid_template_columns="repeat(2, 100px)"))
        
        self.sine = widget.VBox(children=[self.window_type,
                            self.window_size,
                            self.fft_size,
                            self.magnitude_threshold,
                            self.min_dur_sinusoid,
                            self.max_par_sinusoid,
                            self.max_freq_dev,
                            self.slope_freq_dev])

        self.harmonic = widget.VBox(children=[self.window_type,
                                        self.window_size,
                                        self.fft_size,
                                        self.magnitude_threshold,
                                        self.min_harmonic_dur,
                                        self.max_harmonic_num,
                                        self.min_fundamental_freq,
                                        self.max_fundamental_freq,
                                        self.max_error_f0,
                                        self.max_freq_dev_harmonic])

        self.stochastic = widget.VBox(children=[self.hop_size,
                                          self.fft_size,
                                          self.dec_factor])

        self.spr = widget.VBox(children=[self.window_type,
                                   self.window_size,
                                   self.fft_size,
                                   self.magnitude_threshold,
                                   self.min_dur_sinusoid,
                                   self.max_par_sinusoid,
                                   self.max_freq_dev,
                                   self.slope_freq_dev])

        self.sps = widget.VBox(children=[self.window_type,
                                   self.window_size,
                                   self.fft_size,
                                   self.magnitude_threshold,
                                   self.min_dur_sinusoid,
                                   self.max_par_sinusoid,
                                   self.max_freq_dev,
                                   self.slope_freq_dev,
                                   self.stochastic_app_fact])
        self.hpr = widget.VBox(children=[self.window_type,
                                   self.window_size,
                                   self.fft_size,
                                   self.magnitude_threshold,
                                   self.min_harmonic_dur,
                                   self.max_harmonic_num,
                                   self.min_fundamental_freq,
                                   self.max_fundamental_freq,
                                   self.max_error_f0,
                                   self.max_freq_dev])

        self.hps = widget.VBox(children=[self.window_type,
                                   self.window_size,
                                   self.fft_size,
                                   self.magnitude_threshold,
                                   self.min_dur_sinusoid,
                                   self.max_harmonic_num,
                                   self.min_fundamental_freq,
                                   self.max_fundamental_freq,
                                   self.max_error_f0,
                                   self.max_freq_dev_harmonic,
                                   self.stochastic_app_fact])

        
        self.tab = widget.Tab(children=[self.dft,self.stft,self.sine,self.harmonic,self.stochastic,self.spr,self.sps,self.hpr,self.hps])
        tab_names = ["DFT","STFT","SINE","HARMONIC","STOCHASTIC","SPR","SPS","HPR","HPS"]


        for tab_no, tab_name in zip(range(len(tab_names)),tab_names):
            self.tab.set_title(tab_no, tab_name)
            
        self.ipyview = widget.VBox(children=[self.tab])
        
        self.model_dict = {0:"dft",1:"stft"}

        
        def compute_results():
            
            #Create directory for results to be saved
            #If it does not exist
            self.controller.create_dir(GuiView.DIR_NAME)
            
            #Create a dictionary to save the results
            #Along with their keys
            self.result_dict = defaultdict(dict)
            
            """
            try:
                input_file = [audio for audio in self.select_audio.value.keys()]
            except ValueError:
                print("Select an audio!")
            if input_file != []:   
                self.dict["input_file"] = pathlib.Path(input_file[0]).with_suffix("")
            """
            
            #Get the model name to be used for computation
            self.model_name = self.model_dict[self.tab.selected_index]
            
            #Get the model function
            model = getattr(self, self.model_name)
            call_model = getattr(self.model, self.model_name)
            self.result_dict["input_file"] = self.controller.get_selected_audio_filename(self.select_audio)
            
            #Update the result dict with model parameters
            for item in model.children:
                if not hasattr(item,'button_style'):
                    if type(item.value) is str:
                        self.result_dict[item.description] = item.value.lower()
                    else:
                        self.result_dict[item.description] = item.value
            
            #Create the filename containing parameter values
            filename_structure = str("{}_" * (len(self.result_dict.values())+1))
            filename = filename_structure.format(self.model_name,*list(self.result_dict.values()))
            
            #Create a dictionary with relevant model parameters
            model_dict = getattr(GuiView, self.model_name.upper())
            
            #Create the json filepath
            json_filepath = pathlib.Path(GuiView.DIR_NAME + "/results.json")
            
            json = self.controller.create_json(json_filepath)
            
            self.result_dict = self.controller.compute_model_results(json, self.result_dict, model_dict, filename, call_model)
            
            return self.result_dict
                
                
        @mag_spectrum.on_click
        def plot(_):
            results = compute_results()
            self.model.plot_magnitude_phase_spectrum(results["fs"],results["mX"],results["Window Size (M)"])

        @mag_spectogram.on_click
        def plot_mag(_):
            results = compute_results()
            self.model.plot_magnitude_spectogram(results["fs"],results["N"], results["mX"])

        def display(self):
            display(self.ipyview)
    
