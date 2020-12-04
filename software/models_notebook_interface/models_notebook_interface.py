class Models:
    
    
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
        
        
    def plot_magnitude_phase_spectrum(fs, mag_phase, spec_type="Magnitude"):
        """
        Plot magnitude or phase spectrum.
        
        Parameters
        ----------
        fs:
        
        mag_phase:
        
        """
        
        if spec_type.lower() = "magnitude":
            y_label = "Amplitude (dB)"
        elif spec_type.lower() = "phase":
            y_label = "Phase (radians)"
            
        plt.plot(float(fs)*np.arange(mag_phase.size)/float(N), mag_phase, 'r')
        plt.axis([0, fs/2.0, min(mag_phase), max(mag_phase)])
        plt.ylabel(y_label)
        plt.xlabel('Frequency (Hz)')
        plt.title("{} Spectrum".format(spec_type))
        plt.show()
        
            
    def plot_magnitude_spectogram(fs, N, mXr, figsize=(15,5)):
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
        try:
        if freqs.shape[1] > 0:
            num_frames = freqs.shape[0]
            frame_time = H * np.arange(num_frames) / float(fs)
            hfreq[hfreq <= 0] = np.nan
            plt.plot(frame_time, freqs)
            plt.axis([0, x.size/float(fs), 0, max_plot_freq])
            plt.title('Frequencies of {} Tracks'.format(freq_type.capitalize()))
        else raise:...
    
            
    def plot_harmonic_on_top(hfreq, fs, H, color='k', alpha=1, max_plot_freq=4000, freq_type='Stochastic'):
    
        try:
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
        else:
            pass
    
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
