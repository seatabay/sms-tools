import os, sys

sys.path.append(os.path.join("../models/"))

import ipywidgets as widget
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import Layout

class GuiView:

    def __init__(self):
        
        self.fft_value = [2 ** n for n in range(2, 20)]
        self.style = {"description_width": "initial"}
        self.window = widget.Dropdown(
            options=["Rectangular", "Hanning", "Hamming", "Blackman", "Blackmanharris"],
            disabled=False,
            description="Window Type",
            style=self.style,
            description_tooltip="window"
        )
        self.M = widget.IntText(
            value=501, disabled=False, description="Window Size (M)", style=self.style,
            description_tooltip="M"
        )
        self.N = widget.Dropdown(
            options=self.fft_value,
            disabled=False,
            description="FFT-Size (N)",
            style=self.style,
            description_tooltip = "N"
        )
        self.time = widget.FloatText(
            value=0.2, disabled=False, description="Time in sound (s)", style=self.style,
            description_tooltip="time"
        )
        self.H = widget.IntText(
            value=512, description="Hop Size (H)", style=self.style, description_tooltip="H"
        )

        self.magnitude_threshold = widget.IntText(
            value=-80, description="Magnitude Threshold (t)", style=self.style, description_tooltip="t"
        )


        self.max_par_sinusoid = widget.IntText(
            value=150,
            description="Maximum number of parallel sinusoid",
            style=self.style,
            description_tooltip="maxnSines"
        )

        self.min_dur_sinusoid = widget.FloatText(
            value=0.02,
            description="Minimum duration of sinusoidal track",
            style=self.style,
            description_tooltip="minSineDur"
        )

        self.max_freq_dev = widget.IntText(
            value=10,
            description="Max frequency deviation in sinusoidal tracks",
            style=self.style,
            description_tooltip="freqDevOffset"
        )

        self.slope_freq_dev = widget.FloatText(
            value=0.01, description="Slope of frequency deviation", style=self.style,
            description_tooltip="freqDevSlope"
        )

        self.min_harmonic_dur = widget.FloatText(
            value=0.1,
            description="Minimum duration of harmonic tracks",
            style=self.style,
        )

        self.max_harmonic_num = widget.IntText(
            value=100, description="Minimum number of harmonics", style=self.style,
            description_tooltip="nH"
        )

        self.min_fundamental_freq = widget.IntText(
            value=130, description="Minimum fundamental frequency", style=self.style,
            description_tooltip="minf0"
        )

        self.max_fundamental_freq = widget.IntText(
            value=300, description="Maximum fundamental frequency", style=self.style,
            description_tooltip="maxf0"
        )

        self.max_error_f0 = widget.IntText(
            value=7,
            description="Maximum error in f0 detection algorithm",
            style=self.style,
            description_tooltip="f0et"
        )

        self.max_freq_dev_harmonic = widget.FloatText(
            value=0.02,
            description="Maximum frequency deviation in harmonic tracks",
            style=self.style,
            description_tooltip="harmDevSlope"
        )

        self.dec_factor = widget.FloatRangeSlider(
            value=(0, 1),
            min=0.1,
            max=1,
            step=0.001,
            description="Decimation factor(bigger than 0)",
            style=self.style,
            description_tooltip="stocf"
        )

        self.stochastic_app_fact = widget.FloatText(
            value=0.2, description="Stochastic approximation fact", style=self.style,
            description_tooltip="stocf"
        )

        self.input_file = widgets.FileUpload(
            description="Select Audio File", accept=".wav", multiple=False, description_tooltip="inputFile"
        )


        self.mag_spectrum = widget.Button(
            description="Magnitude Spectrum",
            button_style="warning",
            layout=widgets.Layout(width="50%", height="40px"),
        )
        self.phase_spectrum = widget.Button(
            description="Phase Spectrum",
            button_style="warning",
            layout=widgets.Layout(width="50%", height="40px"),
        )
        self.mag_spectogram = widget.Button(
            description="Magnitude Spectogram",
            button_style="warning",
            layout=widgets.Layout(width="50%", height="40px"),
        )
        self.phase_spectogram = widget.Button(
            description="Phase Spectogram",
            button_style="warning",
            layout=widgets.Layout(width="50%", height="40px"),
        )

        self.stochastic_rep = widget.Button(
            description="Stochastic Representation",
            button_style="info",
            layout=Layout(width="50%", height="40px"),
        )

        self.sinusoidal_tracks = widget.Button(
            description="Frequencies of Sinusoidal Tracks",
            button_style="info",
            layout=Layout(width="50%", height="40px"),
        )
        self.harmonic_tracks = widget.Button(
            description="Frequencies of Harmonic Tracks",
            button_style="info",
            layout=Layout(width="50%", height="40px"),
        )

        self.idft_res = widget.Button(
            description="IDFT Synthesized Sound",
            button_style="warning",
            layout=Layout(width="50%", height="40px"),
        )

        self.input_sound = widget.Button(
            description="Input Sound",
            button_style="success",
            layout=Layout(width="50%", height="40px"),
        )
        self.output_sound = widget.Button(
            description="Output Sound",
            button_style="info",
            layout=Layout(width="50%", height="40px"),
        )

        self.phase_spectogram = widget.Button(
            description="Phase Spectogram",
            button_style="info",
            layout=Layout(width="50%", height="40px"),
        )

        self.sinusoidal_audio = widget.Button(
            description="Sinusoidal Output",
            button_style="success",
            layout=Layout(width="50%", height="40px"),
        )
        self.residual_audio = widget.Button(
            description="Residual Output",
            button_style="success",
            layout=Layout(width="50%", height="40px"),
        )

        self.stochastic_audio = widget.Button(
            description="Stochastic Output",
            button_style="success",
            layout=Layout(width="50%", height="40px"),
        )
        self.sinusoidal_residual = widget.Button(
            description="Sinusoidal + Residual",
            button_style="warning",
            layout=widgets.Layout(width="50%", height="40px"),
        )
        
        self.sinusoidal_stochastic = widget.Button(
            description="Sinusoidal + Stochastic",
            button_style="warning",
            layout=widgets.Layout(width="50%", height="40px"),
        )

        self.harmonic_residual = widget.Button(
            description="Harmonic + Residual",
            button_style="warning",
            layout=widgets.Layout(width="50%", height="40px"))
        self.harmonic_stochastic = widget.Button(
            description="Harmonic + Stochastic",
            button_style="warning",
            layout=widgets.Layout(width="50%", height="40px"))
        

        self.dft = widget.VBox(
            children=[
                self.window,
                self.M,
                self.N,
                self.time,
                self.input_file,
                self.input_sound,
                self.mag_spectrum,
                self.phase_spectrum,
                self.output_sound,
            ]
        )
        self.stft = widget.GridBox(
            [
                self.window,
                self.M,
                self.N,
                self.H,
                self.input_file,
                self.input_sound,
                self.mag_spectogram,
                self.phase_spectogram,
                self.output_sound,
            ],
            layouts=widgets.Layout(grid_template_columns="repeat(2, 100px)"),
        )

        self.sine = widget.VBox(
            children=[
                self.window,
                self.M,
                self.N,
                self.magnitude_threshold,
                self.min_dur_sinusoid,
                self.max_par_sinusoid,
                self.max_freq_dev,
                self.slope_freq_dev,
                self.input_file,
                self.input_sound,
                self.sinusoidal_tracks,
                self.output_sound,
            ]
        )

        self.harmonic = widget.VBox(
            children=[
                self.window,
                self.M,
                self.N,
                self.magnitude_threshold,
                self.min_harmonic_dur,
                self.max_harmonic_num,
                self.min_fundamental_freq,
                self.max_fundamental_freq,
                self.max_error_f0,
                self.max_freq_dev_harmonic,
                self.input_file,
                self.input_sound,
                self.harmonic_tracks,
                self.output_sound,
            ]
        )

        self.stochastic = widget.VBox(
            children=[
                self.H,
                self.N,
                self.dec_factor,
                self.input_file,
                self.input_sound,
                self.stochastic_rep,
                self.output_sound,
            ]
        )

        self.spr = widget.VBox(
            children=[
                self.window,
                self.M,
                self.N,
                self.magnitude_threshold,
                self.min_dur_sinusoid,
                self.max_par_sinusoid,
                self.max_freq_dev,
                self.slope_freq_dev,
                self.input_file,
                self.input_sound,
                self.sinusoidal_residual,
                self.sinusoidal_audio,
                self.residual_audio,
                self.output_sound,
            ]
        )

        self.sps = widget.VBox(
            children=[
                self.window,
                self.M,
                self.N,
                self.magnitude_threshold,
                self.min_dur_sinusoid,
                self.max_par_sinusoid,
                self.max_freq_dev,
                self.slope_freq_dev,
                self.stochastic_app_fact,
                self.input_file,
                self.input_sound,
                self.sinusoidal_stochastic,
                self.sinusoidal_audio,
                self.stochastic_audio,
                self.output_sound,
            ]
        )

        self.hpr = widget.VBox(
            children=[
                self.window,
                self.M,
                self.N,
                self.magnitude_threshold,
                self.min_harmonic_dur,
                self.max_harmonic_num,
                self.min_fundamental_freq,
                self.max_fundamental_freq,
                self.max_error_f0,
                self.max_freq_dev,
                self.input_file,
                self.input_sound,
                self.harmonic_residual,
                self.sinusoidal_audio,
                self.residual_audio,
                self.output_sound,
            ]
        )

        self.hps = widget.VBox(
            children=[
                self.window,
                self.M,
                self.N,
                self.magnitude_threshold,
                self.min_dur_sinusoid,
                self.max_harmonic_num,
                self.min_fundamental_freq,
                self.max_fundamental_freq,
                self.max_error_f0,
                self.max_freq_dev_harmonic,
                self.stochastic_app_fact,
                self.input_file,
                self.input_sound,
                self.harmonic_stochastic,
                self.sinusoidal_audio,
                self.stochastic_audio,
                self.output_sound,
            ]
        )

        self.tab = widget.Tab(
            children=[
                self.dft,
                self.stft,
                self.sine,
                self.harmonic,
                self.stochastic,
                self.spr,
                self.sps,
                self.hpr,
                self.hps,
            ]
        )

        self.model_names = {
            0: "dft",
            1: "stft",
            2: "sine",
            3: "harmonic",
            4: "stochastic",
            5: "spr",
            6: "sps",
            7: "hpr",
            8: "hps",
        }

        for tab_no, tab_name in zip(self.model_names.keys(), self.model_names.values()):
            self.tab.set_title(tab_no, tab_name.upper())

        self.ipyview = widget.VBox(children=[self.tab])

    def display(self):
        display(self.ipyview)
        