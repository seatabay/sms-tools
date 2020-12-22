from class_models import *
import inspect
import matplotlib
class GuiController(object):
    
    
    def __init__(self, model, view):
        self.model = model
        self.view = view
        
        self.view.display()
        
        self.model.create_dir()
        
    
        def main():
            self.model_name = self.view.model_names[self.view.tab.selected_index]
            self.model_widget = getattr(self.view, self.model_name)
            results,filename = self.model.compute_model_results(self.model_name, self.model_widget)
            return results,filename
        
        
        def get_plot_parameters(func, results):
            plot_args = inspect.getfullargspec(func)[0]
            
            plot_result = []
            for arg in plot_args:
                if arg in results.keys():
                    plot_result.append(results[arg])
            return plot_result
        
    
        @view.input_sound.on_click
        def plot_input(_):
            results, _= main()
            plot_args= get_plot_parameters(plot_sound, results)
                
            plot_sound(*tuple(plot_args))
            
        @view.mag_spectrum.on_click
        def show_mag_spectrum(_):
            results, _ = main()
            
            plot_args = get_plot_parameters(plot_magnitude_spectrum, results)
                
            plot_magnitude_spectrum(*tuple(plot_args))
            
        @view.phase_spectrum.on_click
        def show_phase_spectrum(_):
            results,_ = main()
            
            plot_args = get_plot_parameters( plot_phase_spectrum, results)
            plot_phase_spectrum(*tuple(plot_args))
            
        @view.phase_spectogram.on_click
        def show_phase_spectogram(_):
            results, _ = main()
            
            plot_args = get_plot_parameters(plot_phase_spectogram, results)
            plot_phase_spectogram(*tuple(plot_args))
            
        @view.mag_spectogram.on_click
        def show_mag_spectogram(_):
            results, _ = main()
            plot_args = get_plot_parameters(plot_magnitude_spectogram, results)
            plot_magnitude_spectogram(*tuple(plot_args))
            
        @view.sinusoidal_tracks.on_click
        def show_sinusoidal_tracks(_):
            results, _ = main()
            plot_args = get_plot_parameters( plot_harmonic_sinusoidal_freqs, results)
            
            plot_harmonic_sinusoidal_freqs(*tuple(plot_args))
            
        @view.harmonic_tracks.on_click
        def show_harmonic_tracks(_):
            
            results, _ = main()
            plot_args = get_plot_parameters(plot_harmonic_sinusoidal_freqs, results)
            
            plot_harmonic_sinusoidal_freqs(*tuple(plot_args))
            
        @view.stochastic_rep.on_click
        def show_stochastic_rep(_):
            
            results, _ = main()
            plot_args = get_plot_parameters(plot_stochastic_representation, results)
            
            plot_stochastic_representation(*tuple(plot_args))
            
        @view.output_sound.on_click
        def play_sound(_):
            
            results, filename = main()
            plot_args = get_plot_parameters(self.model.play_sound, results)
            self.model.play_sound(*tuple(plot_args))
            
        @view.residual_audio.on_click
        def play_residual_output(_):
            
            results, filename = main()
            self.model.play_sound(results["xr"], results["fs"],results["filename"])
        
        @view.sinusoidal_audio.on_click
        def play_sinusoidal_audio(_):
            
            results, filename = main()
            self.model.play_sound(results["ys"],results["fs"],results["filename"])
        
        @view.stochastic_audio.on_click
        def play_stochastic_audio(_):
            
            results, filename = main()
            self.model.play_sound(results["yst"],results["fs"],results["filename"])
            
        @view.sinusoidal_residual.on_click
        def show_sinusoidal_residual(_):
            results, _ = main()
            plot_args = get_plot_parameters(plot_magnitude_spectogram, results)
            plot_magnitude_spectogram(*tuple(plot_args))
            
        @view.harmonic_residual.on_click
        def show_harmonic_residual(_):
            results, _ = main()
            plot_args = get_plot_parameters(plot_magnitude_spectogram, results)
            plot_magnitude_spectogram(*tuple(plot_args))
        @view.sinusoidal_stochastic.on_click
        def show_sinusoidal_stochastic(_):
            results, _ = main()
            plot_args = get_plot_parameters(plot_stochastic_spectogram, results)
            plot_stochastic_spectogram(*tuple(plot_args))
        @view.harmonic_stochastic.on_click
        def show_harmonic_stochastic(_):
            results, _ = main()
            plot_args = get_plot_parameters(plot_stochastic_spectogram, results)
            plot_stochastic_spectogram(*tuple(plot_args))