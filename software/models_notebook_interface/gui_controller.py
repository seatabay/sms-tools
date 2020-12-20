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
        
    
        @view.input_sound.on_click
        def plot_input(_):
            results,filename = main()
            plot_args = inspect.getargspec(Models.plot_sound)[0]
            
            plot_result = []
            for arg in plot_args:
                plot_result.append(results[arg])
                
            Models.plot_sound(*tuple(plot_result))
            
        @view.mag_spectrum.on_click
        def show_mag_spectrum(_):
            results, filename = main()
            plot_args = inspect.getargspec(Models.plot_magnitude_phase_spectrum)[0]
            
            plot_result = []
            for arg in plot_args:
                plot_result.append(results[arg])
                
            Models.plot_magnitude_phase_spectrum(*tuple(plot_result))
