import os

from collections import defaultdict
import pathlib
import json

class GuiModels:
    
    DFT = dict.fromkeys(["fs", "x1", "mX", "pX", "y"])
    STFT = dict.fromkeys(["x1","fs", "mX", "pX", "y"])
    HARMONIC = dict.fromkeys(["fs", "x", "hfreq", "y"])
    HPR = dict.fromkeys(["fs", "x", "mXr", "hfreq"])
    HPS = dict.fromkeys(["fs", "x", "y", "stocenv"])
    SINE = dict.fromkeys(["fs", "x", "tfreq", "y"])
    SPR = dict.fromkeys(["fs", "x", "tfreq", "mXr", "pXr", "y", "ys"])
    STOCHASTIC = dict.fromkeys(["fs", "x", "stocenv", "y"])
    
    DIR_NAME = "Results"
    KEEP_TRACK = 3
    
    
    def dump_json(self, data, filename):
        with open(filename, "w") as fp:
            json.dump(data, fp, indent=4)

    
    def create_json(self):
        """
        Create JSON file to store the results.

        Parameters
        ----------
        json_filepath : str
                Path to the JSON file

        Returns
        -------
                dict
        """
        json_filepath = "{}/results.json".format(GuiModels.DIR_NAME)
        self.dump_json(defaultdict(dict), json_filepath)
        json = self.read_json(json_filepath)
        return json

    def read_json(self, filename):
        """
        Read JSON file from a given filepath.
        
        Parameters
        ----------
        
        filename  : str
                Path to the file
                
        Returns
        -------
        
                dict
        """
        
        if not pathlib.Path(filename).exists():
            self.create_json()
        
        f = open(filename, "r")
        data = json.loads(f.read())
        return data

    def limit_computation_number(self, json_file, data, key):
        json_key = [key for key in json_file.keys()]
        if len(json_file.keys()) < GuiModels.KEEP_TRACK:
            json_file[key] = data
        else:
            del json_file[json_key[0]]
            json_file[key] = data
        return json_file

    def update_json(self, data, key):
        """
        Update JSON file for a given data.
        
        Parameters
        ----------
        data      : dict
                Dictionary containing computation results
        key       : str
                Key name containing model name and parameters
        
        Returns
        -------
                None
        """
        
        with open(
            f"{GuiModels.DIR_NAME}/results.json".format(GuiModels.DIR_NAME), "r+"
        ) as f:
            json_data = json.load(f)
            json_data = self.limit_computation_number(json_data, data, key)
            f.seek(0)
            f.write(json.dumps(json_data))
            f.truncate()
            
    def create_model_param_dict(self, model_widget):
        """
        Create a dictionary containing model and corresponding parameter values.
        
        Parameters
        ----------
        model_widget   : ipywidgets.widgets.widget_box.VBox
                    Ipywidget for selected model
        model_name     : str
                    Name of the selected model
                    
        Returns
        -------
                        : dict
        """
        model_param_dict = {}
        for item in model_widget.children:
            if hasattr(item, 'accept') and hasattr(item, 'value'):
                model_param_dict[item.description_tooltip] = self.get_selected_audio_filename(item)
            elif hasattr(item, 'value') and not hasattr(item, 'accept'):
                if isinstance(item.value, str):
                    model_param_dict[item.description_tooltip] = item.value.lower()
                else:
                    model_param_dict[item.description_tooltip] = item.value
        return model_param_dict
                
    def compute_model_results(
        self, model_name, model_widget
    ):

        """
        Compute results using a selected model.

        Parameters
        ----------


        Returns
        -------


        """
        
        result_dict = {}
        
        #Get a model dict to save the parameter values
        model_dict = getattr(GuiModels, model_name.upper())
        
        #Get the model function from Models class
        model_func = getattr(Models, model_name)
        
        #Create the model dict with selected parameter values
        model_param_dict = self.create_model_param_dict(model_widget)
        filename = self.create_filename(model_param_dict, model_name)
        
        #Read json file
        stored_results = self.read_json("Results/results.json")
        
        # If model results are already computed, return it
        if len(stored_results.keys()) != 0 and filename in stored_results.keys():
            result_dict = stored_results[filename]
            del stored_results
            
        # Otherwise compute the results
        else:
            print(model_param_dict)
            results = model_func(*tuple(model_param_dict.values()))
            for result, key in zip(results, model_dict.keys()):
                if isinstance(result, numpy.ndarray):
                    result_dict[key] = result.tolist()
                else:
                    result_dict[key] = result
                    
            result_dict.update(model_param_dict)
            result_dict["filename"] = "/".join(filename.split("_"))

            # Keep the number of computations stable
            self.update_json(result_dict, filename)
        return result_dict, filename


    def get_selected_audio_filename(self, audio_widget):
        """
        Get the name of the selected audio.
        
        Parameters
        ----------
        audio_widget    : ipywidgets.widgets.widget_upload.FileUpload
                    Ipywidget for selecting audio to be uploaded
                    
        Returns
        -------
                            str
        """
        audio_filename = [audio for audio in audio_widget.value.keys()]
        if audio_filename == []:
            audio_filename.append("bendir.wav")
        return audio_filename[0]

    def create_dir(self):
        """
        Create directory to store results.
        """
        if not pathlib.Path(GuiModels.DIR_NAME).exists():
            os.mkdir(GuiModels.DIR_NAME)

        
    def create_filename(self,model_param_dict, model_name):
        """
        Create filename from model name and corresponding parameter values.
        
        Parameters
        ----------
        model_param_dict : dict
                    Dictionary containing model parameters
                    
        Returns
        -------
                    str
        """
        word_number = len(model_param_dict.keys()) +1
        brackets = "{}_" * word_number
        filename = brackets.format(model_name,*tuple(model_param_dict.values()))
        
        return filename
        
