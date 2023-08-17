import pandas as pd
import torch

from utils.common_functions import load_config


class Evaluator:
    def __init__(self, config_file):
        self.input_label = 'input'
        self.target_label = 'target'
        self.prediction_label = 'predictions'

        config = load_config(config_file)
        self.device = config['model']['device']
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.device == 'gpu' else 'cpu')
        print('Device = {}'.format(self.device))
        if self.device == torch.device('cuda'):
            torch.cuda.empty_cache()

        self._setup_output_config(config)
        self._setup_data_config(config)
        self._setup_model_config(config)

    def _setup_data_config(self, config):
        pass

    def _setup_model_config(self, config):
        pass

    def _setup_output_config(self, config):
        pass

    def _save_prediction(self, input, targets, predictions, output_path):
        prediction_df = pd.DataFrame(
            data=
            {
                self.input_label: input,
                self.target_label: targets,
                self.prediction_label: predictions
            }
        )
        prediction_df.to_csv(output_path, sep=',', index=False)
