import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

from learning.evaluators import Evaluator
from modules.models import TOKENIZERS_CLASSES, TokenizerType, ModelType, MODELS_CLASSES


class Predictor(Evaluator):
    def __init__(self, config_file):
        super(Predictor, self).__init__(config_file)

    def _setup_data_config(self, config):
        self.test_df = pd.read_csv(config['dataset']['test_path'], sep=',')
        self.input_label = config['dataset'].get('input_label', self.test_df.columns[0])
        self.target_label = config['dataset'].get('target_label', self.test_df.columns[-1])
        self.prediction_label = config['dataset'].get('target_label', 'prediction')

        self.tokenizer = TOKENIZERS_CLASSES[TokenizerType[config['tokenizer']['type']]].from_pretrained(
            config['tokenizer']['name'], do_lower_case=config['tokenizer']['do_lower_case']
        )
        encoded_test_data = self.tokenizer.batch_encode_plus(
            self.test_df[self.input_label].values,
            add_special_tokens=config['tokenizer']['add_special_tokens'],
            return_attention_mask=config['tokenizer']['return_attention_mask'],
            pad_to_max_length=config['tokenizer']['pad_to_max_length'],
            max_length=config['tokenizer']['seq_length'],
            return_tensors=config['tokenizer']['return_tensors']
        )
        self.test_input_ids = encoded_test_data['input_ids']
        self.test_attention_masks = encoded_test_data['attention_mask']

    def _setup_model_config(self, config):
        self.model = MODELS_CLASSES[ModelType[config['model']['type']]].from_pretrained(
            config['model']['name'],
            num_labels=config['model'].get(
                'num_label',
                len(np.unique(self.test_df[self.target_label].values))
            ),
            output_attentions=False,
            output_hidden_states=False
        )
        self.model.to(self.device)
        self.model.load_state_dict(
            torch.load(
                config['model']['path'], map_location=torch.device(self.device)
            )
        )

    def _setup_output_config(self, config):
        self.prediction_path = config['res']['prediction_path']
        os.makedirs(self.prediction_path, exist_ok=True)
        self.metrics_path = config['res']['metrics_path']
        os.makedirs(self.metrics_path, exist_ok=True)
        self.all_test_outputs = list()

    def predict(self):
        predict_start = time.perf_counter()
        for input_ids, attention_mask in tqdm(
                zip(self.test_input_ids, self.test_attention_masks), desc='Prediction'
        ):
            input_ids, attention_mask = input_ids.unsqueeze(0).to(self.device), attention_mask.unsqueeze(0).to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            self.all_test_outputs.append(preds.detach().cpu().numpy())
        predict_end = time.perf_counter()
        self.all_test_outputs = np.concatenate(self.all_test_outputs, axis=0)

        accuracy = accuracy_score(self.test_df[self.target_label].values, self.all_test_outputs)

        conf_matrix = confusion_matrix(
            self.test_df[self.target_label].values,
            self.all_test_outputs,
            labels=np.sort(np.unique(self.test_df[self.target_label].values)))
        conf_matrix_display = ConfusionMatrixDisplay(
            conf_matrix,
            display_labels=np.sort(np.unique(self.test_df[self.target_label].values))
        )
        conf_matrix_display.plot()
        conf_matrix_display.figure_.savefig(
            os.path.join(self.metrics_path, 'confusion_matrix.png'), dpi=conf_matrix_display.figure_.dpi
        )

        report = classification_report(
            self.test_df[self.target_label].values,
            self.all_test_outputs, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(self.metrics_path, 'classification_report.csv'), sep=',', index=False)

        print('Accuracy: {0:.7f}, time: {1:.7f}, mean time: {2:.7f}'.format(
            accuracy,
            predict_end - predict_start,
            predict_end - predict_start / len(self.test_input_ids))
        )

        self._save_prediction(
            self.test_df[self.input_label].values,
            self.test_df[self.target_label].values,
            self.all_test_outputs,
            os.path.join(self.prediction_path, 'test.csv')
        )
