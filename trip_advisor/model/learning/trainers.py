import os
import time

from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from learning.evaluators import Evaluator
from modules.models import TOKENIZERS_CLASSES, TokenizerType, MODELS_CLASSES, ModelType
from modules.optimizers import OPTIMIZERS_CLASSES, OptimizerType, SCHEDULERS_CLASSES, SchedulerType
from utils.common_functions import one_hot
from utils.consts import TRAIN_CONST, VAL_CONST


class Trainer(Evaluator):
    def __init__(self, config_file):
        super(Trainer, self).__init__(config_file)

    def _setup_data_config(self, config):
        self.train_df = pd.read_csv(config['dataset']['train_path'], sep=',')
        self.val_df = pd.read_csv(config['dataset']['val_path'], sep=',')
        self.input_label = config['dataset'].get('input_label', self.train_df.columns[0])
        self.target_label = config['dataset'].get('target_label', self.train_df.columns[-1])
        self.prediction_label = config['dataset'].get('target_label', 'prediction')

        self.tokenizer = TOKENIZERS_CLASSES[TokenizerType[config['tokenizer']['type']]].from_pretrained(
            config['tokenizer']['name'], do_lower_case=config['tokenizer']['do_lower_case']
        )
        encoded_train_data = self.tokenizer.batch_encode_plus(
            self.train_df[self.input_label].values,
            add_special_tokens=config['tokenizer']['add_special_tokens'],
            return_attention_mask=config['tokenizer']['return_attention_mask'],
            pad_to_max_length=config['tokenizer']['pad_to_max_length'],
            max_length=config['tokenizer']['seq_length'],
            return_tensors=config['tokenizer']['return_tensors']
        )
        train_input_ids = encoded_train_data['input_ids']
        train_attention_masks = encoded_train_data['attention_mask']
        train_labels = torch.tensor(one_hot(self.train_df[self.target_label].values))
        train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
        self.train_loader = DataLoader(train_dataset,
                                       sampler=RandomSampler(train_dataset),
                                       batch_size=config['dataset']['batch_size'])

        encoded_val_data = self.tokenizer.batch_encode_plus(
            self.val_df[self.input_label].values,
            add_special_tokens=config['tokenizer']['add_special_tokens'],
            return_attention_mask=config['tokenizer']['return_attention_mask'],
            pad_to_max_length=config['tokenizer']['pad_to_max_length'],
            max_length=config['tokenizer']['seq_length'],
            return_tensors=config['tokenizer']['return_tensors']
        )
        val_input_ids = encoded_val_data['input_ids']
        val_attention_masks = encoded_val_data['attention_mask']
        val_labels = torch.tensor(one_hot(self.val_df[self.target_label].values))
        val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
        self.val_loader = DataLoader(val_dataset,
                                     sampler=SequentialSampler(val_dataset),
                                     batch_size=config['dataset']['batch_size'])

    def _setup_model_config(self, config):
        self.model = MODELS_CLASSES[ModelType[config['model']['type']]].from_pretrained(
            config['model']['name'],
            num_labels=config['model'].get(
                'num_label',
                len(np.unique(
                    np.concatenate(
                        (self.train_df[self.target_label].values, self.val_df[self.target_label].values),
                        axis=0
                    )
                )
                )
            ),
            output_attentions=False,
            output_hidden_states=False
        )
        torch.save(self.model.state_dict(), os.path.join(self.model_path, 'init.pt'))
        self.model.to(self.device)
        self.epoch_count = config['eval']['epoch_count']
        self.optimizer = OPTIMIZERS_CLASSES[OptimizerType[config['eval']['optimizer']]](
            self.model.parameters(),
            lr=config['eval']['lr'],
            eps=config['eval']['eps']
        )
        self.scheduler = SCHEDULERS_CLASSES[SchedulerType[config['eval']['scheduler']]](
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epoch_count
        )

    def _setup_output_config(self, config):
        self.model_path = config['res']['model_path']
        os.makedirs(self.model_path, exist_ok=True)
        self.tb_path = config['res']['tb_path']
        os.makedirs(self.tb_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tb_path)
        self.prediction_path = config['res']['prediction_path']
        os.makedirs(self.prediction_path, exist_ok=True)
        self.all_train_outputs = []
        self.all_val_outputs = []
        self.mean_epoch_loss = None

    def epoch(self, epoch_num, mode):
        batch_num = 0
        epoch_losses = list()
        loader = None
        all_outputs = list()
        if mode == TRAIN_CONST:
            self.model.train(True)
            loader = self.train_loader
        elif mode == VAL_CONST:
            self.model.eval()
            loader = self.val_loader
        epoch_start = time.perf_counter()
        for batch in tqdm(loader, desc='Epoch {}, {}'.format(epoch_num + 1, mode)):
            self.model.zero_grad()
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(torch.uint8).to(self.device)
            labels = labels.to(torch.float32).to(self.device)
            if mode == TRAIN_CONST:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else:
                with torch.no_grad:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            epoch_loss = outputs.loss
            epoch_losses.append(epoch_loss.item())
            preds = torch.argmax(outputs.logits, dim=1)
            all_outputs.append(preds.detach().cpu().numpy())
            epoch_loss.backward() if mode == TRAIN_CONST else 0
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) if mode == TRAIN_CONST else 0
            self.optimizer.step() if mode == TRAIN_CONST else 0
            self.scheduler.step() if mode == TRAIN_CONST else 0
            batch_num += 1
        epoch_end = time.perf_counter()
        all_outputs = np.concatenate(all_outputs, axis=0)

        self.mean_epoch_loss = np.mean(epoch_losses)
        self.writer.add_scalar('Loss/{}'.format(mode), self.mean_epoch_loss, epoch_num)

        print('Mean loss = {0}, time: {1:.7f}'.format(
            self.mean_epoch_loss,
            epoch_end - epoch_start)
        )
        return all_outputs

    def train(self):
        for epoch_num in range(self.epoch_count):
            self.all_train_outputs = self.epoch(epoch_num, mode=TRAIN_CONST)
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.model_path,
                    'epoch={0}_loss={1:.7f}.pt'.format(epoch_num, self.mean_epoch_loss)
                )
            )
            self.all_val_outputs = self.epoch(epoch_num, mode=VAL_CONST)
        self._save_prediction(
            self.train_df[self.input_label].values,
            self.train_df[self.target_label].values,
            self.all_train_outputs,
            os.path.join(self.prediction_path, 'train.csv')
        )
        self._save_prediction(
            self.val_df[self.input_label].values,
            self.val_df[self.target_label].values,
            self.all_val_outputs,
            os.path.join(self.prediction_path, 'val.csv')
        )
