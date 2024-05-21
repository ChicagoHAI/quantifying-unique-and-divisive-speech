
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
import argparse
import os
from pathlib import Path
from typing import Any, Dict
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, f1_score, average_precision_score, classification_report
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
import json
import torch
from torch import nn
import pytorch_lightning as pl

from dataloader import TransformerLMDataset

from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}


# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


class Transformer_PL(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        num_labels=None,
        config=None,
        tokenizer=None,
        model=None,
        **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        # TODO: move to self.save_hyperparameters()
        # self.save_hyperparameters()
        # can also expand arguments into trainer signature for easier reading
        mode = "language-modeling"

        self.save_hyperparameters(hparams)
        logger.info(f"Number of Labels: {self.hparams.num_labels}")
        self.step_count = 0
        self.output_dir = Path(self.hparams.output_dir)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
                force_download=True,
                local_files_only=False,
                # **config_kwargs,
            )
            print(self.config)
        else:
            self.config: PretrainedConfig = config

        extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
        for p in extra_model_params:
            if getattr(self.hparams, p, None):
                assert hasattr(self.config, p), f"model config doesn't have a `{p}` attribute"
                setattr(self.config, p, getattr(self.hparams, p))

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
            )
            special_tokens_dict = {'pad_token': '<PAD>', 
            'additional_special_tokens':['<DEBATE_START>', '<DEBATE_END>', '<ENT>']}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        else:
            self.tokenizer: PreTrainedTokenizer = tokenizer
        self.model_type = MODEL_MODES[mode]
        if model is None:
            self.model = self.model_type.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                cache_dir=cache_dir, 
            )
        else:
            self.model = model
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)


    def load_hf_checkpoint(self, *args, **kwargs):
        self.model = self.model_type.from_pretrained(*args, **kwargs)

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
            )
        self.opt = optimizer

        self.train_loader = getattr(self,"train_loader",None)
        if self.train_loader:
            scheduler = self.get_lr_scheduler()
        else:
            return [optimizer]
        return [optimizer], [scheduler]

    def forward(self, **inputs):
        labels = inputs.pop('labels')
        input_ids = inputs.pop('input_ids')
        attention_mask = inputs.pop('attention_mask')

        note_embeds = self.model.transformer.wte(input_ids[...,:-1].contiguous()) #return [B, L-1, D]
        if attention_mask is not None:
            attention_mask = attention_mask[...,:-1].contiguous()
        
        outputs = self.model(inputs_embeds=note_embeds, attention_mask=attention_mask) #[B, L-1, D]
        logits = outputs.logits
        target_input_ids = input_ids[:,1:].contiguous() #return [B, L-1]
        # loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), target_input_ids.view(-1))
        return loss, logits


    def training_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None

        outputs = self(**inputs)
        loss= outputs[0]

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log( "rate", lr_scheduler.get_last_lr()[-1])

        return {"loss": loss}
    
    def training_epoch_end(self, outputs):
        mean_loss = torch.mean(torch.tensor([x["loss"] 
                                             for x in outputs 
                                             if not torch.isnan(x["loss"])]))

        self.log("train_loss", mean_loss)

        loss_log_path = self.output_dir.joinpath("logs/loss_log.txt")
        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        rate =  lr_scheduler.get_last_lr()[-1]
        with open(loss_log_path, "a") as f:
            f.write(f"train_epoch, {mean_loss}, {rate}\n")
        return #self.validation_end(outputs)

    def validation_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "structured": batch[5], "doctor_ids": batch[6]}
        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]


        return {"loss":tmp_eval_loss.item(), "labels":batch[3], "stays":batch[4]}

    def validation_epoch_end(self, outputs):
        mean_loss = np.mean([x["loss"] for x in outputs if not np.isnan(x["loss"])])
        print(mean_loss)
        self.log("val_loss", mean_loss)

        loss_log_path = self.output_dir.joinpath("logs/loss_log.txt")
        with open(loss_log_path, "a") as f:
            f.write(f"val_epoch, {mean_loss}\n")
        return #self.validation_end(outputs)

    def test_step(self, batch, batch_nb):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "structured": batch[5], "doctor_ids": batch[6]}

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        self.log('test_loss', tmp_eval_loss.item())
        return {"loss":tmp_eval_loss.item(), "labels":batch[3], "stays":batch[4]}

    def test_epoch_end(self, outputs):
        # print(outputs)
        # roc_auc, pr_auc, f1  = self.vote_score(outputs, mode="test")
        # self.log("test_ROCAUC", roc_auc)
        # self.log("test_PRAUC", pr_auc)
        # self.log("test_F1", f1)
        mean_loss = np.mean([x["loss"] for x in outputs if not np.isnan(x["loss"])])
        self.log("test_loss", mean_loss)

        loss_log_path = self.output_dir.joinpath("logs/loss_log.txt")
        with open(loss_log_path, "a") as f:
            f.write(f"test_epoch, {mean_loss}")
        return #self.validation_end(outputs)
    
    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        if self.hparams.max_epochs is not None:
            num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
            effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
            dataset_size = len(self.train_loader.dataset)
            return (dataset_size / effective_batch_size) * self.hparams.max_epochs
        else:
            return self.hparams.max_steps / effective_batch_size

    def setup(self, stage):
        if stage == "fit":
            self.train_loader = self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)

    def get_dataloader(self, type_path, batch_size, shuffle=False):
        # todo add dataset path
        type_path = "validation" if type_path == "valid" else type_path
        filename = f"{self.hparams.data_dir}/{type_path}.csv"

        data = pd.read_csv(filename)
        dataset = TransformerLMDataset(
            self.tokenizer, data, self.hparams.max_seq_length)
        logger.info(f"Loading {type_path} dataset with length {len(dataset)} from {filename}")
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=self.hparams.num_workers,
                                                collate_fn=dataset.collate_fn)
        
        return data_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.get_dataloader("valid", self.hparams.eval_batch_size, shuffle=True)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)


    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("best_tfmr")
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument("--learning_rate", default=5e-6, type=float, help="The initial learning rate for Adam.")
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar,
            type=str,
            help="Learning rate scheduler",
        )
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--num_workers", default=16, type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=3, type=int)
        parser.add_argument("--train_batch_size", default=4, type=int)
        parser.add_argument("--eval_batch_size", default=4, type=int)
        parser.add_argument("--adafactor", action="store_true")
        parser.add_argument("--structured", action="store_true")
        parser.add_argument("--doctor", action="store_true")

        return parser
