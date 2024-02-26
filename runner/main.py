#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys

import numpy as np

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import libs.configs.default as cfg
from libs.data import create_dataset, DataCollator
from libs.data.data_args import DataTrainingArguments
from libs.data.image_processor import DonutImageProcessor
from libs.model.model_args import ModelArguments
from libs.trainers import PreTrainer as Trainer
from libs.model.vision_encoder_decoder.modeling_vision_encoder_decoder import myVisionEncoderDecoderModel

import transformers
from transformers import (
    AutoTokenizer, 
    DonutProcessor,
    VisionEncoderDecoderModel,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)


def build_model(
        model_name_or_path,
        encoder_model_path, 
        decoder_model_path, 
        processor, 
        finetune_flag, 
    ):
    if finetune_flag:
        model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)
    else:
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_model_path, decoder_model_path,
            # ignore_mismatched_sizes=True
        )
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    # special token id for model decoder
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    return model


def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],

    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"dataset_name: {data_args.dataset_name}")
    logger.info(f"decoder_task: {data_args.decoder_task}")
    logger.info(f"use_plugin: {data_args.use_plugin}")
    logger.info(f"use_lora: {data_args.use_lora}")
    logger.info(f"use_type: {data_args.use_type}")
    logger.info(f"encoder_path: {model_args.encoder_path}")
    logger.info(f"decoder_path: {model_args.decoder_path}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    finetune_flag = model_args.finetune
    model_name_or_path = model_args.model_name_or_path
    decoder_model_path = model_args.decoder_path
    encoder_model_path = model_args.encoder_path

    # config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(decoder_model_path)
    img_processor = DonutImageProcessor.from_pretrained(encoder_model_path)
    processor = DonutProcessor(image_processor=img_processor, tokenizer=tokenizer)

    # model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    #     swin_model_path, bert_model_path,
    #     # ignore_mismatched_sizes=True
    # )
    model = build_model(
        model_name_or_path,
        encoder_model_path,
        decoder_model_path,
        processor,
        finetune_flag
    )

    # Initialize pretrain dataset and valid dataset
    train_annot_path = cfg.datasets[data_args.dataset_name]['train_path']
    valid_annot_path = cfg.datasets[data_args.dataset_name]['valid_path']
    train_dataset = create_dataset(
        dataset_name=data_args.dataset_name,
        json_path=train_annot_path,
        processor=processor,
        max_length=512,
        task=data_args.decoder_task, 
        use_plugin=data_args.use_plugin
    )
    valid_dataset = create_dataset(
        dataset_name=data_args.dataset_name,
        json_path=valid_annot_path,
        processor=processor,
        max_length=512, 
        task=data_args.decoder_task, 
        use_plugin=data_args.use_plugin
    )
    data_collator = DataCollator(processor.tokenizer)

    # Metrics
    def compute_metrics(p):
        predictions, labels = p
        mask = labels > -1
        pred = np.logical_and(predictions==labels, mask)
        correct_nums = pred.sum()
        total_nums = max(mask.sum(), 1e-6)
        acc = correct_nums / total_nums
        results = {
            "precision": acc,
        }
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("************************\n")
        return results

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=valid_dataset if training_args.do_eval else None,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
