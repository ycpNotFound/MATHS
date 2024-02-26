# MATHS
Code for paper: MATHS: Multimodal Transformer-based Human-readable Solver

Multimodal mathematical reasoning has gained increasing attention in recent times. However, previous effective methodshave not tried to reason in the form of natural language. In thispaper, we introduce a model named MATHS (MultimodAl Transformer-based Human-readable Solver) for visual arithmetic and geometry problems in multimodal mathematical reasoning tasks. Drawing inspiration from Multimodal Large Language Models (MLLMs), our approach involves generating problem-solving processes expressed in natural language, in order to leverage the inherent reasoning capabilities embedded within language models. To address the challenge of precise calculations for language models, our work proposes a Math-Constrained Generation (MCG) method to impose hard constraints on generated outputs. Extensive experiments demonstrate our model excels in visual arithmetic task, and achieves results that are either better or comparable to existing methods in geometry problems.

## Dataset

Create Machine Number Sense dataset and annotation files.

```bash
python libs/data/generator/main.py
```

## Train

```bash
bash sh_files/number-sense/finetune.sh
```

## Inference

```bash
bash sh_files/number-sense/inference.sh
```

## Tokenizer

If you want to train your model on UniGeo dataset, you may need add special tokens to your tokenizer. Code from `scripts/add_special_tokens.py` can help you to do this.
