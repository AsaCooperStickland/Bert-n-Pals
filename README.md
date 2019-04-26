# PyTorch implementation of BERT and PALs

## Introduction
Work by Asa Cooper Stickland and Iain Murray, University of Edinburgh.
Code for [BERT and PALs](https://arxiv.org/abs/1902.02671); most of this code is from https://github.com/huggingface/pytorch-pretrained-BERT (who are not affilied with the authors) and we reuse some of their documentation. 
The only files we modified/created for multi-task learning were `modeling.py` which contains the BERT model formulation and `run_multi_task.py` which performs multi-task training on the GLUE benchmark.

For our documentation see the 'Multi-task learning with PALs and alternatives' section below!

## PyTorch models for BERT (old documentation BEGINS)

We included three PyTorch models in this repository that you will find in [`modeling.py`](modeling.py):

- `BertModel` - the basic BERT Transformer model
- `BertForSequenceClassification` - the BERT model with a sequence classification head on top
- `BertForQuestionAnswering` - the BERT model with a token classification head on top

Here are some details on each class.

### 1. `BertModel`

`BertModel` is the basic BERT Transformer model with a layer of summed token, position and sequence embeddings followed by a series of identical self-attention blocks (12 for BERT-base, 24 for BERT-large).

The inputs and output are **identical to the TensorFlow model inputs and outputs**.

We detail them here. This model takes as inputs:

- `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the word token indices in the vocabulary (see the tokens preprocessing logic in the scripts `extract_features.py`, `run_classifier.py` and `run_squad.py`), and
- `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
- `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max input sequence length in the current batch. It's the mask that we typically use for attention when a batch has varying length sentences.

This model outputs a tuple composed of:

- `all_encoder_layers`: a list of torch.FloatTensor of size [batch_size, sequence_length, hidden_size] which is a list of the full sequences of hidden-states at the end of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), and
- `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a classifier pretrained on top of the hidden state associated to the first character of the input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

An example on how to use this class is given in the `extract_features.py` script which can be used to extract the hidden states of the model for a given input.

### 2. `BertForSequenceClassification`

`BertForSequenceClassification` is a fine-tuning model that includes `BertModel` and a sequence-level (sequence or pair of sequences) classifier on top of the `BertModel`.

The sequence-level classifier is a linear layer that takes as input the last hidden state of the first character in the input sequence (see Figures 3a and 3b in the BERT paper).

### 3. `BertForQuestionAnswering`

`BertForQuestionAnswering` is a fine-tuning model that includes `BertModel` with a token-level classifiers on top of the full sequence of last hidden states.

The token-level classifier takes as input the full sequence of the last hidden state and compute several (e.g. two) scores for each tokens that can for example respectively be the score that a given token is a `start_span` and a `end_span` token (see Figures 3c and 3d in the BERT paper).

## Requirements

This code was tested on Python 3.5+. The requirements are:

- PyTorch (>= 0.4.1)
- tqdm
- scikit-learn (0.20.0)
- numpy (1.15.4)

## Training on large batches: gradient accumulation, multi-GPU and distributed training

BERT-base and BERT-large are respectively 110M and 340M parameters models and it can be difficult to fine-tune them on a single GPU with the recommended batch size for good performance (in most case a batch size of 32).

To help with fine-tuning these models, we have included three techniques that you can activate in the fine-tuning scripts `run_classifier.py` and `run_squad.py`: gradient-accumulation, multi-gpu and distributed training. For more details on how to use these techniques you can read [the tips on training large batches in PyTorch](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255) that I published earlier this month.

Here is how to use these techniques in our scripts:

- **Gradient Accumulation**: Gradient accumulation can be used by supplying a integer greater than 1 to the `--gradient_accumulation_steps` argument. The batch at each step will be divided by this integer and gradient will be accumulated over `gradient_accumulation_steps` steps.
- **Multi-GPU**: Multi-GPU is automatically activated when several GPUs are detected and the batches are splitted over the GPUs.
- **Distributed training**: Distributed training can be activated by suppying an integer greater or equal to 0 to the `--local_rank` argument. To use Distributed training, you will need to run one training script on each of your machines. This can be done for example by running the following command on each server (see the above blog post for more details):

```bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=$THIS_MACHINE_INDEX --master_addr="192.168.1.1" --master_port=1234 run_classifier.py (--arg1 --arg2 --arg3 and all other arguments of the run_classifier script)
```

Where `$THIS_MACHINE_INDEX` is an sequential index assigned to each of your machine (0, 1, 2...) and the machine with rank 0 has an IP adress `192.168.1.1` and an open port `1234`.


## Multi-task learning with PALs and alternatives (old documentation ENDS)

We provide some basic details of the parts of the code used for multi-task learning:

`BertPals` and `BertLowRank`: These classes contains two linear layers which project down to the smaller hidden size (called `hidden_size_aug` in the code), and, for PALs, a multi-head attention mechanism without the final projection matrix inbetween.

`BertLayer`: In the original code this class contains an entire BERT layer, and we modify it to include an optional BERTMulti layer or an LHUC transformation.

`BertEncoder`: In the original code this implemented a module that applied a series of BERT layers to the input. We modify this class, to optionally tie together all the encoder and decoder matrices, and either set each layer to 'multi-task mode', or add attention modules to add to the top of the model. 

We implement our multi-task sampling methods (annealed, proportional etc.) with `np.random.choice`. 

The [GLUE data](https://gluebenchmark.com/tasks) can be downloaded with
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e). This README assumes it is located in `glue/glue_data`.

## Getting the pretrained weights

You can convert any TensorFlow checkpoint for BERT (in particular [the pre-trained models released by Google](https://github.com/google-research/bert#pre-trained-models)) in a PyTorch save file by using the [`./pytorch_pretrained_bert/convert_tf_checkpoint_to_pytorch.py`](convert_tf_checkpoint_to_pytorch.py) script.

This CLI takes as input a TensorFlow checkpoint (three files starting with `bert_model.ckpt`) and the associated configuration file (`bert_config.json`), and creates a PyTorch model for this configuration, loads the weights from the TensorFlow checkpoint in the PyTorch model and saves the resulting model in a standard PyTorch save file that can be imported using `torch.load()`

You only need to run this conversion script **once** to get a PyTorch model. You can then disregard the TensorFlow checkpoint (the three files starting with `bert_model.ckpt`) but be sure to keep the configuration file (`bert_config.json`) and the vocabulary file (`vocab.txt`) as these are needed for the PyTorch model too.

To run this specific conversion script you will need to have TensorFlow and PyTorch installed (`pip install tensorflow`). The rest of the repository only requires PyTorch.

Here is an example of the conversion process for a pre-trained `BERT-Base Uncased` model:

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch \
  $BERT_BASE_DIR/bert_model.ckpt \
  $BERT_BASE_DIR/bert_config.json \
  $BERT_BASE_DIR/pytorch_model.bin
```

You can download Google's pre-trained models for the conversion [here](https://github.com/google-research/bert#pre-trained-models).
We use the BERT-base uncased: `uncased_L-12_H-768_A-12` model for all experiments. 

## BERT and PALs

The bert config files (example: `uncased_L-12_H-768_A-12\pals\_config.json`) contain the settings neccesary to reproduce the important results of our work. 

`pals_config.json`: Contains the configuration for PALs with small hidden size 204.

`low_rank_config.json`: Contains the configuration for low-rank layers with small hidden size 100.

`top_attn_config.json` and `top_bert_layer_config.json` Contain the configuration for adding projected attention layers with hidden size 204 or an entire bert layer to the top of the base model.

`houlsby_config.json`: Contains configuration for approximately recreating the setup of a [concurrent paper](https://arxiv.org/abs/1902.00751) by Houlsby et. al that adds adapters to both layernorms in each BERT layer.

`houlsby_plus_plas_config.json`: Same as the previous setting but replace one of the low rank adapters from the previous setup with a PAL adapter. NOT TESTED THOUROUGHLY.

Choose the `sample` argument to be 'anneal', 'sqrt', 'prop' or 'rr' for the various sampling methods listed in the paper. Choose 'anneal' to reproduce the best results. 

Here's an example of how to run the PALs method with annealed sampling (with all settings the same as in the paper.):

```shell
export BERT_BASE_DIR=/path/to/uncased_L-12_H-768_A-12
export BERT_PYTORCH_DIR=/path/to/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue/glue_data
export SAVE_DIR=/tmp/saved

python run_multi_task.py \
  --seed 42 \
  --output_dir $SAVE_DIR/pals \
  --tasks all \
  --sample 'anneal'\
  --multi \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/ \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/pals_config.json \
  --init_checkpoint $BERT_PYTORCH_DIR/pytorch_model.bin \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 25.0 \
  --gradient_accumulation_steps 1

```
