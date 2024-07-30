# Distributed Training

## I. Background

To optimize the training of large models, we can:

  * reduce trainable parameter number

    - finetune: prompt-tuning, PTuning, Prefix-Tuning, LoRA, IA3
    - Use PEFT module

  * use different precision:
    
    - half_precision, INT8, NF4
    - Use Bitsandbytes module

There is another way to optimize the training: use all available resources such as several devices, which is called distributed training.


There are 3 types of parallel computation:

  * Data Prallel (DP): one copy of model per GPU,  trained on different dataset. The problem with this approach is that if the model is too large, it can't be done.

  * Pipeline Paralle (PP): the model is decomposed to be put on different GPU.

  * Tensor Parallel (TP): the weights were decomposed to be put on different GPU.

So which to choose:

  * if the model can fit in the GPUs, one can use DP
  * if not, one can use PP or TP
  * there is another approach is to use the mix of the 3, called 3D-Parallel

## II. Realization

The purpose of this section is to show how certain distributed concepts were realized. 
In practice using transformers, we don't need most of those codes since transformers' trainer provides most of those functionalities.
However, understanding of some of those concepts can help code effectively and debug easily.

The examples are:

  * DP: hf_transformers_distributed_dp.ipynb
    Example to show how DP works using pytorch and transformers. The code can be run directly in the jupyter notebook.

  * DDP: hf_transformers_distributed_ddp.ipynb
    Example to show how DP works using pytorch and transformers. This notebook serves as a note to explain how to generate the code and a base to provide code to implement DDP.
    The codes to use DDP are:
      - ddp_train_torch.py
      - ddp_train_transformers.py
    Use elastic launch to run the codes. For details, see the notebook.

  * accelerate: hf_transformers_distributed_accelerate.ipynb
    To illustrate how accelerate API, which is a framework to help use distributed methods. It contains a successive modifications from a base training code. So one has to follow the order of the notebook to be able to understand. The final code files includes all modification in the notebook, which explains each change and marks them with a identification.
    The files used in this notebook are:
      - accelerate_torch.py
      - accelerate_torch_mixed.py
      - default_config.yaml
      - zero2_config.json
      - zero3_config.json
    However, we also used the file "ddp_train_transformers.py" to illustrate some tools since the trainer of transformers need no change to adapt to some HF tools.

  