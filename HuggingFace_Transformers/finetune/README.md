# Fintuning

The idea behind finetuning is to train a small amount of parameters compared to the base model, which can minimize the memory usage and storage.

A summary of all types of tuning techniques.
https://arxiv.org/pdf/2303.15647


the files: 

 * hf_transformers_finetune_bitfit.ipynb
 * hf_transformers_finetune_prompt.ipynb
 * hf_transformers_finetune_ptuning.ipynb
 * hf_transformers_finetune_prefix.ipynb




Here are the comparison of trainable parameters of the above mentioned fine-tuning methods.
We listed only the methods with default parameters. The parameters will vary if we choose different configurations.

| Fine-Tuning 	            | trainable params  | %      |
|---	                    |---	            |---	 |
| BitFit                    | 408576            | 0.0384 |
| Soft Prompt (10 tokens)   | 15360             | 0.0014 |
| Hard promt  (7 tokens)    | 10752  	        | 0.0010 |
| P-Tuning                  | 7097856           | 0.6619 |
| Prefix-Tuning (-Project)  | 737280 	        | 0.0692 |
| Prefix-Tuning (+Project)  | 115696128         | 9.7964 |
| lora                      | 1179648           | 0.1106 |
| IA3                       | 258048            | 0.0242 |
