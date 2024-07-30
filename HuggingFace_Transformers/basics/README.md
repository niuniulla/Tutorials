# Basics of Transformers

This introduce the basic components used in transformers to train and use a model.


There are 6 notebooks:

 * hf_transformers_basics_0_intro: Introduce the pipeline and the steps behind the pipeline.
 * hf_transformers_basics_1_model: Present the model tools of transformers, how to load and train a model.
 * hf_transformers_basics_2_datasets: Present the datasets module, how to load and process the hf data.
 * hf_transformers_basics_3_tokenizer: present the tokenizer tools, how to load and tokenize data.
 * hf_transformers_basics_4_evaluate: present the evaluate module, how to load metrics and use them in the training.
 * hf_transformers_basics_5_trainer: show how to use trainer to manage training.
 * hf_transformers_basics_6_nlp: A summary of the above materials, and present some technique to optimize the training resources.

 


How this work:

In the model notebook, we used a classification example to show all the steps to train a model. The steps using pytorch logic to train are:

 * step1. load dataset
 * step2. split data
 * step3. tokenizer
 * step4. dataloader
 * step5. load model
 * step6. define optimizer
 * step7. evaluation
 * step8. Define Training
 * step9. Train

And we gradually replace the steps by the transformers modules and APIs to simplify this training processus. So it is preferably to read the notebooks in order to be able to follow and understand the changes.