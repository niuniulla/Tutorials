# my P2P calble is not connected, so I have to disable this option when running multiple GPUs
import os
os.environ["NCCL_P2P_DISABLE"] = "1"

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import evaluate

ckp = "google-bert/bert-base-uncased"

# load data
data = load_dataset("davidberg/sentiment-reviews")


################
# A. fix split #
################

split_data = data["train"].train_test_split(test_size=0.2, seed=42)

#############################

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(ckp)

label2id = {"positive": 0, "negative": 1}

# process data

def process(samples):

    _data = {"review":[], "division":[]}
    for i in range(len(samples["review"])):
        if samples["division"][i] in label2id.keys():
            _data["review"].append(samples["review"][i])
            _data["division"].append(samples["division"][i])

    toks = tokenizer(_data["review"], max_length=128, truncation=True, padding="max_length", return_tensors="pt")

    toks["labels"] = [label2id.get(d) for d in _data["division"]]

    return toks

tokenized_data = split_data.map(process, batched=True, remove_columns=split_data["train"].column_names)

# load model

model = AutoModelForSequenceClassification.from_pretrained(ckp)

# metric
acc_fct = evaluate.load("accuracy")
f1_fct = evaluate.load("f1")

def metric(pred):

    preds, refs = pred

    preds = preds.argmax(axis=-1)

    acc = acc_fct.compute(predictions=preds, references=refs)
    f1 = f1_fct.compute(predictions=preds, references=refs)

    acc.update(f1)

    return acc


# %%
# training
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    resume_from_checkpoint="./ckpts",
    bf16=True
)

trainer = Trainer(
    model=model, 
    args=args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True),
    compute_metrics=metric
)


# %%
trainer.train()


