
# %%
import evaluate, os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# my P2P cable is not connected, so I have to disable this option when running multiple GPUs
os.environ["NCCL_P2P_DISABLE"] = "1"

###########
# A. init #
###########

# if train on linux, setup nccl

import torch.distributed as dist

dist.init_process_group(backend="nccl")

#######################################

ckp = "google-bert/bert-base-uncased"

# load data
data = load_dataset("davidberg/sentiment-reviews")

# dataset
from torch.utils.data import Dataset

class dataset(Dataset):

    label2id = {"positive": 0, "negative": 1}

    def __init__(self, _data):

        super().__init__()

        self.data = {"review":[], "division":[]}
        for i in range(len(_data["train"]["review"])):
            
            if _data["train"][i]["division"] in dataset.label2id.keys():
                self.data["review"].append(_data["train"][i]["review"])
                self.data["division"].append(_data["train"][i]["division"])
            # else:
            #     print(_data["train"][i]["review"], _data["train"][i]["division"])

    
    def __getitem__(self, index):

        return self.data["review"][index], dataset.label2id.get(self.data["division"][index])
    
    def __len__(self):

        return len(self.data["review"])


# construct dataset
ds = dataset(data)
print(len(ds))


# split dataset
from torch.utils.data import random_split
import torch

################
# G. fix split #
################

# If we use the below line,
# trainset, validset = random_split(ds, lengths=[0.9, 0.1]) 
# the split is done randomly every time
# This causes data leakage for multi-gpus since each gpu will do this on its own
# and each gpu will have its own split randomly, which cause the problem where
# valid data of one gpu can be the train data of another gpu, where comes from the data leakage.
# Hence, the evaluation accuracy during training will be higher than expected or benchmark.
# To avoid this, we fix the split each time using a seed value.

trainset, validset = random_split(ds, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42)) 

# load tokenizer

tokenizer = AutoTokenizer.from_pretrained(ckp)


# function collate

import torch

def collate_fct(batch):

    texts, labels = [], []

    for item in batch:

        texts.append(item[0])
        labels.append(item[1])

    toks = tokenizer(texts, max_length=512, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=True)

    toks["labels"] = torch.tensor(labels)

    return toks


# dataloader

from torch.utils.data import DataLoader

##############
# B. sampler #
##############

# for dataloader, we should add a sampler to handle data on the gpus

from torch.utils.data.distributed import DistributedSampler

trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_fct, sampler=DistributedSampler(trainset))
validloader = DataLoader(validset, batch_size=32, collate_fn=collate_fct, sampler=DistributedSampler(validset))

#######################################

# load model

import torch

model = AutoModelForSequenceClassification.from_pretrained(ckp)

###################
# C. model to gpu #
###################

from torch.nn.parallel import DistributedDataParallel as DDP

# we should send model to the rank 0 gpu
# the rank is LOCAL_RANK, this info is obtained using os.
# os return a str of the local rank, so we should convert it to int

if torch.cuda.is_available():
    model = model.to(int(os.environ["LOCAL_RANK"]))

# and then wrap the model to ddp

model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

# We should also update all "to" function's parameter by this local rank inforamtion hereafter.
# See the lines marked by ***

#######################################

# define optimizer

from torch.optim import Adam

optimizer = Adam(model.parameters(), lr=2e-5)


def metric():

    model.eval()

    acc_num = 0
    count = 0

    with torch.inference_mode():

        for batch in validloader:

            batch = {k:v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()} # ***

            output = model(**batch)

            pred = torch.argmax(output.logits, dim=-1)

            count += len(batch["labels"])

            acc_num += (batch["labels"].int() == pred.int()).sum()

    ################
    # F. final acc #
    ################

    dist.all_reduce(acc_num) # by default is sum

    return acc_num / len(validset)

    #######################################


# train

def train(epoch=3, log_step=100):

    gStep = 0

    for e in range(epoch):

        model.train()

        ##############
        # H. shuffle #
        ##############

        # when we set up the distributed sampler, we remove the shuffle option
        # To optimize the training, we can do the shuffle here:

        trainloader.sampler.set_epoch(e)

        #######################################

        for batch in trainloader:
            
            if torch.cuda.is_available():
                batch = {k:v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()} # ***

            optimizer.zero_grad()

            output = model(**batch)

            loss = output.loss

            loss.backward()

            optimizer.step()

            if gStep % log_step  == 0:
                
                ##################
                # D. reduce loss #
                ##################

                # synchronize results on GPUs using All Reduce mode with average operation
                # if we don't do this, each gpu will report a different loss value

                dist.all_reduce(loss, op=dist.ReduceOp.AVG)

                #######################################

                #############
                # E. report #
                #############

                # here we print only for rank 0
                # other wise, it will print n times where n is the number of gpus

                if int(os.environ["RANK"]) == 0:
                    print(f"epoch {e+1} / {epoch}: global step: {gStep}, loss: {loss.mean().item()}")
                
                #######################################


            gStep += 1

        acc = metric( )

        if int(os.environ["RANK"]) == 0:
            print(f"epoch: {e+1} : acc: {acc}")


# start training
train()

# example output
# epoch 1 / 3: global step: 0, loss: 0.688331663608551
# epoch 1 / 3: global step: 0, loss: 0.7560659050941467
# epoch 3 / 3: global step: 100, loss: 0.13599860668182373
# epoch 3 / 3: global step: 100, loss: 0.08184975385665894

# the above code output twice the results, since the code was run on each GPU
# and each GPU reports its own results.
# We can see that the loss values are different.