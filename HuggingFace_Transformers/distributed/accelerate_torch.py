
# argument
# 1) enter in terminal: accelerate config
#    Some prompt questions will show in terminal, use keyboard to select(arrows), valid (enter) the options
#    Arguments used in this context are shown below: 
#       * Please select a choice using the arrow or number keys, and selecting with enter
#           ➔  This machine                                                                                                                       
#              AWS (Amazon SageMaker)  
#       * Which type of machine are you using?                                                                                                   
#         Please select a choice using the arrow or number keys, and selecting with enter
#          ➔  No distributed training
#             multi-CPU
#             multi-XPU
#             multi-GPU
#             multi-NPU
#             multi-MLU
#             TPU
#       * How many different machines will you use (use more than 1 for multi-node training)? [1]:
#       * Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: 
#       * Do you wish to optimize your script with torch dynamo?[yes/NO]:                                                                        
#       * Do you want to use DeepSpeed? [yes/NO]:                                                                                                
#       * Do you want to use FullyShardedDataParallel? [yes/NO]:                                                                                 
#       * Do you want to use Megatron-LM ? [yes/NO]:                                                                                             
#       * How many GPU(s) should be used for distributed training? [1]:2
#       * What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:
#       * Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]:
#       * Do you wish to use FP16 or BF16 (mixed precision)?
#       * Please select a choice using the arrow or number keys, and selecting with enter
#         ➔  no
#            fp16
#            bf16
#            fp8
#
# 2) At the end, a message is shown: accelerate configuration saved at ~/.cache/huggingface/accelerate/default_config.yaml
#    The file contains all options:
#     ```
#     compute_environment: LOCAL_MACHINE
#     debug: false
#     distributed_type: MULTI_GPU
#     downcast_bf16: 'no'
#     enable_cpu_affinity: false
#     gpu_ids: all
#     machine_rank: 0
#     main_training_function: main
#     mixed_precision: 'no'
#     num_machines: 1
#     num_processes: 2
#     rdzv_backend: static
#     same_network: true
#     tpu_env: []
#     tpu_use_cluster: false
#     tpu_use_sudo: false
#     use_cpu: false
#     ```
# 3) we can now run the program using: accelerate launch accelerate_torch.py 
# 
# 4) To modify the config, there are some options:
#     * redo step 1) to regenerate the config file
#     * modify directly the config file
#     * copy the config file and modify it, we then can use it using the argument "--config_file" to pass the 
#       the new config file to the accelerator program


import os, torch
from torch.optim import Adam
import torch.distributed as dist
from datasets import load_dataset


#########################
# A. import accelerator #
#########################

from accelerate import Accelerator

##################################

from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# dataset
from torch.utils.data import Dataset

class dataset(Dataset):

    label2id = {"positive": 0, "negative": 1}

    def __init__(self):

        super().__init__()

        # load data
        _data = load_dataset("davidberg/sentiment-reviews")

        self.data = {"review":[], "division":[]}

        for i in range(len(_data["train"]["review"])):
            
            if _data["train"][i]["division"] in dataset.label2id.keys():
                self.data["review"].append(_data["train"][i]["review"])
                self.data["division"].append(_data["train"][i]["division"])
    
    def __getitem__(self, index):

        return self.data["review"][index], dataset.label2id.get(self.data["division"][index])
    
    def __len__(self):

        return len(self.data["review"])


# dataloader
def get_dataloader(ckp):

    # construct dataset
    ds = dataset()

    # split dataset
    _trainset, _validset = random_split(ds, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42)) 

    # load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(ckp)

    # function collate
    def collate_fct(_batch):

        _texts, _labels = [], []

        for item in _batch:

            _texts.append(item[0])
            _labels.append(item[1])

        toks = _tokenizer(_texts, max_length=512, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=True)

        toks["labels"] = torch.tensor(_labels)

        return toks

    # dataloader
    #########################
    # E. remove ddp sampler #
    #########################

    # comment the old loaders

    # trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_fct, sampler=DistributedSampler(trainset))
    # validloader = DataLoader(validset, batch_size=32, collate_fn=collate_fct, sampler=DistributedSampler(validset))

    # we put back the shuffle option here

    _trainloader = DataLoader(_trainset, batch_size=32, collate_fn=collate_fct, shuffle=True)
    _validloader = DataLoader(_validset, batch_size=32, collate_fn=collate_fct, shuffle=False)

    # IMPORTANT: and also remove the sampler shuffle in the training loop, marked by ***

    ##################################

    return _trainloader, _validloader


# model
def get_model(ckp):    

    # load model
    _model = AutoModelForSequenceClassification.from_pretrained(ckp)

    ##########################
    # F. remove the ddp wrap #
    ##########################

    # if torch.cuda.is_available():
    #     model = model.to(int(os.environ["LOCAL_RANK"]))

    # # ddp model
    # model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    ##################################

    # define optimizer
    _optimizer = Adam(_model.parameters(), lr=2e-5)

    return _model, _optimizer


# evaluate
def metric(_model, _validloader, _accelerator: Accelerator):

    _model.eval()

    acc_num = 0

    with torch.inference_mode():

        for batch in _validloader:

            ######################
            # G.2. remove to gpu #
            ######################

            # we don't need to copy data to gpus since accelerator will be in charge of this.
            # so remove the following.

            # batch = {k:v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}

            ##################################

            output = _model(**batch)

            pred = torch.argmax(output.logits, dim=-1)

    ###################
    # H. gather preds #
    ###################

            # we use the accelerator's function to handle the data.
            # In this case, it will return the prediction corresponding the the original data, 
            # but not the padded data.

            # be carefule, the input of gather is a tuple

            preds, refs = _accelerator.gather_for_metrics((pred, batch["labels"]))

            # so we can simply compare the predictions to the references provided by accelerator

            acc_num += (refs.int() == preds.int()).sum()

    # we don't need the reduce anymore, so remove it

    # dist.all_reduce(acc_num) # by default is sum

    ##################################

    return acc_num / len(_validloader.dataset)



# train

def train(_model, _optimizer, _trainloader, _validloader, _accelerator: Accelerator, _epoch=3, _log_step=20):

    gStep = 0

    for e in range(_epoch):

        _model.train()

        # trainloader.sampler.set_epoch(e) # ***

        for batch in _trainloader:
            
            ######################
            # G.1. remove to gpu #
            ######################

            # we don't need to copy data to gpus since accelerator will be in charge of this

            # if torch.cuda.is_available():
            #     batch = {k:v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}

            ##################################

            _optimizer.zero_grad()

            output = _model(**batch)

            loss = output.loss

            ###########################
            # D. accelerator backward #
            ###########################

             # loss.backward()
            _accelerator.backward(loss)

            ##################################

            _optimizer.step()

            if gStep % _log_step  == 0:

                ####################
                # I. change report #
                ####################

                # remove the reduce of DDP, remove the following 3 lines

                # dist.all_reduce(loss, op=dist.ReduceOp.AVG)

                # if int(os.environ["RANK"]) == 0:
                #     print(f"epoch {e+1} / {_epoch}: global step: {gStep}, loss: {loss.mean().item()}")
                
                # use accelerator's reduce which is equavalent to all_reduce of DDP
                # accelerator's reduce is not inplace, which is different than DDP

                loss = _accelerator.reduce(loss, "mean")
                
                # use the print of acelerator to report
                # simple print will print n times with n the number of gpus
                
                _accelerator.print(f"epoch {e+1} / {_epoch}: global step: {gStep}, loss: {loss.item()}")

            gStep += 1

        acc = metric(_model, _validloader, _accelerator)

        if int(os.environ["RANK"]) == 0:
            print(f"epoch: {e+1} : acc: {acc}")


def main():

    # my P2P calble is not connected, so I have to disable this option when running multiple GPUs
    os.environ["NCCL_P2P_DISABLE"] = "1"

    #############################
    # B. Initialize accelerator #
    #############################

    # checkpoint for model
    ckp = "google-bert/bert-base-uncased"

    # fisrt, we don't need to specify the backend, 
    # this will be taken care of by the accelerator

    # if train on linux, setup nccl
    # dist.init_process_group(backend="nccl")

    # then we instanciate the accelerator

    accelerator = Accelerator()

    ##################################

    trainloader, validloader = get_dataloader(ckp)

    model, optimizer = get_model(ckp)


    ################
    # C. wrap objs #
    ################

    model, optimizer, trainloader, validloader = accelerator.prepare(model, optimizer, trainloader, validloader)

    ##################################

    train(model, optimizer, trainloader, validloader, accelerator)


if __name__ == "__main__":

    main()