

import os, torch, math
from torch.optim import Adam
import torch.distributed as dist
from datasets import load_dataset
from accelerate import Accelerator
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
    _trainloader = DataLoader(_trainset, batch_size=16, collate_fn=collate_fct, shuffle=True)
    _validloader = DataLoader(_validset, batch_size=32, collate_fn=collate_fct, shuffle=False)

    return _trainloader, _validloader


# model
def get_model(ckp):    

    # load model
    _model = AutoModelForSequenceClassification.from_pretrained(ckp)

    # define optimizer
    _optimizer = Adam(_model.parameters(), lr=2e-5)

    return _model, _optimizer


# evaluate
def metric(_model, _validloader, _accelerator: Accelerator):

    _model.eval()

    acc_num = 0

    with torch.no_grad():

        for batch in _validloader:

            output = _model(**batch)

            pred = torch.argmax(output.logits, dim=-1)

            preds, refs = _accelerator.gather_for_metrics((pred, batch["labels"]))

            acc_num += (refs.int() == preds.int()).sum()

    return acc_num / len(_validloader.dataset)



# train

def train(_model, _optimizer, _trainloader, _validloader, _accelerator: Accelerator, resume=None, _epoch=3, _log_step=10):

    gStep = 0

    #############################
    # VIII.B. add resume option #
    #############################

    # set resume step and epoch from where we resume training

    resume_step = 0
    resume_epoch = 0

    # decide if we resume training or restart

    if resume is not None:

        # load resources to resume

        _accelerator.load_state(resume)

        # compute number of steps in a loader by taking consideration of accumulated steps

        steps_per_epoch = math.ceil(len(_trainloader) / _accelerator.gradient_accumulation_steps)

        # get the current step from the saving file name

        resume_step = gStep = int(resume.split("step_")[-1])

        # compute the epoch

        resume_epoch = resume_step // steps_per_epoch

        # compute the resume step of the resume epoch, this is not the global step
        # this will be used to skip the batches in the loader while training (see below)

        resume_step -= resume_epoch * steps_per_epoch

        _accelerator.print(f"resume from checkpoint -> {resume}, epoch: {resume_epoch}, step: {resume_step}")

    # change the range of the training loop, now start from the resume epoch instead of 0

    #for e in range(_epoch):
    for e in range(resume_epoch, _epoch):

        _model.train()

        # skip the steps if resume

        if resume and e == resume_epoch and resume_step != 0:
            active_dataloader = _accelerator.skip_first_batches(_trainloader, resume_step * _accelerator.gradient_accumulation_steps)
        else:
            active_dataloader = _trainloader

    ##########################################

        for batch in active_dataloader:

            ###########################
            # V.B. accumulate context #
            ###########################

            # use acceletor to set a accumulation context

            with _accelerator.accumulate(_model):

                _optimizer.zero_grad()

                output = _model(**batch)

                loss = output.loss

                _accelerator.backward(loss)

                _optimizer.step()

                #################################
                # V.C. update accumulation step #
                #################################

                # if we increment simply the global step, it will only count the data load but not the accumulation
                
                if _accelerator.sync_gradients:

                    gStep += 1

                    # meanwhile, we should put the report under the condition too
                    # otherwise, it will print twice the step with different losses

                    if gStep % _log_step  == 0:

                        loss = _accelerator.reduce(loss, "mean")

                        _accelerator.print(f"epoch {e+1} / {_epoch}: global step: {gStep}, loss: {loss.item()}")
                        _accelerator.log({"loss": loss.item()}, gStep) # VI.D. log variables

                ##########################################

                    ###############
                    # VII.A. save #
                    ###############

                    if gStep % 50 == 0 and gStep != 0:

                        # option1: uncomment the following line to use the accelerato's save 
                        # _accelerator.save_model(_model, _accelerator.project_dir + f"/step_{gStep}")
                        
                        # option2
                        _accelerator.unwrap_model(_model).save_pretrained(
                            save_directory  = _accelerator.project_dir + f"/step_{gStep}/model",
                            is_main_process = _accelerator.is_main_process,
                            state_dict      = _accelerator.get_state_dict(_model),
                            save_function   = _accelerator.save
                        )

                    ##########################################


                        ###########################
                        # VIII.A. save for resume #
                        ###########################

                        _accelerator.save_state(_accelerator.project_dir + f"/step_{gStep}")

                        ##########################################



        acc = metric(_model, _validloader, _accelerator)

        _accelerator.print(f"epoch: {e+1} : acc: {acc}")
        _accelerator.log({"acc": acc}, gStep) # VI.D. log variables

        ######################
        # VI.C. end tracking #
        ######################
        
        _accelerator.end_training()

        #################################################


def main():

    # my P2P calble is not connected, so I have to disable this option when running multiple GPUs
    os.environ["NCCL_P2P_DISABLE"] = "1"

    # checkpoint for model
    ckp = "google-bert/bert-base-uncased"

    #############################
    # IV.A. set mixed precision #
    #############################

    # can use mixed_precision="bf16"/"fp16"

    ##################################
    # V.A. set gradient accumulation #
    ##################################

    # specifiy the accumulation steps

    #####################
    # VI.A. set logging #
    #####################

    # can be any of Tensorboard, Wandb, CometML, Aim, MLFlow, Neptune, Visdom
    # but depending on the log lib, there are more or less requirements to adjust.

    accelerator = Accelerator(mixed_precision="bf16",           # <- for mixed precision
                              gradient_accumulation_steps=1,    # <- for gradient accumulation
                              log_with="tensorboard",           # <- for logging
                              project_dir="./ckpts")            # <- for logging
    
    #################################################


    ##########################
    # VI.B. initiate tracker #
    ##########################
    
    accelerator.init_trackers("runs")

    #################################################

    trainloader, validloader = get_dataloader(ckp)

    model, optimizer = get_model(ckp)

    model, optimizer, trainloader, validloader = accelerator.prepare(model, optimizer, trainloader, validloader)


    ###########################
    # VIII.C. resume training #
    ###########################

    # comment training without resume

    # train(model, optimizer, trainloader, validloader, accelerator)

    # add argument of resume, which is the folder name of last saved step

    train(model, optimizer, trainloader, validloader, accelerator, resume=accelerator.project_dir+"/step_100")



if __name__ == "__main__":

    main()