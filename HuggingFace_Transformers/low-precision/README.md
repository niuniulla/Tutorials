# Low Precision Training Large Models

The optimizations used to reduce training resources:

 - basic
   * accumulated gradient
   * gradient checkpoint
   * optimization
   * lower input length
   * freeze layers

 - finetuning:
   * prompt tuning
   * P-Tuning
   * prefix tuning
   * lora
   * IA3

The above pentioned optimizations were based on trainign process but don't allow model optimizations, which means that for a larger model, it can't be loaded directly on GPU for training.

We have mentioned before that the default precision  is f32 ~ 4 bytes. The idea for model optimization is to use lower precision for model: f16 (half precision), bfloat16, int8, fp4, nf4.



We tried to compare the different precision, however, this is indicative and not exact values.

Llama-2-7b-chat-hf
model size: 6.738415616 GB
total required memory:  134.77 GB
Trainable size: 4.2 GB
batch 1

|       | load size (GB)  | training size (GB)  | time (s)  | 
|---    |---              |---                  |---        |   
| full  | 20.8            | OOM                 |  -        |
| half  | 13.1            | 13.4                | 309       |
| 8 bit | 7.07            | 7.58                | 1024      |
| 4 bit | 4.3             | 4.6                 | 501       |

