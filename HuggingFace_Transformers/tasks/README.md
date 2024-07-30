# Tasks

This section aims to:

* choose model for different tasks such as indexing, text generation...
* train the model
* Do inference to generate results

Each notebook contain one type or a subtype of a task, and follow a structure:

  I. Presentation: explain what the task is about, the metrics to use and the input data structure...
 
 II. The model details, input, output, loss...

 III. The training + inference application details:

    1) import of modules
    2) Data loading
    3) data spliting
    4) tokenization of dataset
    5) model loading / define
    6) metrics
    7) define training arguments
    8) construct trainer
    9) train
    10) inference

Each notebook is independent of each other (except the indexing is the following of the match), and one should be able to handle one without following a specific order.

