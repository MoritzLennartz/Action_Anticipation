# Action_Anticipation

<h1>This repository is dedicated to the action anticipation code of the paper:
<b>'Is attention to bounding boxes all you need for pedestrian action prediction?'</b></h1>
https://arxiv.org/abs/2107.08031

<h1>
<b>Problem Formulation:</b>
</h1>

We formulate the pedestrian action prediction as a binary classification problem where the objective is to predict whether the pedestrian will start crossing the street at some time t given the observation of length m. The event is at the time that the pedestrian starts to cross or at the last frame the pedestrian is observable in case no crossing takes place. 

The observation data sequence for each pedestrian is sampled so that the last frame of observation is between 1 and 2s (or 30 − 60 frames at 30 fps) prior to the crossing event start (the critical crossing frame is provided directly in the dataset annotations).


<h1>
<b>Data Preprocessing:</b>
</h1>

For every pedestrian, we generate observation sequences going from the lower bound of the time to event interval into its upper bound (TTE = [30 - 60] in our case), where each obs_sequence is 0.5 sec (16 frames) along with an overlap of 60% with the next obs_sequence.



<h1>
<b>Models Testing:</b>
</h1>

We applied multiple Transformer variations for the classification task.

* Encoder-only architecture with classifier on the top: TEO model.
* Encoder-only architecture with pooling layers between the Encoder Layers and a classifier on the top: TEP model.
* Full Transformer Encoder-Decoder model with weighted average of the regression and classification Loss: TED model.

To test the pre-trained models:

1- Download the models in the checkpoints repesatoire from:
https://drive.google.com/drive/folders/1FMB1ywiQ9-mT2g3YO6GrN7hOZ9bdqyUF?usp=sharing

2- run the Test_TEO.py, Test_TEP.py, or Test_TED.py file according to the tested variant model.

For training the models from scratch:

1- run the TEO_experiment.py, TEP_experiment.py, or TED_experiment.py file according to the training variant model.








