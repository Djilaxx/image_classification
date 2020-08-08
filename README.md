# **Image Classification**

This repository contains image classification projects i've worked on, either for fun/education or competition on Kaggle. \
Each project have it's own readme containing information about the specific problematics of each. 

I train the models locally on my pc using a Nvidia 1080 GPU. 

## **Task**
---
The task folder contains the specific code about each project, a config.py file containing  most of the hyperparameters of the model, and a training.py file that start the training cycle. \
To start training a model on any task use this command in terminal :
```
python -m task.aerial_cactus.training
```
You can replace the **aerial_cactus** with any folder in task.
Default parameters train for **5** fold using a **Resnet18** model and return validation **accuracy** after each epoch. 
You can change these parameters as such :
```
python -m task.aerial_cactus.training --folds=3 --model=RESNET34 --metric=F1_SCORE
```

## **Image_dataset**
---
The image dataset folder contain a dataset class that loads images and corresponding labels and return them as tensors ready to be used by your model.

## **Models**
---

## **Trainer** 
---

## **Utils**
---

## **To do** 
---