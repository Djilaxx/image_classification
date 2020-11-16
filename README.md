# **Image Classification**

This repository contains an image classification pipeline and some of the projects i've worked on, either for fun/education or competition on Kaggle. \
Each project have it's own readme containing information about the specific problematics of each. 

I train the models locally on my pc using a Nvidia 1080 GPU. 
## **Data**

The data is not in the repository directly if you want to launch a model on one the projects in here you must download the data and change the config file in the task folder to be adequate. \
Links to the datasets are in the tasks Readme.
## **Task**
---
The task folder contains the specific code about each project :
 * config.py file containing  most of the hyperparameters of the model.
 * augment.py that contain the specific augmentations you want to perform on the images for training, validation and testing

### **What if i want to add a new task ?**
To add a new task you'll to create a few things : 
* a new folder in the task/ folder containing a __init__.py file, a config.py file and a augment.py file
* copy and paste the content of another config.py file and change the information to be adequate with your task
* Add the augment you require to the augment.py file

To start training a model on any task use this command in terminal :
```
python -m run --task=aerial_cactus
```
You can replace the **aerial_cactus** with any folder in task.
Default parameters train for **5** folds using a **resnet18** model, **cross_entropy** loss and return validation **accuracy** after each epoch. 
You can change these parameters as such :
```
python -m run --task=aerial_cactus --folds=3 --model=RESNET34 --loss=cross_entropy --metric=F1_SCORE
```

The parameters can take different values :
* **task** : The task you want to train a model on, atm you can train a model on the aerial_cactus task, melanoma_classification & blindness_detection
* **folds** : this parameter determine the number of folds to create into the dataset. If you choose 5 for example, the dataset will be divided in 5, train a model on 4 folds and validate on the last (folds 0, 1, 2, and 3 for training and 4 for validation. Then, it'll train on folds 0, 1, 2, 4 and validate on 3 etc...).
* **model** : You can choose any model that is in the models/ folder, name must be typed in MAJ like in the example above.
* **loss** : You can choose any loss function that is in the loss/ folder.
* **metric** : The training procedure return loss throughout training and validation, and accuracy by default, but you can choose to return another validation metric if needed (the metrics you can choose from are in **utils/metrics.py**)

## **To do** 
---
* Configure the inference file
* Add scheduler
* Add more models
* Add more loss functions available
* Add metrics
* Debug the AUROC metric (not usable atm)
* Add logger to have information about each training run you perform