# machine_learning_task_part_2
This repository includes part 2 solution of the Machine Learning task.

## Repository structure

-The solution for this part is provided as three notebooks in the 'notebooks' directory:

1. Machine_Learning_Engineer_Task_Part_2.ipynb: This notebook includes downloading and preparing the data, training the classifier model and evaluation.
2. Receptive_Field_Calculation.ipynb: This notebooks includes calculating the receptive field of the used classifier model.
3. MAACs_FLOPs_Calculation.ipynb: This notebook includes calculating MAACs and FLOPS for each layer in the used classifier model.

-The 'utils' directory includes two modified Python scripts that need to be replaced in the repositories used to calculate the receptive field and FLOPS and MAACs. More information is provided in the notebooks.

## Steps

1. Selecting and downloading the dataset
2. Dataset visualization and preparation
3. Model selection and training
4. Model evaluation and error analysis
5. Receptive field calculation
6. FLOPS and MAACs calculation

## Approach

1. Selecting and downloading the dataset
    - I chose the clothing images dataset available at: [Dataset](https://github.com/alexeygrigorev/clothing-dataset-small).
    - This dataset is subset of the dataset available at: [Dataset](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full).
    - This dataset contains the most 10 popular classes: dress, hat, longsleeve, outwear, pants, shirt, shoes, shorts, skirt and t-shirt.
    - The dataset is split into training, validation and test sets.
    - I chose this dataset and not the larger one because the larger dataset contains irrelevant classes such as: not sure and skip, and some other classes have very few images. Therefore, I chose the 10-class dataset as I think it would be better for training a classifier for production.

2. Dataset visualization and preparation
    - I visualized the distribution of the training, validation and test sets.
    - The training set contains 3068 images, the validation set contains 341 images and the test set contains 372 images.
    - The class 't-shirt' has the largest number of images in the dataset.
    - The visualization of the dataset distribution is provided in the notebook.
    - I created dataset generators for the training, validation and test sets using Keras generators with a batch size of 32.
    - I resized the images to (224, 224), applied normalization between (-1, 1) and applied augmentation techniques: horizontal flipping and rotation.

3. Model selection and training
    - I chose transfer learning approach to train a classifier for classifying the images into 10 classes.
    - I chose transfer learning approach because the dataset is not large, and I think that it is better to start from a pre-trained model trained on a large dataset to use that knowledge in this problem. I think this is better than training a model from scratch which requires a large dataset. Moreover, I think that transfer learning approach is better for generalization in this case, and especially if this model runs in production.
    - I chose MobileNetV2 as a base model, and initialized its weights using ImageNet weights. I found that ImageNet dataset contains close images to this problem, which helps in transfer learning in this case as the clothing dataset domain is close to the domain of the ImageNet dataset.
    - I added a global average pooling layer and a dense layer with 10 neurons (corresponding to the 10 classes) with a softmax activation function on top of the baseline MobileNetV2 model.
    - I used the MobileNetV2 base model as a feature extractor and froze its weights, and trained only the classifier head.
    - I trained the model with Adam optimizer using a learning rate of 0.0001 for 200 epochs.
    - I used categorical cross entropy as the loss function.
    - The best checkpoint can be found as a Keras .hdf5 file at: [Best Checkpoint](https://drive.google.com/file/d/1-5yrB8P9lJbRbR2ZNqratGDGNgz7TXzP/view?usp=sharing).
4. Model evaluation and error analysis
    - I calculated the accuracy, loss, precision, recall, F1-score and confusion matrix using the best checkpoint and the unseen test data.
    - I chose these metrics for a better error analysis to identify the classes that the classifier predicts correctly and the classes that the classifier has problems in their predictions.
5. Receptive field calculation
    - I used this repository to calculate the receptive field of the model: [Repo.](https://github.com/google-research/receptive_field).
    - The overall receptive field is reported to be 491.
6. FLOPS and MAACs calculation
    - I used this repository to calculate MAACs and FLOPS of every layer in the model: [Repo.](https://github.com/ckyrkou/Keras_FLOP_Estimator).
    - The reported MAACs and FLOPS for every layer is reported in the notebook.
    - I added a function to get the most computationally expensive layers and their MAACS and FLOPS. The desired number of layers can be configured by the user.

## Results

- The best checkpoint has a CCE loss of 0.46 and an accuracy of 85.48% on the unseen test data.
- Precision, recall and F1-score:

Class | Precision | Recall | F1-score
 ------------ | ------------- | ------------ | ------------- 
Dress | 0.73 | 0.73 | 0.73 
Hat | 0.90 | 0.75 | 0.82
Longsleeve | 0.82 | 0.81 | 0.81
Outwear | 0.75 |  0.87 | 0.80 
Pants | 0.88 | 1.00 | 0.93
Shirt | 0.67 | 0.54 | 0.60
Shoes | 1.00 | 0.99 | 0.99 
Shorts | 0.96 | 0.77 | 0.85
Skirt | 1.00 | 0.75 | 0.86
T-shirt | 0.83 | 0.92 | 0.87

- Confuion matrix:

![Confusion Matrix](https://github.com/ahmedanwar88/machine_learning_task_part_2/blob/main/cm.png)

## Comments and Discussions

1. F1-scores and confusion matrix
    - From the F1-scores, it is clear that most of the classes are classified correctly. The lowest F1-scores are for shirt (0.60) and dress (0.73).
    - From the confusion matrix, it is clear that most confusions are between:
        - Shirt: predicted as longsleeve or outwear.
        - Dress: predicted as t-shirt.
        - Shorts: predicted as pants.
2. Receptive Field
    - The receptive field is calculated to be 491 in x and y dimensions.

3. FLOPS and MAACs
    - The most 10 computationally expensive layers are reported to be:
    
Layer | FLOPS
------------ | -------------
Conv_1 | 40140800.0
block_1_expand | 38535168.0
block_16_project | 30105600.0
Conv1 | 21676032.0
block_2_expand | 21676032.0
block_2_project | 21676032.0
block_3_expand | 21676032.0
block_11_expand | 21676032.0
block_11_project | 21676032.0
block_12_expand | 21676032.0