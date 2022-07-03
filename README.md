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
3. Model selection and training
    - I chose transfer learning approach to train a classifier for classifying the images into 10 classes.
    - I chose transfer learning approach because the dataset is not large, and I think that it is better to start from a pre-trained model trained on a large dataset to use that knowledge in this problem. I think this is better than training a model from scratch which requires a large dataset. Moreover, I think that transfer learning approach is better for generalization in this case, and especially if this model runs in production.
    - I chose MobileNetV2 as a base model, and initialized its weights using ImageNet weights. I found that ImageNet dataset contains close images to this problem, which helps in transfer learning in this case as the clothing dataset domain is close to the domain of the ImageNet dataset.
    - I added a global average pooling layer and a dense layer with 10 neurons (corresponding to the 10 classes) with a softmax activation function on top of the baseline MobileNetV2 model.
4. Model evaluation and error analysis
5. Receptive field calculation
6. FLOPS and MAACs calculation

## Results

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
