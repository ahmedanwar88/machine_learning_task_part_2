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
    - I chose the clothing images dataset available at: LINK : [Dataset](https://github.com/alexeygrigorev/clothing-dataset-small)
2. Dataset visualization and preparation
3. Model selection and training
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
