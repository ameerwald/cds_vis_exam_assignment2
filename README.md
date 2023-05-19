
# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks

## Github repo link 

This assignment can be found at my github [repo](https://github.com/ameerwald/cds_vis_exam_assignment2).

## Data

The data for this assignment is from the Cifar 10 dataset which is a collection of 60000 32x32 images with 10 classes. Read more about it at this [link](https://www.cs.toronto.edu/~kriz/cifar.html).

## Assignment description 
In this assignment we had to make two different scripts training a logistic regression classifier and a neural network classifier all on the Cifar10 dataset. This was to be done using ```scikit-learn```. We needed to do the following:

- Load the Cifar10 dataset
- Preprocess the data (e.g. greyscale, reshape)
- Train a classifier on the data
- Save a classification report

## Repository 

| Folder         | Description          
| ------------- |:-------------:
| Data      | Empty, data is imported in the script 
| Models  | Saved models for the logistic regression and neural network classifiers      
| Out  | Classification Reports, one for each classifier    
| Src  | Py scripts, one for each classifier and one for preprocessing the data    
| Utils  | Preprocessing script with utility functions

## To run the scripts 
As the dataset is too large to store in my repo, use the link above to access the data. Download and unzip the data. Then create a folder called  ```data``` within the assignment 2 folder, along with the other folders in the repo. Then the code will run without making any changes. If the data is placed elsewhere, then the path should be updated in the code.

1. Clone the repository, either on ucloud or something like worker2
2. From the command line, at the /cds_vis_exam_assignment2/ folder level, run the following lines of code. 

This will create a virtual environment, install the correct requirements.
``` 
bash setup.sh
```
While this will run the scripts and deactivate the virtual environment when it is done. 
```
bash run.sh
```

This has been tested on an ubuntu system on ucloud and therefore could have issues when run another way.

## Discussion of Results 
Based on the classification reports for each classifier in the ```out``` folder, the neural network classifier slightly out performs the logistic regression classifier with 35% accuracy over 31%. It is also clear that both classifiers poorly perform with the cat images compared to the other categories. 