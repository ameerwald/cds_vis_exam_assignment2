
# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks

This assignment can be found at my github [repo](https://github.com/ameerwald/cds_lang_exam_assignment2).

This assignment asked for two different scripts training a logistic regression classifier and a neural network classifer all on the Cifar10 dataset. This was to be done using ```scikit-learn```. We needed to do the following:

- Load the Cifar10 dataset
- Preprocess the data (e.g. greyscale, reshape)
- Train a classifier on the data
- Save a classification report

The Cifar 10 dataset is a collection of 60000 32x32 images with 10 classes. Read more about it at this [link](https://www.cs.toronto.edu/~kriz/cifar.html).

## Repository 

| Folder         | Description          
| ------------- |:-------------:
| In      | Empty, data is imported in the script 
| Models  | Saved models for the logistic regression and neural network classifiers      
| Out  | Classification Reports, one for each classifier    
| Src  | Py scripts, one for each classifer and one for preprocessing the data    
| Utils  | Unfinished preprocessing script. Eventually want the load_data() and preprocessing() here and called into the other scripts

## To run the scripts 

From the command line, at the assignment 2 folder level, run the following chunk of code. If it is run within a folder in assignment 2 the file paths need to be edited. 
``` 
bash setup.sh
bash run.sh
```

This has been run on an ubuntu system on ucloud and therefore could have issues when run another way. 
