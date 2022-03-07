# Silica 2D ML ResNet training
This file contains the instrucutions to train the convolution neural networks for the prediction of silica properties. In particular, we have trained a ResNet50 for the prediction of different silica 2D properties as disorder, strain, fracture location and fracture path. 

## System requirements
The training of the cnn has been performed on Google Colab version Pro, with GPU.

## Installation guide
Since we have worked on Google Colab you don't need to install anything (you can simply run the Google Colab notebooks stored in the 'Colab_Notebooks' folder). 
The datasets and the output are stored in a google drive account connected to the Google Colab notebooks (the notebooks contain all the instruction to connect to Google Drive and read the data).
In order to respect the dependecies used in our tests you need to:

-Create a folder in your Google Drive, named 'silicanets-data'.
-Inside 'silicanets-data' you need to create two folder: 'data' and 'output'. The 'data' folder stores the silica datasets containing the images and mechanical properties. The 'output' folder will store the weights and the information of the trained cnn.

The datasets containing silica images and its mechanical properties can be obtained from ZENODO-LINK. You simply have to create the following dependecies:

- silicanets-data/data/ml_dataset_fracture_atoms_var02.tar.gz, the dataset for the predicition of the path fracture.
- silicanets-data/data/ml_dataset_var02.tar.gz, the dataset containing silica configurations with a fixed disorder = 0.2
- silicanets-data/data/ml_dataset_variable_var.tar.gz, the dataset containing silica configurations with variable disorder

## Instructions to train the CNN

In order to train the CNN for the prediction of silica properties you simply need to run the notebooks contained in 'Colab_Notebooks'.
The 'Colab_Notebooks' folder contains:

- Notebook 'learn-disorder' for the prediction of the disorder of a silica 2D configuration
- Notebook 'learn-strain' for the prediction of the rupture strain of a silica 2D configuration
- Notebook 'learn-location-1-d' for the prediction of the first bond break location of a silica 2D configuration.
- Notebook 'learn-fratcure-atoms' for the prediction of an image which represents the fracture path of a silica 2D configuration. This notebooks contains a custom architecture based on ResNet50 and inspired by autoencoders.

Once the CNN are trained and tested you can find all the output results in the 'silicanets-data/output/model-name' folder. Each 'model-name' folder contains the trainig,validation and test sets, the saved wieghts and history of the trained CNN and some figures which summarize the results of the testing.

The training procedure requires 2 hours with a Google Colab Pro version with the use of GPU.



