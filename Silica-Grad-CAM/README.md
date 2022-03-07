# Grad-CAM notebooks
This file contains the instrucutions to generate Grad-CAM attention heatmaps. We have used a neural network interpretation method to understand the decisions of the trained cnn. It is suggested to read and understand first the README file of the 'Silica-CNN' folder.

## System requirements
The Grad-CAM attention heatmaps have been generated on Google Colab version Pro. 

## Installation guide
Since we have worked on Google Colab you don't need to install anything (you can simply run the Google Colab notebooks stored in the 'Colab_Notebooks' folder). 
In order to visualize Grad-CAM heatmaps you need a trained CNN. Our code is based on the 'silicanets-data' dependecies explained for the training of CNN (go to the README of 'Silica-CNN' folder). In the case that you don't want to retrain a CNN you can download the 'trained_models' folder from ZENODO LINK (this model predicts the rupture strain at variable disorder). You need simply to recreate the 'silicanets-data' structure in your Google Drive and place the trained model in the 'silicanets-data/output' folder.

## Instructions to generate Grad-CAM attention heatmaps

In order to generate Grad-CAM attention heatmaps you simply need to run the notebooks contained in 'Colab_Notebooks'.
The 'Colab_Notebooks' folder contains:

- Notebook 'Grad-CAM_4_pixels' which produce attention heatmaps from the 'conv5_ block' of ResNet50.
- Notebook 'Grad-CAM_8_pixels' which produce attention heatmaps from the 'conv4_ block' of ResNet50.
- Notebook 'Grad-CAM_16_pixels' which produce attention heatmaps from the 'conv3_ block' of ResNet50.
- Notebook 'Grad-CAM_32_pixels' which produce attention heatmaps from the 'conv2_ block' of ResNet50.

Each notebook produces a Grad-CAM attention heatmap of a selected silica configuration displayed as an image.
The codes take few minutes to generate the Grad-CAM attention heatmap.





