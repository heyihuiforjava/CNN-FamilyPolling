# CNN-FamilyPolling

Description : Python scripts to train and use CNN (convolution Neural Networks) family for images classification.

The original purpose of this project is to classify H passivated Si(100) STM images for Philip Moriarty's group in Physics school at Uni of Nottingham.

With those scripts you can easily train a family (described in report.pdf) of CNN of your choice over the dataset of your choice, then evaluate its accuracy or use it for new images classification. The CNN used during this project are provided as a starting point. 

Those python scripts use the *keras* lib with the *TensorFlow* back end.

The dataset of images used during this project can be generated here : https://github.com/OGordon100/SPM-Toolbox 
_____________________________________________________________
**Files :**

- report.pdf \n

  Report associated with this work. Probably better to read it first.

- CNN_generation.py

  Contains the CNN generation functions used by the main_training.py script. One can add its own function to generate desired model. The function's prototype is : `def CNN_xxx(input_shape, output_number):` with `input_shape` a tuple of integers and `output_number` an integer. The function must return a **compiled** keras model. Then the new function must be referenced in the `CNN_global()`'s dictionary. 
  
- main_training.py

  Script used for train a new family of CNN over the dataset of your choice. Most parameters can be changed using the variable available and the begining of the main part of the script.
   
- main_training.py

  Script used for evaluate a family of CNN over the dataset of your choice. Most parameters can be changed using the variable     available and the begining of the main part of the script. The function `majority_polling_global(x_test, y_test=None)` can be                   used to have the CNN made new guesses for images in `x_test` if `y_test` is not provided, or to evaluate the accuracy of the CNN otherwise.
  
_____________________________________________________________
**Folders :**

- models/ 

  Contains the models generated, trained and used by the pythons scripts. 

- dataset/

  Place here the dataset you want to train the CNN with. 
  The dataset must be split in 4 parts :
  - `x_train` : contains the picture of the training set in a matrix a shape `(nbOfImages, imSize, imSize, nbOfColor)`.
  - `y_train` : contains expected outpurs of the training set in a matrix a shape `(nbOfImages, nbOfOutput)`.
  - `x_test` : similar to x_train with testing images
  - `y_test` : similar to y_train with testing expected outputs
  
  To use the trained CNN provided, the folder datasets/ must contain a STM dataset .mat file as provided by https://github.com/OGordon100/SPM-Toolbox 

- test_modules/

  Contains some useful script to display and save as png the images from dataset.
  
- .idea/

  Part of the PyCharm project, do not modify.
