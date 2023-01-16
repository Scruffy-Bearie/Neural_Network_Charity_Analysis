# Neural Network Charity Analysis - Preprocessing Data and Altering Model Paramaters to Improve Predictive Accuracy

## Overview

An artificial neural network is a group of interconnected neurons that employs a mathematical or computational model to engage in connected computation and can, potentially, change its structure based on external or internal information that flows through the network.  More directly, neural networks are non-linear statistical data modeling or decision-making tools that can be used to model relationships and patterns in complex data sets.  The client for this study, Alphabet Soup, is a non-profit philanthropic foundation that raises and donates funds to other organisations that has become concerned with the proportion of their donations that are not used effectively.  The purpose of this analysis was to, using application and success-tracking data provided by Alphabet Soup, develop a neural network capable of detecting future applicants likely to make effective use of donations and to, moreover, make modifications to the original data set and/or neural network model such that the predictive accuracy of the model was equal to or greater than 75%.  

## Results

After first importing all relevant modules, the data set was loaded into a Jupyter Notebook file running in/on a previously established machine learning kernel. Before continuing with data pre-processing, the following variables, considered to be neither “targets” or “features” of the data set were removed:

•	[‘EIN’]

•	[‘NAME’]

Once the “irrelevant” data had been removed, the remaining columns were examined to determine the number of unique features in each and based on the results, the columns [‘APPLICATION_TYPE’] and [‘CLASSIFICATION’] were selected for “binning” of unique values with counts of less than 500 and 1800 respectively.  This done, all columns containing categorical data were treated with the “onehotencoder”, the resultant encoded data frame merged with the original data frame and the original columns containing categorical data removed. 

To complete pre-processing, the “target” and “features” were isolated as described below, divided into “training” and “testing” data sets using “train_test_split” and the “feature” data set treated with the StandardScaler.

•	y (“target”) = [‘IS_SUCCESSFUL’]

•	X (“features”) = All “other” columns in the data set (i.e. the data set without the [‘IS_SUCCESSFUL’] column

Based on information provided, the initial neural network was constructed using the following parameters:

•	Number of input features: Number of rows in feature data set (len(X_train[0]))

•	Number of hidden layers (neurons in layer): 2 (80,30)

•	Activation Functions for Hidden Layers: relu, relu

•	Activation Function for Output Layer: sigmoid

•	Number of Epochs: 100

Constructed as described above, the neural network performed as summarised in the diagrams below (see Figures 1, 2 and 3) and the model saved as “AlphabetSoupCharity.h5”.

### Figure 1: Summary Statistics for the Initial Neural Network
![]( https://github.com/Scruffy-Bearie/Neural_Network_Charity_Analysis/blob/main/IMAGES/Image1.png)

### Figure 2: Loss as a Function of Number of Epochs for the Initial Neural Network
![]( https://github.com/Scruffy-Bearie/Neural_Network_Charity_Analysis/blob/main/IMAGES/Figure1.png)

### Figure 3: Accuracy as Function of Number of Epochs for the Initial Neural Network
![]( https://github.com/Scruffy-Bearie/Neural_Network_Charity_Analysis/blob/main/IMAGES/Figure2.png)


### MODEL OPTIMIZATION – ATTEMPT #1

Given that the stated objective of 75% accuracy was not achieved with the initial model, an attempt was made to optimise the model by first, searching for and removing potential outliers in the [‘ASK_AMT’] column (See Figure 4):

### Figure 4: Box and Whisker Plot for the [‘ASK_AMT’] Column from the Application Data Frame
![]( https://github.com/Scruffy-Bearie/Neural_Network_Charity_Analysis/blob/main/IMAGES/Image2.png)

Once the outliers had been removed, the pre-processing of data was performed as previously described and the neural network constructed as follows:


•	Number of input features: Number of rows in feature data set (len(X_train[0]))

•	Number of hidden layers (neurons in layer): 3 (120, 80, 60)
  
  The number of hidden layers was increased to 3 based on accepted ideology regarding improved accuracy with increased number of layers; the number of neurons in the     initial layer was increased to 120 such that # neurons = 3X # input features and the number of neurons on the second and third layer was selected to produce a          “tapering” affect

•	Activation Functions for Hidden Layers: tanh, tanh, sigmoid, sigmoid
  
  The activation function for the first and second layer was changed to “tanh” based on “trial and error” 

•	Activation Function for Output Layer: sigmoid

•	Number of Epochs: 200
  
  The number of training epochs was increased in the hopes of achieving a higher overall accuracy for the model

Constructed as described above an using the data modified as described, the neural network performed as summarised in the diagrams below (see Figures 5, 6 and 7) and the model saved as “AlphabetSoupCharityOPTIMIZE1.h5”.

### Figure 5: Summary Statistics for the Neural Network for the First Optimization Attempt
![]( https://github.com/Scruffy-Bearie/Neural_Network_Charity_Analysis/blob/main/IMAGES/Image3.png)

### Figure 6: Loss as a Function of Epochs for the First Optimization Attempt
![]( https://github.com/Scruffy-Bearie/Neural_Network_Charity_Analysis/blob/main/IMAGES/Figure3.png)

### Figure 7: Accuracy as a Function of Epochs for the First Optimization Attempt
![]( https://github.com/Scruffy-Bearie/Neural_Network_Charity_Analysis/blob/main/IMAGES/Figure4.png)


### MODEL OPTIMIZATION – ATTEMPT #2

Given that the stated objective of 75% accuracy had still not been achieved, an attempt was made to further optimise the model searching for and altering potentially problematic data types – specifically entries in the [‘INCOME_AMT’] column which were classified as “object” data type (See Figures 8)

### Figure 8: Data Types for the Application Data Frame Before and After Altering [‘INCOME_AMT’]
![]( https://github.com/Scruffy-Bearie/Neural_Network_Charity_Analysis/blob/main/IMAGES/Image4.png)

Once the data type for the [‘INCOME_AMT’] column had been changed, the pre-processing of data was performed as previously described and the neural network was constructed as described in OPTIMIZATION1.  The neural network performed as summarised in the diagrams below (see Figures 9, 10 and 11) and the model saved as “AlphabetSoupCharityOPTIMIZE2.h5”.

### Figure 9: Summary Statistics for the Neural Network for the Second Optimization Attempt
![]( https://github.com/Scruffy-Bearie/Neural_Network_Charity_Analysis/blob/main/IMAGES/Image5.png)

### Figure 10: Loss as a Function of Epochs for the Second Optimization Attempt
![]( https://github.com/Scruffy-Bearie/Neural_Network_Charity_Analysis/blob/main/IMAGES/Figure5.png)

### Figure 11: Accuracy as a Function of Epochs for the Second Optimization Attempt
![]( https://github.com/Scruffy-Bearie/Neural_Network_Charity_Analysis/blob/main/IMAGES/Figure6.png)


### MODEL OPTIMIZATION – ATTEMPT #3

Given that the stated objective of 75% accuracy had still not been achieved, an attempt was made to further optimise the model using the Keras-Tuner.  The data set, as modified for OPTIMIZATION 1 and 2, was loaded into a Colab notebook and, based on results achieved so far, the Keras-Tuner was provided instructions to test possible neural networks with up to 6 hidden layers (See Figure 12):

### Figure 12: Instructions Provided to the Keras-Tuner for Model Optimization
![]( https://github.com/Scruffy-Bearie/Neural_Network_Charity_Analysis/blob/main/IMAGES/Image6.png)

Based on the results of the Keras-Tuner, a final neural network was constructed and trained and tested using the data set modified and pre-processed as described for the initial neural network and OPTIMIZATION attempts 1 and 2.  The resultant neural network performed as summarised in the diagrams below (see Figures 13, 14 and 15) and the model saved as “AlphabetSoupCharityOPTIMIZE3.h5”.

### Figure 13: Summary Statistics for the Neural Network for the Third Optimization Attempt
![]( https://github.com/Scruffy-Bearie/Neural_Network_Charity_Analysis/blob/main/IMAGES/Image7.png)

### Figure 14: Loss as a Function of Epochs for the Third Optimization Attempt
![]( https://github.com/Scruffy-Bearie/Neural_Network_Charity_Analysis/blob/main/IMAGES/Figure7.png)

### Figure 15: Accuracy as a Function of Epochs for the Third Optimization Attempt
![]( https://github.com/Scruffy-Bearie/Neural_Network_Charity_Analysis/blob/main/IMAGES/Figure8.png)


## Analysis

Accuracy of the initial neural network was found to be 72.4%, while the accuracy of the models produced by eliminating outliers from the [‘ASK_AMT’] column and by changing the data type of the [‘INCOME_AMT’] column were found to be 72.3% and 72.7% respectively.  Finally, the neural network produced using the altered data set (outliers removed and altered data type) along with results of the Keras-Tuner was found to be 72.5%.

Given that the overall objective was to achieve a model with predictive accuracy of 75% (or greater) it seems obvious that more work would be required to meet the clients expectations.  That said, seemingly endless tinkering and toying with neural network model parameters produced no significant improvement in the predictive accuracy and it could be that, due to specifics of the data set, continued attempts to model the data with a neural network might not be the most cost effective approach.  That said, and given the “binary” nature of the target variable, any attempts to further model the Alphabet Soup data set for sake of improved predictive accuracy may best be made with an unsupervised machine learning model such as the random forest classifier. 
