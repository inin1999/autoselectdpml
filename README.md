# MACHINE LEARNING MODEL OF DIFFERENTIAL PRIVACY BUDGET AUTOMATIC SELECTION MECHANISM  
In this work, we train the machine learning model using accuracy, privacy leakage and dataset as features to find the appropriate privacy budget.

Because different privacy budgets will affect the accuracy and privacy leakage of the machine learning model.  
Users can either **""ask for privacy leakage only""** or **""ask for accuracy and privacy leakage""**.  
Our trained model will give us the appropriate privacy budget.  
**[If the user asks for privacy leaks only, the possible accuracy rates will be given separately.]**

# How to Use
1. Put the labeled data into the folder named "original_data".
2. Use one of the files "preprocess_dataset_**.py" to preprocess the data.
2.5. To merge the preprocessed data from all scenarios, use "merge_dataset.py".  
The pre-processed data will be stored in a folder named "data".
3. Training the model
