# MACHINE LEARNING MODEL OF DIFFERENTIAL PRIVACY BUDGET AUTOMATIC SELECTION MECHANISM  
In this work, we train the machine learning model using accuracy, privacy leakage and dataset as features to find the appropriate privacy budget.


Because different privacy budgets will affect the accuracy and privacy leakage of the machine learning model.  
Users can either **"ask for accuracy and privacy leakage"** or **"ask for privacy leakage only"**.  
</br>
**[Ask for Accuracy and Privacy Leakage]**  
<img src="https://github.com/inin1999/autoselectdpml/blob/main/img/System_Architecture_for_Accuracy%20_and_Privacy_Requirements.png" width="450"/>  
</br>
Our trained model will give us the appropriate privacy budget.  
**If the user asks for privacy leaks only, the possible accuracy rates will be given separately.**  
**[Ask for Privacy Leakage Only]**  
<img src="https://github.com/inin1999/autoselectdpml/blob/main/img/System_Architecture_for_Privacy_Requirements.png" width="450"/>  
</br>

# DataSet  
In this work, two datasets are used, **Cifar100** and **Purchase100**

# How to Use
1. Put the labeled data into the folder named "original_data".
2. Use one of the files "preprocess_dataset_**.py" to preprocess the data.  
(2.5. To merge the preprocessed data from all scenarios, use "merge_dataset.py".)  
The pre-processed data will be stored in a folder named "data".
3. Training the model  
The models Decision Tree, CNN, KNN, Logistic_Regression, Random_Forest can be trained in the folders **cifar100_model**, **cifar100_model_leak**, **purchase_model** and **purchase_model_leak**.  
The trained model will be saved in a folder called model.  
4. Testing the model  
The folder named test can be used to test the trained model.

