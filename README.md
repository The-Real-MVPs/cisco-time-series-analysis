
# Project Overview:
This team was tasked to build a model that can predict the sales for Cisco in 2023.



# Project Goals:
* Produce a Final GitHub repository containing our work
* Provide a well-documented jupyter notebook that contains our analysis
* Create presentation with our findings 
* Create a model that will forecast the sales for Sisco in 2023

# Reproduction of this Data:

         
* Clone the Repository then run the ```Final_Report_NLP-Project.ipynb``` Jupyter Notebook. 
* You will need to ensure the below listed files, at a minimum, are included in the repo in order to be able to run.
   * 
   * 
   * 
   * 
   * 

    
# Initial Thoughts
During Covid Sisco sales would increase and that would create an upward trend for years following Covid

# The Plan
* Acquire data from 
* Clean and Prepare the data 
* Explore data 
* Answer the following initial questions:

    * **Question 1.** Are there any trends in the data?
    
    * **Question 2.** Is there any seasonality in the data?

    * **Question 3.** What is the sales mean? 

    * **Question 4.** How does the average sale per month(or week) differ before in different time periods  pandemic/during pandemic /after pandemic?
    
    * **Question 5.** What are the average daily sales per week (per month)?
    
    * **Question 6.** What months have the highest/ lowest sales mean? 

    * **Question 7.** Which quarter has the highest and lowest sales mean? 

    * **Question 8.** What are the trends in sales with school districts as a subgroup (seasonality)?

    * **Question 9.** What are the negative numbers in sales? 

    * **Question 10.** Are there any anomalies in the sales? 

    * **Question 11.** Top customers by city or zipcode?

    * **Question 12.** Question top sales by hub type?

    * **Question 13.** What year has the highest sales/ lowest sales mean? 



* Develop a Model to forecast sales for 2023 
    * Evaluate models on train and validate data 
    * Select the best model based on the smallest difference in the accuracy score on the train and validate sets.
    * Evaluate the best model on test data
* Run Custom Function on a single random Data input from `GitHub` `Readme` file to predict program language of that project.
* Draw conclusions

# Data Dictionary:

    
## Features - CHANGE ONCE WE HAVE FINAL COLUMNS SELECTED
|Feature    |Description       |
|:----------|:-----------------|
|`original`| The original data we pulled from the `GitHub`|	
|`first_clean`| Text after cleaning the `html` and `markdown` code|
|`clean`|Tokenized text in lower case, with latin symbols only|
|`lemmatized`|Lemmatized text|
|`sentiment`|The coumpound sentiment score of each observation|
|`lem_length`|The length of the lemmatized text in symbols|
|`original_length`|The length of the original text in symbols|
|`length_diff`|The difference in length between the orignal_length and the length of the `clean` text|
||**Target variable:**|
|`language`|`JavaScript`, `C#`, `Java` or `Python` programming languages|


# Acquire

* We acquired the data from ....
* Each row represents.......
* Each column represents ......
We acquired 432 entries.

# Prepare

**Prepare Actions:**

* **NULLS:** There were no null values all repositories contained a readme for us to reference
* **FEATURE ENGINEER:** 
* **COLUMNS KEPT:** All Data acquired was used.
* **RENAME:** Columns for Human readability.    
* **REORDER:** 




# Explore

* 
* 
* 
* 

## Exploration Summary of Findings:
* 
* 
* 
* 
* 
* 
* 

# Modeling

### Features that will be selected for Modeling:
* All continious variables:
    - 
    - 
    - 
    - 

**The models we created**

We used following classifiers (classification algorithms): 
- Decision Tree, 
- Random Forest, 
- Logistic Regression,
- Gaussian NB,
- Multinational NB, 
- Gradient Boosting, and
- XGBoost. 


## Modeling Summary:
- The best algorithm  is 
- It predicts with accuracy:
    - % on the train set
    - % on the validate set
    - % on the test set
- It makes % better predictions on the test set than the baseline model.


# Conclusions: 
*The goals of the project were:*
- 
- 
- 

*Result:*

- 
- 
- 


## **Recommendations and next steps:**
- 
- 
