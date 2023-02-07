## Project Links
* Click the buttons below to see the Project Repo and Canva presentation.  

[![GitHub](https://img.shields.io/badge/Project%20GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/The-Real-MVPs/cisco-time-series-analysis)
[![Canva](https://img.shields.io/badge/Project%20Canva-%2300C4CC.svg?style=for-the-badge&logo=Canva&logoColor=white)] (https://www.canva.com/design/DAFZcPULMG4/fGHlG_J86ECvpvWeMMNh4w/edit)

[![Canva](https://img.shields.io/badge/Project%20Canva-%2300C4CC.svg?style=for-the-badge&logo=Canva&logoColor=white)](https://www.canva.com/design/DAFZcPULMG4/fGHlG_J86ECvpvWeMMNh4w/edit)

## Meet Group 1
|Team Member         |[LinkedIn]                                               |[GitHub]                              |
|:-------------------|:--------------------------------------------------------|:-------------------------------------|
|Allante Staten      |[![LinkedIn](https://img.shields.io/badge/Allante's%20linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/allantestaten)|[![GitHub](https://img.shields.io/badge/Allante's%20GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/allantestaten)|
|John Chris Rosenberger        |[![LinkedIn](https://img.shields.io/badge/Chris'%20linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/johnrosenberger/)|[![GitHub](https://img.shields.io/badge/Chris'%20GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jcrosenberger)|
|Nadia Paz           |[![LinkedIn](https://img.shields.io/badge/Nadia's%20linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nadiapaz)|[![GitHub](https://img.shields.io/badge/Nadia's%20GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nadia-paz)|
|Yvette Ibarra |[![LinkedIn](https://img.shields.io/badge/Yvette's%20linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yvette-ibarra01/)|[![GitHub](https://img.shields.io/badge/Yvette's%20GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Yvette-Ibarra)|

# Project Overview:
This team was tasked to build a model that can predict the sales for Cisco. Our goal is to explore historical Cisco sales to Texas public agencies (school districts, government agencies, municipalities, etc) to present to people who might want to buy into the cisco ecosystem if it is a good idea going forward.

# Project Goal
The goal of the project is to make the exploration of Cisco's sales with public agencies, define what features can be useful to create a prediction model. Build a model that can forecast future sales for the company. 

# Project Deliverables:
* Produce a Final GitHub repository containing our work
* Provide a well-documented jupyter notebook that contains our analysis
* Create presentation with our findings 
* Create a model that will forecast the sales for Cisco

# Project Description and Intitial Thoughts.
In this project we will focus on Time Series Analysis and we'll attempt to create forecast a model that can predict future sales. We expect to hit difficulties during modeling process, as the beginning of pandemic led to anomaly increase of telecommunication sales followed by significant drop in sales after it. That anomaly will "confuse" the models making them over- or underestimate sales dramatically depending of the time period included to train the model.

# Executive Summary
**The goal** 
The goal of the project is to make the exploration of Cisco's sales with public agencies, define what features can be useful to create a prediction model. Build a model that can forecast future sales for the company. 

**Key takeaways**

* __Acquire__ 
The data file contains information about DIR Cooperative Contract Sales. To get the information about Cisco we filtered the data by vendor name.

* __Prepare__ 
    - Removed columns with nulls only. 
    - Removed the data before 2018.
    - Added features based on the date information.
    - Created an additional data frame that summarized information about sales for every customer by the end of the day.

* __Explore__
    - Our target variable purchase_amount doesn't show the seasonality or trend. It looks more like noise data that is extremely hard to predict using traditional statistic methods.

    - Cisco customers represented in DIR data set are divided into following groups:
        - Local Governments - 36.8%
        - Independent School Districts - 31.9 %
        - Higher Education Institutions - 21.1%
        - State Agencies - 9.8%
        - Other public agencies - 0.4%
    - All sales per day of the week significantly differ from the average daily sales. 
    - Week days have higher sales than the average and weekend days almost don't have sales. 
    - Monday, Tuesaday and Wednesday seem to have same average amount of sales. 
    - Friday has the higher average sales amount among all days of the week.
    - The highest sales are happening in July, followed by April, October and June.
    - The lowest sales happen on February.
    - March has almost the same results as average sales, but there was peak in sales on March, 2020 when pandemic just started, it might happened that March is typically low on sales.
    - April results might be affected by beginning of pandemic, too.
    - Highest sales are in the 3rd quarter.
    - Lowest sales are in the 1st quarter.
    - Average qurterly sales per each quarter are not significantly different from the overall average quarterly sales.
    - In the beginning of the pandemic there was an an abnormal spike in sales.
    - There in a big outlier in order_quantity. Right before the pandemic one agency places the order for 4M Cisco's products.

* __Modeling__
    - Our best model is XGBoost regressor. It outperformed statistical ARIMA forecasting models.
    - XGBoost Regressor beats the baseline model on all train, validation and test sets. Anyway, there is much more work to do. We'd like to improve results by making weekly/monthly resamples hoping find some seasonality. We'd like to remove the some pandemic sales anomalies as well. If this won't improve model performance, our next step is going to be splitting data on 5 different sets based on the customer type and make predictions for every set.

# Reproduction of this Data:
         
* Clone the Repository
* Download the `csv` file from  [Texas Open Data Portal](https://data.texas.gov/dataset/OFFICIAL-DIR-Cooperative-Contract-Sales-Data-Fisca/w64c-ndf7).
* Create a `data` folder in the main directory and move the downloaded file into the `data` folder.
* Install all Data Science libraries
* Run the final notebook.

# The Plan
* Acquire data from 
* Clean and Prepare the data 
* Explore data 
* Answer the following initial questions:

    * **Question 1.** How the purchase amount is changing over the time?

    * **Question 2.**  Is there any trend in total purchase amount per month?

    * **Question 3.**  What are the customer types represented in the data?

    * **Question 4.**  Is the trend the same for every customer type?

    * **Question 5.** How did monthly purchase amount change for school districts during the pandemic of COVID-2019?

    * **Question 6.** Is there any significant difference in sales by the day of the week?
    
    * **Question 7.** Is there any significant difference in sales by the month?

    * **Question 8.** Is there any significant difference in sales by the quarter?



* Develop a Model to forecast sales
    * Evaluate models on train and validate data 
    * Select the best model based on the smallest difference in the accuracy score on the train and validate sets.
    * Evaluate the best model on test data
* Draw conclusions

# Data Dictionary
|Feature    |Description       |
|:----------|:-----------------|
|`customer_name`| Customer name|	
|`customer_type`| Local Government, State Agency, School District, Higher Education Institution, Other customers|
|`customer_city`|The city where the customer is located|
|`reseller_name`|The reseller that sells Cisco products to the customer|
|`reseller_city`|The city where the reseller is located|
|`customer_zip`|Customer's zip code|
|`order_quantity`|The order's quantity|
|`unit_price`|The price of the product|
|`po_number`|Purchase order number|
|`shipped_date`|The date the order was shipped|
|`order_date_copy`|The order date. Copy, because the original is used as a data frame|
|Date Features that we've created|
|`month_name`|January, February, ..., Decmber|
|`day_name`|Monday, Tuesday, ..., Sunday|
|`year`|2018, 2019, ... 2022|
|`quarter`|1, 2, 3, 4 for the 1st quarter, 2nd quarter etc|
|`month`|1, 2, 3, ..., 12 for January, February etc|
|`week`|1, 2, 3, ..., 53 The number of the week|
|`day_of_week`|0, 1, 2, ..., 6. 0: Monday, 1: Tuesday.... 6: Sunday|
|`day_of_year`|1, 2, 3..., 366 Numerical value of each day of the year|
||**Target variable:**|
|`purchase_amount`|`unit_price` * `order_qunatity`|


# Acquire and Prepare
We acquired the data from the [Texas Open Data Portal](https://data.texas.gov/dataset/OFFICIAL-DIR-Cooperative-Contract-Sales-Data-Fisca/w64c-ndf7). The origin of the data is 	[Texas Department of Information Resources](	https://dir.texas.gov/). The data file contains information about DIR Cooperative Contract Sales. To get the information about Cisco we filtered the data by vendor name.

1. **Acquire** Our first cleaning step was dropping the  7 columns with null values only and renaming columns into programming friendly format simply by replacing whitespaces with underscores and making them lower case. Our next step was dropping columns with no value for exploration or modeling purposes (for example, `vendor_adress`, `reseller_phone` etc.)

2. **Clean** We have created additional features based on the `order_date` information:
    `year` : year
    `quarter`: quarter
    `month`: month number
    `month_name`: month name
    `day_of_week`: day of week number
    `day_name`: day of week name
    `day_of_year`: day of the year
    `week`: week of the year number

3. **Prepare** The original data set had two issues. First, there was lots of missing data in 2017. We took decision to remove everything from 2017 and add +1 year to everything before. Second, the original data contains the accounting information about every transaction where every row represents a transaction. One company could have many transactions per day, including those where the `purchase_amount` was equal to zero, one cent or being negative. To fix this issue and make the data more readable we have created as well a summary data frame whith the final `purchase_amount` per day per company. This combined the number of rows from 261,886 to 34,401 rows. 

- Outliers were kept for this iteration of explore of modeling.
- To avoid data leakagee divided the data into train and test sets. 
- After the exploration we split the test set into validation and test sets as well.
    - `train` set contains data from Jan, 1 2018 till Dec, 31 2021
    - `validate` set contains data from Jan, 1 2022 till Jun, 30 2022
    - `test` set contains data from July, 1 2022 till November, 29 2022
    
- __The target variable of the project is ```purchase_amount```__


# Exploration Findings:

* **Question 1.** How the purchase amount is changing over the time?
- The purchase amount doesn't show the seasonality or trend.
- There is lots of missing data in 2017. It would negatively affect models, and provide incorrect information about the purchase amount. That's why we are going to remove 2017 and add one year to each year from 2014 till 2016.
- It looks more like `noise` data that is extremely hard to predict using traditional statistic methods.
* **Question 2.** Is there any trend in total purchase amount per month?
- After calculating the total monthly purchase amount we observed a **clear long-term positive trend** in our taget variable. Specially during 2016
- In the beginning of 2018 the sales went down a lot. We don't know for sure if the beginning of 2018 was also affected by missing data or it is a real decrease in sales for Cisco.
- The end of 2019 and beginning of 2020 there bring an increase in sales. This is the start of COVID-19 pandemic.
- Starting the middle of 2020 period sales drop back to the trend and even lower. If the values in validate and test sets continue to descrease, **our future model going to overestimate the total purchase amount**.
* **Question 3.** What are the customer types represented in the data?
- Cisco customers represented in DIR data set are divided into following groups:
    - Local Governments - 36.8%
    - Independent School Districts - 31.9 %
    - Higher Education Institutions - 21.1%
    - State Agencies - 9.8%
    - Other public agencies - 0.4%
* **Question 4.** Is the trend the same for every customer type?
- School Districts have the biggest slope of the long-term trend.
- Local Governments and State Agencies follow the same trend.
- Higher Educational Institutions have less obvious trend line.
- Higher Educational Institutions have a negative trend after 2018.
* **Question 5.** How did monthly purchase amount change for school districts during the pandemic of COVID-2019?
- School districts where constantly buying Cisco products even before COVID-19. But since the beginning of pandemic their purchase amount at least doubled.
* **Question 6.** Is there any significant difference in sales by the day of the week?
- All daily purchase amounts are significantly different from the average of weekly purchase amounts
- Week days have higher average purchase amounts compared to average of weekly purchase amounts
- Weekend days and Mondays purchase amounts are well below average of weekly purchase amounts
- Friday has the highest average sales amount among all days of the week.
* **Question 7.** Is there any significant difference in average of total purchase amount by month?
- Only February is significantly different from the average monthly purchase amount. February has the lowest number of sales.
- July and April have the highest amount of sales.
* **Question 8.** Is there any significant difference in sales by the quarter?
    - Highest sales are in the 3rd quarter.
    - Lowest sales are in the 1st quarter.
    - Average qurterly sales per each quarter are not significantly different from the overall average quarterly sales.

# Modeling

### Features that will be selected for Modeling:
We kept only order date and purchase amount for our ARIMA models. To create regression model we also kept features that are based on the date: day of week, week number, month, year.

**The models we created**
We created ARIMA statistical models and XGBoost Regressor. XGBoost Regressor outperformed all statistical models.


## Modeling Summary:
- Our best model beats a baseline on all train, validation and test sets. Anyway, there is much more work to do. It overestimates the total purchase amount from the test set by 89%.
- We definitely recommend to use the data after mid 2020 only for training the model in order to make a forecast because of the current negative trend
- Another approach would be splitting data on 5 different sets based on the customer type and create different models for each set.

In fact, we have create a forecast for Cisco Systems, Inc trades with public agencies. We used XGBoost as our prediction model and trained it on the data from June, 2020 till the end of 2021, and validated it on the available data for 2021 (till Nov, 29). Then, we generated a list of dates from January 1, 2023 till December 31, 2023. **This model recognized the current economic slow down** and made the forecast with lower total purchase amounts.


# Conclusions: 
The goals of this project were:
- Explore the historical data of the DIR contracts with Cisco. 
- Create a model for the sales forecast of Cisco with public agencies.

*Result:*
We made a data exploration and created the regression model that can forecast the sales based on the date features only.

## **Recommendations and next steps:**
- split data on 5 different sets based on the customer type and mae predictions for every set.
