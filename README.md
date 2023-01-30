
# Project Overview:
This team was tasked to build a model that can predict the sales for Cisco.

# Project Goal
The goal of the project is to make the exploration of Cisco's sales with public agencies, define what features can be useful to create a prediction model. Build a model that can forecast future sales for the company. 

# Project Deliverables:
* Produce a Final GitHub repository containing our work
* Provide a well-documented jupyter notebook that contains our analysis
* Create presentation with our findings 
* Create a model that will forecast the sales for Cisco

# Project Description and Intitial Thoughts.
In this project we will focus on Time Series Analysis and we'll try to create forecast model that can predict future sales. We expect to hit difficulties during modeling process, as the beginning of pandemic led to anomaly increase of telecommunication sales followed by significant drop in sales after it. That anomaly will "confuse" the models making them over- or underestimate sales dramatically depending of the time period included to train the model.

# Executive Summary
**The goal** 
The goal of the project is to make the exploration of Cisco's sales with public agencies, define what features can be useful to create a prediction model. Build a model that can forecast future sales for the company. 
**Key takeaways**

* __Acquire__ 
The data file contains information about DIR Cooperative Contract Sales. To get the information about Cisco we filtered the data by vendor name.

* __Prepare__ 
    - Removed columns with nulls only. 
    - Added features based on the date information.
    - Created an additional data frame that summarized information about sales for every customer by the end of the day.

* __Explore__
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
    - Our best model beats a baseline on all train, validation and test sets. Anyway, there is much more work to do. We'd like to improve results by making weekly/monthly resamples hoping find some seasonality. We'd like to remove the some pandemic sales anomalies as well. If this won't improve model performance, our next step is going to be splitting data on 5 different sets based on the customer type and make predictions for every set.

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

    * **Question 1.** Is there any significant difference in sales by the day of the week?
    
    * **Question 2.** Is there any significant difference in sales by the month?

    * **Question 3.** Is there any significant difference in sales by the quarter?

    * **Question 4.** Is there any significant change in monthly sales percentage in our data?
    
    * **Question 5.** Any other surprises during pandemic beside the sales increase?



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


# Acquire
We acquired the data from the [Texas Open Data Portal](https://data.texas.gov/dataset/OFFICIAL-DIR-Cooperative-Contract-Sales-Data-Fisca/w64c-ndf7). The origin of the data is 	[Texas Department of Information Resources](	https://dir.texas.gov/). The data file contains information about DIR Cooperative Contract Sales. To get the information about Cisco we filtered the data by vendor name.

# Prepare
* Dropped the columns with nulls only
* Renamed the columns: replaced whitespaces with underscores and made them lower case
* Created additional features based on date: `month_name`, `day_name`, `year`, `quarter`, `month`, `week`, `day_of_week`, `day_of_year`

# Exploration Findings:
* **Question 1.** Is there any significant difference in sales by the day of the week?
    - All sales per day of the week significantly differ from the average daily sales. 
    - Week days have higher sales than the average and weekend days almost don't have sales. 
    - Monday, Tuesaday and Wednesday seem to have same average amount of sales. 
    - Friday has the higher average sales amount among all days of the week.
* **Question 2.** Is there any significant difference in sales by the month?
    - The highest sales are happening in July, followed by April, October and June.
    - The lowest sales happen on February.
    - March has almost the same results as average sales, but there was peak in sales on March, 2020 when pandemic just started, it might happened that March is typically low on sales.
    - April results might be affected by beginning of pandemic, too.
* **Question 3.** Is there any significant difference in sales by the quarter?
    - Highest sales are in the 3rd quarter.
    - Lowest sales are in the 1st quarter.
    - Average qurterly sales per each quarter are not significantly different from the overall average quarterly sales.
* **Question 4.** Is there any significant change in monthly sales percentage in our data?
    - In the beginning of the pandemic there was an an abnormal spike in sales.
* **Question 5.** Any other surprises during pandemic beside the sales increase?
    - There in a big outlier in order_quantity. Right before the pandemic one agency places the order for 4M Cisco's products.

# Modeling

### Features that will be selected for Modeling:
We kept only order date and purchase amount for our ARIMA models. To create regression model we also kept features that are based on the date: day of week, week number, month, year.

**The models we created**
We created ARIMA statistical models and XGBoost Regressor. XGBoost Regressor outperformed all statistical models.


## Modeling Summary:
 Our best model beats a baseline on all train, validation and test sets. Anyway, there is much more work to do. We'd like to improve results by making weekly/monthly resamples hoping find some seasonality. We'd like to remove the some pandemic sales anomalies as well. If this won't improve model performance, our next step is going to be splitting data on 5 different sets based on the customer type and make predictions for every set.


# Conclusions: 
The goals of this project were:
- Explore the historical data of the DIR contracts with Cisco. 
- Create a model for the sales forecast of Cisco with public agencies.

*Result:*
We made a data exploration and created the regression model that can forecast the sales based on the date features only.

## **Recommendations and next steps:**
- create weekly/monthly resamples hoping find some seasonality and trends
- remove the some pandemic sales anomalies as well

or, as final step
- split data on 5 different sets based on the customer type and mae predictions for every set.
