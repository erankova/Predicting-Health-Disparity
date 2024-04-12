# Prediction of Health Measures Based on Location

**Data Scientist:** Elina Rankova

<p align="center">
  <img src="https://media.licdn.com/dms/image/C4D12AQE3ln6Z4Mo8Pw/article-cover_image-shrink_720_1280/0/1642087174753?e=1718236800&v=beta&t=etvT9WV2qS3g7hAnCds-vQIn3ob9JqJn9vvB9annTWM">
</p>
<a href="https://www.linkedin.com/pulse/benefits-population-health-management-phm-amazing-charts/">Image Source</a>

## Business Problem and Understanding

**Stakeholders:** Company executives specializing in operational expansion to improve public health prevention where it is most needed.

As technology evolves, public health and healthcare in particular has long been been lagging behind. In recent years, more companies are leaning on data and technology to inform growth opportunities with the goal of helping those most in need.

As an independent contractor specializing in expanding resource availability to undeserved populations, this project aims to identify overall health disparity across the United States utilizing public county and census data provided by the CDC. Proving population health measure specifics based on an overall Health Disparity Index can help direct those striving to close care gaps to expand where there is the greatest need.

For Phase 1 of this initiative we will define a Health Disparity Index (HDI) based on a number of health measures and population by geolocation.

**The goal:** Provide clients with a holistic understanding of the measures contributing to an overall Health Disparity Index aggregated by geolocation to better understand what United States regions most provide opportunity for growth.

## Data Understanding

For this task we will be using CDC provided PLACES and SDOH county data spanning years 2017-2021. Our data set contains a total of 28287 rows from the SDOH data and 780890 from the PLACES datasets.

**Datasets**
- <a href="https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/dv4u-3x3q/about_data">Local Data for Better Health, County Data 2020 release</a>
- <a href="https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/pqpp-u99h/about_data">Local Data for Better Health, County Data 2021 release</a>
- <a href="https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/duw2-7jbt/about_data">Local Data for Better Health, County Data 2022 release</a>
- <a href="https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/swc5-untb/about_data">Local Data for Better Health, County Data 2023 release</a>
- <a href="https://data.cdc.gov/500-Cities-Places/SDOH-Measures-for-County-ACS-2017-2021/i6u4-y3g4/about_data">SDOH Measures for County, ACS 2017-2021</a>

Utilizing all the health measures defined within the cagtegories specified below we will create a custom Health Disparity Index (HDI) as our target for this regression task.

**Categories within which health measures are defined:** Health Outcomes, Prevention, Health Risk Behaviors, Health Status, Disabilities, and Social Determinants of Health

### Feature Exploration

Right away we can tell that there are a few columns that can be dropped
- Missing too many values
  - `Data_Value_Footnote_Symbol`
  - `Data_Value_Footnote`
  - `Latitude`
  - `MOE`
- `Low_Confidence_Limit` and `High_Confidence_Limit` - the SDOH dataset does not have these columns.

> `Geolocatioin` and `Geolocation` are the same feature when checking the source data websites only the 2020 release has a `Geolocatioin` column while the rest have `Geolocation`. `MeasureID` and `MeasureId` have the same issue it seems. These are combined and the misspelled column is dropped.

Transformations
- Object type and Categorical columns that will need to be transformed
- Since state should be treated as a categorical variable we should adjust this in our dataset
- `LocationID` should be an object since this is not an actual continuous variable and is currently an integer

**Additional observations**
- `Year` can be dropped as it can only provide relevant information for the data we retrieved from the PLACES datasets. Otherwise, we have the same date range as seen in the added SDOH data (2017-2021).
- Rows with US as the state are dropped as it is an error. `StateDesc` is dropped since it's the same info as `StateAbbr`.
- `DataSource` is dropped since all of the information is coming from the Behavioral Risk Factor Surveillance System or the 5-year American Community Survey which only contains SDOH data.
- `Data_Value_Unit` is dropped since all of our data values are in percentages.
- `Data_Value_Type` will be helpful for interpretation of our predictions since it denotes what percentage the `Data_Value` represents. `DataValueTypeID` corresponds to this feature is dropped since `Data_Value_Type` is more informative. 
- `CategoryID` corresponds to `Category` so it is dropped since `Category` is easier to interpret.
- `MeasureId` corresponds the same way to `Measure`. However `Measure` values can be quite lengthy depending on the measure. We also have `Short_Question_Text` corresponding to these features and is more informative than `MeausureId` but shorter than `Measure` so `Short_Question_Text` is kept and create a reference dictionary before dropping the other columns.

### Statistical Analysis

Before moving onto defining the target, the numerical distribution of the `Data_Value` and `TotalPopulation` columns is evaluated. There are some outliers in both columns we should consider visualizing and potentially dropping to get a more accurate representation of our distributions.

<div align="center">

| Statistic | Data_Value  | TotalPopulation |
|-----------|-------------|-----------------|
| count     | 808694.000  | 808694.000      |
| mean      | 30.404      | 103356.9        |
| std       | 24.767      | 331892.8        |
| min       | 0.000       | 57.0            |
| 25%       | 10.700      | 10806.0         |
| 50%       | 22.100      | 25629.0         |
| 75%       | 40.200      | 67490.0         |
| max       | 99.400      | 10105520.0      |

</div>
``


We should also normalizing the `Data_Value` column since the values are represented differently based on `Data_Value_Type`. We can also consider weighing it if we consider one type more important than another.

Visualizing these features, we can estimate that any `Data_Value` above ~85 is an outlier. It is harder to determine for `TotalPopulation`. We should look at the values of these more closely.

<p align="center">
  <img src="https://github.com/erankova/Capstone/assets/155934070/ff7494f9-5c48-4870-b4ff-7ef2acfee85d" alt="Boxplot">
</p>

Taking closer at the number of rows that are outside of the the IQR, there are quite a few rows with population and data value outside of bounds.

<div align="center">

| Description        | Count   |
|--------------------|---------|
| TotalPopulation    | 111,425 |
| Data_Value         | 15,851  |

</div>



As expected, the `Data_Value_Type` impacts the `Data_Value` outliers

<div align="center">

| Metric               | Outliers |
|----------------------|----------|
| Crude Prevalence     | 2,117    |
| Percentage           | 610      |
| **Total Outliers**   | **5,661**|

</div>



> Population doesn't demonstrate the same diffence so we can treat TotalPopulation collectively in the dataset.

The Data Value distribution for each Data Value Type isn't normal. So before we move on using it to create the HDI we want to create a new column with the normalize values using the `RobustScaler` since we have been using IQR to identify outliers.

<p align="center">
  <img src="https://github.com/erankova/Capstone/assets/155934070/70bf866a-ceca-4ed0-a1ea-81a8c88d6f2e" alt="Data Type Distribution">
</p>

### Defining Health Disparity Index

After the dataframe is aligned with project goals, the target variable `Health_Disparity_Index` can be defined. 

To start, we will create a general index taking the sum of all scaled values for each location and create a feature in the full dataset to represent this disparity index. 

Comparing random samples of `Sum_Idx` vs `Data_Value` shows us that there is a lot more uniformity in the `Scaled_Value` before aggregation of all of the data value types. This makes sense since our aggregated `Sum_Idx` is combining all values per geolocation, creating and therefore creates less variability when visualizing by location.

<p align="center">
  <img src="https://github.com/erankova/Capstone/assets/155934070/3059cc02-31d5-436f-8204-a53dc4f85d29" alt="Sum_Idx Distribution">
</p>

**note:** scale value error bars represent the variablility around the median, showcasing the 75th percentile at the top and 25th percentile towards the lower end of the bars.

When we create an `Avg_Idx` we now see our geolocation distribution but on a different scale and with more goeographical distinction. 

> This would make sense if for example, this segregation is representing rural vs urban areas.

<p align="center">
  <img src="https://github.com/erankova/Capstone/assets/155934070/1aa2c1be-3c53-417d-8189-59856f2776ba" alt="Avg_Idx Distribution">
</p>

Lastly we will add a population weight to each scaled value, this will adjust our results to consider the density of a region when calculating the HDI. The weighted average provides more of a spread than the `Avg_Idx, and the same relationship as the `Avg_Idx` which means population definitely has an impact on the HDI and geolocation does as well.

<p align="center">
  <img src="https://github.com/erankova/Capstone/assets/155934070/0044d1c9-ec37-4fd0-b6d9-cee8a282fe65" alt="Weighted_Idx Distribution">
</p>

### Geolocation to HDI Visualization

Before finalizing the decision to use the population weighted HDI, we visualize the index in comparison to population using the `Geolocation`. To do so, we first convert the feature to wkt (well-known text) format for use with the geopandas library.

Looking at population and the HDI across all of the geolocations we can see that the higher the population, the lower the HDI. In particular we can see that in part of the west coast, Florida, and part of the east coast demonstrate this inverse relationship.

> This inverse relationship may make sense when you consider factors such as but not limited to; greater access to care, diversity of services, and public health funding in areas with more population density.

<p align="center">
  <img src="https://github.com/erankova/Capstone/assets/155934070/b20c4afb-fca4-4c6e-9bec-9793010acefb" alt="Population by Geolocation">
</p>

This further confirms our decision to use population as a weight for our HDI definition.

## Data Preparation

Since our dataset is so large we will first train our model on a smaller training set to save computational power and time. Columns we used to create our `Weighted_Idx` are dropped. 

> In future iterations would want to consider using `Geolocation` to create location features and explore spatial relationships. However, to keep computational cost relatively low, we will note this as a Phase 2 addition. 

`StratifiedShuffleSplit` is used to create subsets of our data while maintaining proportions of the target variable. Running a subset of the data will give us a way to understand model performance while maintaining computational efficiency. Once best fitting model is identified, we will run it on our extra hold out data that wasn't used for training to further validate our results.

### Base Model and Metrics

`StandardScaler` is used on all our numerical data to make sure we are comparing our features on the same scale. And `OneHotEncoder` is used on the rest of the object or categorical categories. 

Our base model will be a standard `LinearRegression`.

<div align="center">

| Metric        | Train Value         | Test Value         |
|---------------|---------------------|--------------------|
| RMSE          | 0.036465476         | 0.036466192        |
| R-Squared     | 0.678842896         | 0.678760292        |
| MAE           | 0.032773953         | 0.032738450        |

</div>
It looks like the primary RMSE metric is pretty good. However, it is expected that it would be small considering the range of our HDI so we will aim to improve these further. $R^2$ can also stand to improve. Thankfully, we don't see much overfitting or underfitting.

We will keep these metrics in mind as we test other models.

> Even after dropping a significant amount of features, we are left with fairly high dimentionality with 4425 features. This is something we will want to consider as we test and refine our models.

## Modeling

With our base metrics defined, we can finetune the base and try other models to improve our metrics. 

We will start by testing different regularization techniques to address our moderate $R^2$ and improve our error metrics in this high dimensional space.

### <div align="center">Lasso</div>

<div align="center">

| Model Name | Val Train Score | Val Test Score |
|------------|-----------------|----------------|
| Lasso      | 0.047102        | 0.047102       |

</div>

<div align="center">

| Model Name | Train Score | Test Score | Train R2  | Test R2   | Train MAE | Test MAE  |
|------------|-------------|------------|-----------|-----------|-----------|-----------|
| Lasso      | 0.047084    | 0.047078   | 0.464568  | 0.464604  | 0.039646  | 0.039654  |

</div>
<div align="center">Looks like our Lasso model did slightly worse in error metrics and significantly worse in $R^2$. </div>

### <div align="center">Ridge</div>
























