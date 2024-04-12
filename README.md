# Prediction of Health Measures Based on Location

**Data Scientist:** Elina Rankova

[IMAGE PLACEHOLDER]

## Business Problem and Understanding

**Stakeholders:** Company executives specializing in operational expansion to improve public health prevention where it is most needed.

As technology evolves, public health and healthcare in particular has long been been lagging behind. In recent years, more companies are leaning on data and technology to inform growth opportunities with the goal of helping those most in need.

As an independent contractor specializing in expanding resource availability to undeserved populations, this project aims to identify overall health disparity across the United States utilizing public county and census data provided by the CDC. Proving population health measure specifics based on an overall Health Disparity Index can help direct those striving to close care gaps to expand where there is the greatest need.

For Phase 1 of this initiative we will define a Health Disparity Index (HDI) based on a number of health measures and population by geolocation.

**The goal:** Provide clients with a holistic understanding of the measures contributing to an overall Health Disparity Index aggregated by geolocation to better understand what United States regions most provide opportunity for growth.

## Data Understanding

For this task we will be using CDC provided PLACES and SDOH county data spanning years 2017-2021
Datasets
- <a href="https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/dv4u-3x3q/about_data">Local Data for Better Health, County Data 2020 release</a>
- <a href="https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/pqpp-u99h/about_data">Local Data for Better Health, County Data 2021 release</a>
- <a href="https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/duw2-7jbt/about_data">Local Data for Better Health, County Data 2022 release</a>
- <a href="https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/swc5-untb/about_data">Local Data for Better Health, County Data 2023 release</a>
- <a href="https://data.cdc.gov/500-Cities-Places/SDOH-Measures-for-County-ACS-2017-2021/i6u4-y3g4/about_data">SDOH Measures for County, ACS 2017-2021</a>

Utilizing all the health measures defined within the cagtegories specified below we will create a custom Health Disparity Index (HDI) as our target for this regression task.

**Categories within which health measures are defined:** Health Outcomes, Prevention, Health Risk Behaviors, Health Status, Disabilities, and Social Determinants of Health
