# Modeling-NYC-FHV-Uber-Lyft-Trip-Data-2022-2023 

### Description of the Data Set ###
I propose to analyze and model the "NYC FHV (Uber/Lyft) Trip Data Expanded (01/2022-06/2023)" dataset, only a short period of time frame was picked because it’d take a long time to process if it were to be analyzed through databricks. This dataset contains comprehensive information about every For-Hire Vehicle (FHV) trip in New York City spanning the years 2022 to 2023. The dataset includes attributes such as:

 0   hvfhs_license_num     object        
 1   dispatching_base_num  object        
 2   originating_base_num  object        
 3   request_datetime      datetime64[ns]
 4   on_scene_datetime     datetime64[ns]
 5   pickup_datetime       datetime64[ns]
 6   dropoff_datetime      datetime64[ns]
 7   PULocationID          int64         
 8   DOLocationID          int64         
 9   trip_miles            float64       
 10  trip_time             int64         
 11  base_passenger_fare   float64       
 12  tolls                 float64       
 13  bcf                   float64       
 14  sales_tax             float64       
 15  congestion_surcharge  float64       
 16  airport_fee           float64       
 17  tips                  float64       
 18  driver_pay            float64       
 19  shared_request_flag   object        
 20  shared_match_flag     object        
 21  access_a_ride_flag    object        
 22  wav_request_flag      object        
 23  wav_match_flag        object        


The dataset provides information about the total miles for the passenger trip, the total time in seconds for the passenger trip, the base passenger fare before tolls, tips, taxes, and fees, the total amount of all tolls paid in the trip, the total amount collected in the trip for the Black Car Fund, the total amount collected in the trip for NYS sales tax, the total amount collected in the trip for NYS congestion surcharge, and the airport fee of $2.50 for both drop off and pick up at LaGuardia, Newark, and John F. Kennedy airports.
Moreover, the dataset includes the total amount of tips received from the passenger, the total driver pay (not including tolls or tips and net of commission, surcharges, or taxes), the flag indicating whether the passenger agreed to a shared/pooled ride and whether the passenger shared the vehicle with another passenger who booked separately at any point during the trip.

### Intended Modeling Objective ###

My primary objective is to build a predictive model that can forecast demand for For-Hire Vehicles in different areas of New York City. This model will serve multiple purposes:

1) Demand Forecasting: We will use historical trip data to predict the tips received for FHV services in specific boroughs and neighborhoods at different times of the day and week.

2) Supply Optimization: By understanding demand patterns, we aim to help FHV service providers (e.g Uber, Lyft) optimize their supply of vehicles in high-demand areas during peak hours, reducing the “risk” of losing tips in the case where there are traffic jams, etc. 

3) Pricing Strategy: The model will enable FHV companies to adjust pricing dynamically based on demand, improving both passenger experience and driver earnings.

4) Traffic and Route Optimization: Analyzing trip data will provide insights into traffic patterns and optimal routes, allowing drivers to minimize trip durations.

By modeling NYC FHV trip data, I aim to support the efficient operation of For-Hire Vehicle services in New York City, benefiting both passengers and service providers. This project aligns with the goal of improving urban mobility and transportation services in the city while enhancing the profitability of FHV companies.

*** URL/Location for downloading the data: ***
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
