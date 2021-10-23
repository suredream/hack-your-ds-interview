 

 

Case-study: Explore the use of machine learning using remote sensing to map off-season field-level corn yield in a certain region (somewhere in the US), the candidate has the freedom to select certain data sources (public or commercial if they have experience) and design a poc modeling approach. 

 

Interview Questions 

Aim to map field-level corn yield at regional scale, how to collect the reference data?  

 

Question 1 answer rubric: 

Poor answer:  
 

Only state data collection approaches generally (google search, kaggle, government, open-source datasets, asking stakeholder, field campaign) but without thinking on the business need, data quality, standards.  

 

Good answer:  

Quick started with 1-2 authoritative data sources. Think over the angles (quality, accuracy, coverage, cost) to assess the datasets before grab them in. Provide some ways (distribution, frequency, grouping) to check the consistency among different sources. 

 

Great answer:  

Besides general mindset of data understanding, demonstrate domain knowledge to related datasets as well (e.g. USDA NASS/ARS catalog, soil & weather datasets in geoscience domain). Also provide some ideas about how to carry out the EDA to the data collected, outliers detection, temporal anomaly, etc. 

 

How to design the sampling scheme to balance the target data further for EO data collection? 

Question 2 answer rubric: 

Poor answer:  Lack of concepts/knowledge to apply sample/resample in the context of statistical/geospatial context. Knowing the limitation from the reality (scarcity of good data and costly of ground truth collection) but don’t give too much think about how to overcome / improve the situation. 

 

Good answer:  Be able to state the working hypothesis of geospatial sampling – the goal of generate good representative of the given area. Be able to carry out a meaningful sampling procedure. E.g, a stratified + random/grid sampling or similar approach is brought-in to capture that information. Any more detail is a plus (based on what zone/group information the stratified was carried out, how to determine the minimum number within each group. Balanced sampling afterwards is another plus 

 

Great answer: Candidates show attention / deeper understanding to how to compare the sample and the population and test the statistically significant to represent the whole population. A bigger plus would be that candidates demonstrate his reasoning on sampling framework is aligned with the model selection as well. 

  

 

To kick-off a MVP model, how will you select the remote sensing data sources, the bands combination, auxiliary dataset plus ARD processing ? 

 

Question 3 answer score points: 

Poor answer: 

Can provide technique practices but no good reasoning why doing that, or how to do the trade-off according to the processing and modeling decision. 

 

Score points:   

Public satellite imagery (S1/S2, Landsat 8), RGBN bands 

Commercial dataset ( if he/she is familiar with any commercial,  ask what’s the benefit of using them, and how to do the necessary pre-processing (e.g. usually AC didn’t comes with commercial datasets and will be costly). 

Meteorology data (precipitation, temperature, soil moisture, humidity) 

Field data (boundary, planting date, harvest data) 

Preprocessing about the cloudy imager (cloud masking, composites) 

Spatial aggregation (field-level, mean, median, outlier detection) 

Temporal accumulation (how to select time periods, moving windows, weighted accumulate, temporal smooth) of the datasets is needed (align satellite with meteorology) 

Alignment RS time-series, meteorology data with other features in a standard way. 

Operations to take care of missing values (filling-average, forward-filling, backward-filling, advanced smoothing techniques) 

 

Say more about how to development this MVP yield mapping model using your ML skills 

 

Question 4 answer score points: 

Poor answer:  Only general understanding of the ML techniques or hands-on experience, no critical reasoning. 

 

Score points:  

Show understanding of baseline MVP model  

Simple model for quick start 

Wide used, recognized strength and weakness (Tree-based regression or statistics model) 

Not delivery to the end user but interpretable step to test the working assumption 

How to do train/test split and why 

How to do feature engineering and selection 

How to carry out cross-validation and why 

How to validate/evaluate the mode results 

From the preliminary results, how to test where the bottleneck is from (data distribution or data quality, feature selection) 

 

How to communicate your model across teams 

 

Question 5 answer score points: 

Poor answer:  Only focus on reporting model metrics/accuracy without be aware of high-level communication; lack of the working experience from business and engineering perspective. 

 

Score points:  

Besides model performance, also care about facts of resources cost, latency as well. Have a mindset to trade-off these factors (accuracy / cost / latency) to fulfill needs. 

Be able to communicate the limitation of current MVP model (data size, processing-level, model assumption) 

Be able to communicate the engineering challenging of current pipeline (scalability, data availability) 

If limitation and nice-to-do has been identified, be able to communicate the way you prioritize these items for the further model development. 

 

Reference 

Modeling Science Team technique interview guide prior to inviting them onsite.  

https://gistbok.ucgis.org/bok-topics/evolution-geospatial-reasoning-analytics-and-modeling 