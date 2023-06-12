# Technology and Golf

This repository contains all files and ipython notebooks used in this project. A full outline of all the files with descriptions can be found below.

To view the Slide Deck, ["click here."](https://www.canva.com/design/DAFlL_rZscs/vmOX1jMR1lOC3iLS-5c6MA/edit?utm_content=DAFlL_rZscs&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) 

To view the Final Report, ["click here."](https://github.com/Birdie-Eye-View/codeup-capstone/blob/main/mvp/final_report.ipynb)

To view the PGA Stats, ["click here."](https://www.pgatour.com/stats)

To view datasets:

Set 1, ["click here."]([https://www.pgatour.com/stats](https://drive.google.com/uc?export=download&id=1T5zczOftHU3BXFXFb8czAvpN8uWmBp0C))

Set 2, ["click here."]([https://www.pgatour.com/stats](https://drive.google.com/uc?export=download&id=118ULHKdAmNeuvc_1EKPP24D9DW3W-5bg))

Set 3, ["click here."]([https://www.pgatour.com/stats](https://drive.google.com/uc?export=download&id=1u5AxqlhZxuCIgq7x8GSJt0S5eKQZzLi_))

___
## Table of Contents

- [Technology and Golf](#technology-and-golf)
  - [Table of Contents](#table-of-contents)
  - [Project Summary](#project-summary)
  - [Project Planning](#project-planning)
    - [Project Goals](#project-goals)
    - [Project Description](#project-description)
    - [Initial Questions](#initial-questions)
  - [Data Dictionary](#data-dictionary)
  - [Outline of Project Plan](#outline-of-project-plan)
    - [Data Acquisition](#data-acquisition)
    - [Data Preparation](#data-preparation)
    - [Exploratory Analysis](#exploratory-analysis)
    - [Modeling](#modeling)
    - [Deliverables](#deliverables)
  - [Lessons Learned](#lessons-learned)
  - [Instructions For Recreating This Project](#instructions-for-recreating-this-project)

___
## Project Summary

The USGA is proposing a rule change to the game of golf. They are suggesting that technology will eventually allow the average pga tour player to exceed the current driving distance regulation of 317 yards. By using time series analysis to make predictions, our project aims to investigate if the average tour golfer’s driving distance will exceed the current standard, when we could expect this to occur, and how it could impact professional golf. 
___
## Project Planning

1. Will the drive average meet or exceed 317 yards by Jan 1, 2026?
2. How has the average driving distance increased over time?
3. How has par 4 and par 5 scoring changed over time?
4. Have changes to golf clubs influenced driving distance?
5. Have changes to golf balls influenced driving distance?
___
## Data Dictionary
|      Field 		      |        Data type 		   |				       Description				            |
|---------------------|------------------------|----------------------------------------------|
| age_x               |                   int64| Player's current age           				      |
| age_y               |                 float64| Player's predicted age when they reach 317yds|
| age_y_bin           |                category| Predicted ages sorted by 20-29,30-39,40-49   |
| birthplace          |                  object| Birth city and country of player             |
| dob                 |                  object| Player's date of birth                       |
| drive_avg           |                 float64| Average drive length in yards                |
| height              |                  object| Players height                               |
| par_4_avg           |                 float64| Avg par 4 score                 		          |
| par_5_avg           |                 float64| Avg par 5 score                              |
| player              |                  object| Name of golfer                               |
| predicted_years     |                 float64| Predicted years count to reach drive avg 317 |
| predicted_years_bin |                category| Predicted years sorted by 20-29,30-39,40-49  |
| weight              |                  object| Players weight                               |
| year                |                   int64| Year index for player stats                  |


## Outline of Project Plan

Gather data on PGA tour golfers, prepare and clean the data, explore and visualize to answer key questions. Use time series modeling to predict when the drive average will reach 317 yards and then summarize our finding.

--- 
### Data Acquisition

Data was aquired by webscraping the PGA tour website. We looped through every year available, 1987 to 2023 scraping data from three different statistic sources on this website; driving distance, average par 4 score, and average par 5 score.

### Data Preparation

After scraping these webpages, we extracted the desired data into thier 3 respective dataframes. We then merged them together into one, stratifying on the year and player feature to promote data integrity. We pulled random samples and audited the data to ensure it merged correctly without errors. We removed any nonsensical or rundundant columns caused by the merge, and renamed columns to promote readability

### Exploratory Analysis

Simply put, we found the regression analysis is telling us that there is a relationship between driving distance and scoring average: as driving distance increases, scoring average tends to decrease. However, the relationship is not very strong, and driving distance only explains a small portion of the variation in scoring average. It's important to keep in mind we are looking at the complete tour average. It is possible that analyzing the top percentage of players could yeild much different results compared to the entirety of the tour.

### Modeling

Our model's performance was significantly enhanced, as evidenced by the reduction in RMSE from the baseline of 3.51 to 2.27 on the test data. This improvement of approximately 35.33% demonstrates a substantial increase in prediction accuracy and suggests that our model is able to capture the underlying patterns and variability within the data with confidence.

### Deliverables

We predict the drive average will reach 317 yards years after the USGA's predicted Jan 2026 date. We understand that decisions to "roll back" the golf ball may be necessary to ensure that the most prestigous and tradition-important courses remain relevant and remain a test to golfing elites. However, we strongly recommend the USGA consider not making rule changes based on outlier holes or golfers. Even among the elite, skill gaps can and should be present, and we recommend further research to determine if this rule change affects players of all driving distances and not just those at the top.

## Lessons Learned

Take the time and detail to really make sure you prepare and clean up your dataset. It makes going into exploration and visualizations much easier and understandable. 
Make a detailed plan before begging so you can know exactly how you are progressing. Answer your initial questions first before exploring other avenues of the data 
it is easy to get lost in exploration. 
___
## Instructions For Recreating This Project

All needed datasets are hosted and linked in the notebook/final.py. Clone down this repository and open it in jupyter notebook or jupyter lab to run the code. 

