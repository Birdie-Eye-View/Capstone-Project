# Technology and Golf

This repository contains all files and ipython notebooks used in this project. A full outline of all the files with descriptions can be found below.

To view the Slide Deck, ["click here."] 

To view the Final Report, ["click here."]

To view the Dashboard, ["click here."]

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
___
## Project Planning
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
---
### Data Acquisition

Data was aquired by webscraping the PGA tour website. We looped through every year available, 1987 to 2023 scraping data from three different statistic sources on this website; driving distance, average par 4 score, and average par 5 score.

### Data Preparation

After scraping these webpages, we extracted the desired data into thier 3 respective dataframes. We then merged them together into one, stratifying on the year and player feature to promote data integrity. We pulled random samples and audited the data to ensure it merged correctly without errors. We removed any nonsensical or rundundant columns caused by the merge, and renamed columns to promote readability

### Exploratory Analysis

### Modeling

### Deliverables

## Lessons Learned
___
## Instructions For Recreating This Project

