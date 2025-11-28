# Preprocessing

In this step, we preprocess the data in following steps

## Feature Filtering

We use BorutaPy package to filter the features. **30 out of 2547** featrues were selected using boruta filtering using RandomForest as an estimator. For all parameters of boruta, default values were used.

## Feature Ranking

1. Minimum Redundancy Maximum Relevance

  To rank features by maximum relevance and minimum redundancy we applied mRMR feature ranking. Which uses F-test for relevance measurement, Pearson correlation for redundancy and returns features ordered by importance.

2. MCFS Feature Ranking

Then we tried to rank the features using Monte Carlo ensemble approach.

When comparing mRMR vs MCFS, there was not much agreement betwee two methods as shown in the scatterplot.

![alt text](../images/feature_ranking_mRMR_MCFS.png)


3. ..