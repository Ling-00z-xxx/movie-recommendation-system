# movie-recommendation-system
A simple personalized movie recommendation baseline built with Python and the MovieLens 32M dataset.

## Project Overview

This project explores how user rating history can be transformed into a personalized movie recommendation pipeline.

Instead of implementing a complex deep learning recommender, I built an interpretable baseline that combines:

- User preference profiling
- Genre-based content matching
- Movie rating statistics
- Weighted rating
- Recommendation ranking

## Dataset

The project uses the MovieLens 32M dataset from GroupLens.

The dataset contains:

- 32M+ ratings
- 87K+ movies
- 200K+ users

The raw dataset is not included in this repository due to its size.

## Recommendation Pipeline

```text
User Rating History
        ↓
Highly-Rated Movies
        ↓
Genre Preference Profiling
        ↓
Filter Watched Movies
        ↓
Genre Matching
        ↓
Movie Quality & Rating Reliability
        ↓
Recommendation Score
        ↓
Top-N Recommendations
