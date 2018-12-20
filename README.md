# Recommendation Engine with IBM Watson
This project was concieved in collaboration with Udacity and IBM Watson

## Overview
In this project I aim to create a recommendation system for the IBM Watson community. The recommendation system
suggests articles for users to interact.

### Features
The recommender system can make recommendations in a number of ways:
1. Collaborative Filtering
- Takes into account the similarity of users and recommends the most popular articles read by similar users
2. Rank Based Recommendations
- Recommends the highest ranked articles starting with the most highly ranked 
3. (IN PRODUCTION) Content Based Filtering
- Produces recommendations based on similarity to material the user has interacted with previously. Utilizes Natural Language Processing (NLP) methodology to analyse and rank articles by similarity.
4. SVD - Matrix Factorization Recommendations
- Utilises matrix operations to predict the ranking (or in this case the boolean interaction variable)  

## Usage
[See Deployed Web App](https://recommendation-eng.herokuapp.com/)
- Collaborative Filtering
- Rank based recommendations
- dataset visualisation
