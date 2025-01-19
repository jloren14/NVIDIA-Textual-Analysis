# Reasons for NVIDIA's stock price increase using textual analysis on Business News
The repository contains the data and R script with the solution to the analysis I was given during the course "Applied Textual Data Analysis" at NHH Norwegian School of Economics. The assignment was a final project for the course and it was completed in a group of three students: Julia Lorenc, Giuseppe Pio Lando and Claudia dal Prà. 

## Table of contents
* [Project Overview](#project-overview)
* [Data](#data)
* [Analysis](#analysis)
* [Dependencies](#dependencies)

## Project Overview
The stock price of NVIDIA rose significantly in the period from 2016 to 2024. The goal of the project is to analyze this stock price increase using textual analysis on Business News.

## Data
The [raw data](https://drive.google.com/drive/folders/1fUOkbKAl7jnrN8nVlFVpKJT3iQYFIlkh?usp=sharing) consists of closed captions of all shows on Bloomberg, a main business news channel, from 2016 to 2024.

## Analysis
Our analysis has the following steps:
1. Data preprocessing
     * Data cleaning and text tokenizing for 2016-2024 files: removing titles, timestamps, stopwords, and non-alphabetic characters, lemmatizing the text
     * Combining all yearly datasets into a single corpus
  
2. Constructing the corpus
     * Filtering articles mentioning "NVIDIA" using keyword matching with `grepl`
     * Building a KWIC (Key Word in Context) analysis for "NVIDIA"
     * Processing the resulting context by cleaning text further and generating bigrams
  
3. Competition analysis
     * Selecting a subset of news articles for each day based on daily article counts
     * Training Word2Vec models for each year to analyze terms most closely associated with "NVIDIA"
  
4. Sentiment analysis
     * Computing sentiment scores using the Loughran-McDonald lexicon for articles mentioning "NVIDIA"
     * Visualizing relationships between sentiment, news count, and stock prices over time
     * Conducting linear regression to analyze the relationship between stock returns and sentiment/news count
  
5. MNIR model
     * Merging bigram data with stock prices, filtering out NA values (weekends and holidays)
     * Training MNIR models to identify bigrams most associated with high or low NVIDIA stock prices
     * Extracting and ranking bigrams by their association with stock prices
  
6. Topic modelling
     * Building the corpus, using the unigrams and constructing the Document Term Matrix (DTM)
     * Training an LDA model with 30 topics using Gibbs sampling
     * Identifying topics and their top terms, associating them with specific themes (e.g., AI, gaming, competition)
     * Generating word clouds for selected topics
  
7. Vizualization

## Dependencies
The project requires the following R packages:
* dplyr
* tidyverse
* stringr
* tokenizers
* stopwords
* tm
* tidytext
* slam
* proxy
* topicmodels
* wordcloud
* textclean
* word2vec
* xtable
* parallel
* sentimentr
* tidyr
* quanteda
* textstem
* NLP
* SnowballC
* data.table
* readr
* textir
   
