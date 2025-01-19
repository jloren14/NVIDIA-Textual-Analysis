library(dplyr)
library(stringr)
library(tokenizers)
library(stopwords)
library(tm)
library(tidytext)
library(slam)
library(proxy)
library(topicmodels)
library(wordcloud)
library(textclean)
library(word2vec)
library(xtable)
library(tidyverse)
library(parallel)
library(sentimentr)
library(tidyr)
library(quanteda)
library(textstem)  
library(NLP)
library(SnowballC)
library(data.table)
library(readr)
library(textir)

setwd("/Users/giuseppelando/Desktop/Final submission")
base_dir <- "/Users/giuseppelando/Desktop/Final submission/raw_TV_data_2016_to_2024/"

# Loop through each year from 2016 to 2024
for (year in 2016:2024) {
  # Construct the file path for the current year
  file_path <- paste0(base_dir, "Bloomberg.Text.", year, ".1.csv")
  
  # Read the CSV file for the current year
  news_data <- read.csv(file_path, header = FALSE, col.names = c("ID", "date", "text"))
  
  # Extract and format the date from the ID column
  news_data$date <- sapply(news_data$ID, function(id) {
    date_str <- str_extract(id, "[[:digit:]]{8}") # Extract the first 8 digits
    format(as.Date(date_str, format = "%Y%m%d"), "%Y-%m-%d") # Format as yyyy-mm-dd
  })
  
  # Create a unique ID by combining date and a secondary identifier
  news_data$ID <- sapply(news_data$ID, function(id) {
    str_extract(id, "[[:digit:]]{8}_[[:digit:]]{6}") # Extract date and unique ID
  })
  
  # Clean the text column by removing titles and time sections
  news_data$text <- sapply(news_data$text, function(text) {
    text <- gsub("\\[\\[TITLE\\.START\\]\\].*?\\[\\[TITLE\\.END\\]\\]", "", text)
    text <- gsub("\\[\\[TIME\\.START\\]\\].*?\\[\\[TIME\\.END\\]\\]", "", text)
    text
  })
  
  # Clean and tokenize the text
  cleaned_conv <- sapply(news_data$text, function(text) {
    # Convert to lowercase, remove symbols, punctuation, and whitespace
    conv_text <- tolower(text) %>%
      gsub("♪", "", .) %>%
      removePunctuation() %>%
      removeNumbers() %>%
      str_squish()
    
    # Tokenize, remove stopwords, and filter tokens by length
    cleaned_tokens_vector <- conv_text %>%
      as_tibble() %>%
      unnest_tokens(input = value, output = tokens) %>%
      filter(!tokens %in% stopwords()) %>%
      filter(nchar(tokens) >= 3 & nchar(tokens) <= 20) %>%
      pull(tokens)
    
    # Return cleaned and tokenized text as a single string
    paste(cleaned_tokens_vector, collapse = " ")
  })
  
  # Assign the cleaned text to the dataframe and drop the original text column
  news_data$cleaned_text <- cleaned_conv
  news_data <- news_data %>% select(-text)
  
  # Save the cleaned data to a new CSV file
  output_file <- paste0("news_", year, "_cleaned.csv")
  write.csv(news_data, output_file, row.names = FALSE)
}

data_full <- c()

# Load all data using data.table for better performance
for (i in 2016:2024) {
  data <- read_csv(paste0("/Users/giuseppelando/Desktop/ATDA EXAM/Data/news_", as.character(i), "_cleaned.csv"))
  data_full <- bind_rows(data_full, data)  
}

### Task 1: Corpus construction

# We try to restrict now the dataframe just to keep the news in which nvidia is cited

# Use grepl with word boundaries to find rows containing the exact word "nvidia"
nvidia_news <- data_full %>%
  filter(grepl("\\bnvidia\\b", cleaned_text, ignore.case = TRUE))

grepl("nvidia", nvidia_news[1,3], ignore.case = TRUE) # Test 

# Save results to CSV
fwrite(nvidia_news, "nvidia_news.csv")
nvidia_news <- read.csv("nvidia_news.csv") 

#### Build a KWIC on this updated dataframe

toks <- tokens(x = nvidia_news$cleaned_text)

kw.env <- kwic(x = toks,                      
               pattern = "\\bnvidia\\b" ,     
               valuetype = "regex",          
               window = 15,                  
               case_insensitive =  T,        
)

### Let's manipulate the dataframe
# Goal 1: remove not useful columns (from and to)
# Goal 2: merge the columns 'pre', 'keyword' and 'post'
# Goal 3: merge the rows of a single document to obtain a single string 

kw.env$text <- paste(kw.env$pre, kw.env$keyword, kw.env$post)  
kw.env$docname <- as.numeric(gsub("text", "", kw.env$docname)) 

nvidia_context_news <- kw.env %>%
  as_tibble() %>%
  select(-c(pre,keyword, post, from, to, pattern, )) %>%
  group_by(docname) %>%
  summarize(cleaned_text_unique = paste(text, collapse = " "), .groups = "drop")

nvidia_context_news$docname <- nvidia_news$ID
nvidia_context_news$date <- nvidia_news$date

fwrite(nvidia_context_news, "nvidia_context_news.csv") 
nvidia_context_news <- read_csv("nvidia_context_news.csv")

# Update Preprocessing to eliminate specific noisy words, 
custom_stopwords <- c("hour", "katie", "mandeep", "annabelle", "pretty", "important", "mark", "almost", "percent", "nvidia", "year", "company", "entheem", "topic", "scan", "caroline", stopwords("en"))
nvidia_context_news <- nvidia_context_news %>%
  # Remove all non-alphabetic characters and keep only letters and spaces
  mutate(cleaned_text_unique = str_replace_all(cleaned_text_unique, "[^a-zA-Z\\s]", "")) %>%
  # Remove specific unwanted word "emtheem" (ensuring word boundaries with \\b)
  mutate(cleaned_text_unique = str_replace_all(cleaned_text_unique, "\\bemtheem\\b", "")) %>%
  # Remove custom stopwords (irrelevant or frequently used words) defined in `custom_stopwords`
  mutate(cleaned_text_unique = removeWords(cleaned_text_unique, custom_stopwords)) %>%
  # Lemmatize the text to reduce words to their base or root form
  mutate(cleaned_text_unique = lemmatize_strings(cleaned_text_unique)) %>%
  # Remove any excess whitespace from the text
  mutate(cleaned_text_unique = str_squish(cleaned_text_unique))

# Generate bigrams
text_bigrams <- sapply(nvidia_context_news$cleaned_text_unique, function(text){
  tokenize_ngrams(text, n = 2, ngram_delim = "_")})

bigrams_sum <- 0

for (i in seq_along(text_bigrams)) {
  no_bigrams <- length(text_bigrams[[i]])
  bigrams_sum <- bigrams_sum + no_bigrams
}
bigrams_sum

# After the preprocessing and cleaning we will create a corpus containing 243642 bigrams from 2340 documents
corpus <- Corpus(VectorSource(text_bigrams))
# Build DTM with refined bounds
dtm <- DocumentTermMatrix(Corpus(VectorSource(text_bigrams)),
                          control = list(bounds = list(global = c(20, 500))))


########################################################################################################################
########################################################################################################################
########################################################################################################################

### Task 2: Competition analysis

# Define the range of years
years <- 2016:2024

# Loop through each year
for (year in years) {
  
  # Import cleaned data set for the current year
  file_name <- paste0("news_", year, "_cleaned.csv")
  news_cleaned <- read_csv(file_name)
  
  # Group by date and count articles
  news_reduced <- news_cleaned %>%
    group_by(date) %>%
    mutate(daily_article_count = n()) %>%
    ungroup() %>%
    group_by(date) %>%
    filter(
      (daily_article_count > 15 & row_number() <= 5) |  # Retain up to 5 news if count > 15
        (daily_article_count <= 15 & daily_article_count > 9 & row_number() <= 3) | # Retain up to 3 news if count is between 9 and 15
        (daily_article_count <= 9 & row_number() == 1)    # Retain 1 news if count < 9
    ) %>%
    ungroup()
  
  # Estimate the word2vec model
  model <- word2vec(
    x = news_reduced$cleaned_text,
    dim = 100,
    type = "skip-gram",
    iter = 20,
    min_count = 5,
    stopwords = stopwords(),
    threads = 10
  )
  
  # Predict the closest words to "nvidia"
  closest_words <- predict(
    object = model,
    newdata = c("nvidia"),
    type = "nearest",
    top_n = 20
  )
  
  # Convert the results to a data frame for LaTeX table output
  results_table <- as.data.frame(closest_words)
  colnames(results_table) <- c("Word", "Similarity")
  
  # Generate LaTeX table and save to file
  latex_table <- xtable(results_table, caption = paste("Closest Words to 'nvidia' - Year", year))
  table_file <- paste0("closest_words_", year, ".tex")
  
  # Save the LaTeX table to a .tex file
  print(latex_table, file = table_file, include.rownames = FALSE, caption.placement = "top")
  
  print(paste("LaTeX table saved for year", year, "in", table_file))
}


########################################################################################################################
########################################################################################################################
########################################################################################################################

### Task 3: Sentiment analysis

# Handle the stock returns file
# Read the stock price file
stock_prices <- read_csv("NVIDIA_stock_prices.csv")
# Rename some columns
colnames(stock_prices)[colnames(stock_prices) == "Date"] <- "date"
colnames(stock_prices)[colnames(stock_prices) == "Close/Last"] <- "close"
# Change the format from character to date
stock_prices$date <- as.Date(stock_prices$date, format = "%m/%d/%Y")
typeof(stock_prices$date)
# Since the close variable is in character form, we make it numeric
stock_prices <- stock_prices %>%
  mutate(close = as.numeric(gsub("[$]", "", close)))
typeof(stock_prices$close)

# Split the data into chunks for parallel processing
context_speeches <- split(nvidia_context_news, nvidia_context_news$date)

# Set up cluster
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)

# Export required packages and data to the cluster
clusterEvalQ(cl, library(sentimentr))
hash_sentiment_loughran_mcdonald <- lexicon::hash_sentiment_loughran_mcdonald
clusterExport(cl, list("context_speeches", "hash_sentiment_loughran_mcdonald"))

# Perform sentiment analysis in parallel
context_results <- parLapply(cl, context_speeches, function(chunk) {
  chunk$sentimentLM <- sentiment_by(chunk$cleaned_text_unique, polarity_dt = hash_sentiment_loughran_mcdonald)$ave_sentiment
  return(chunk)
})

# Stop cluster
stopCluster(cl)

# Combine results back into a single data frame
context_news_with_sentiment_full_interval <- rbindlist(context_results)

# Save results to CSV
fwrite(context_news_with_sentiment_full_interval, "context.sentimentLM.csv")

# Load the CSV
context_news_with_sentiment_full_interval <- read_csv("context.sentimentLM.csv") 

# Compute the average sentiment score per day 
context_news_with_sentiment_full_interval %>%
  group_by(date) %>%
  summarise(mean.LM=mean(sentimentLM),
            number.of.news=n()) -> context_daily_sentiment_LM_full_interval

context_nvidia_data_full_interval <- context_daily_sentiment_LM_full_interval %>%
  select(date, mean.LM, number.of.news) %>%
  left_join(stock_prices %>% select(date, close), by = "date") %>%
  filter(!is.na(close))

context_nvidia_data_full_interval <- context_nvidia_data_full_interval %>%
  mutate(close = as.numeric(gsub("[$]", "", close)))

# Reshape the data into long format
context_nvidia_data_full_interval_long <- context_nvidia_data_full_interval %>%
  pivot_longer(cols = c(mean.LM, number.of.news, close), 
               names_to = "variable", 
               values_to = "value")

# Plot with facets for each variable and add moving average line for mean.LM

library(tidyquant)  # For the moving average function

# Recode the 'variable' column to have the desired labels
context_nvidia_data_full_interval_long <- context_nvidia_data_full_interval_long %>%
  mutate(variable = recode(variable,
                           "close" = "Close price",
                           "mean.LM" = "LM sentiment score",
                           "number.of.news" = "Number of news"))

# Recode the 'variable' column and set factor levels to control order
context_nvidia_data_full_interval_long <- context_nvidia_data_full_interval_long %>%
  mutate(variable = recode(variable,
                           "close" = "Close price",
                           "mean.LM" = "LM sentiment score",
                           "number.of.news" = "Number of news"),
         variable = factor(variable, levels = c("Close price", 
                                                "LM sentiment score", 
                                                "Number of news")))  


full_interval_context_news <- ggplot(context_nvidia_data_full_interval_long, aes(x = date, y = value, color = variable)) +
  geom_line() +
  
  # Add moving average line for recoded variables
  geom_ma(data = subset(context_nvidia_data_full_interval_long, variable == "LM sentiment score"), 
          aes(x = date, y = value), 
          n = 7, color = "blue", size = 0.5) +  # 7-day moving average
  
  # Use color scale without manually specifying colors
  scale_color_discrete(name = "Variables") +  # Set legend title to "Variables"
  
  facet_wrap(~ variable, scales = "free_y", ncol = 1) + 
  labs(x = "Date", y = "Value", title = "Nvidia: Daily sentiment, news count, and stock price", subtitle = "Pieces of news in which Nvidia is cited directly, full interval (2016-2024)") +
  theme_minimal()

ggsave("nvidia_full_interval_context_news.png", plot = full_interval_context_news, 
       width = 10, height = 8, dpi = 300, bg = "white", device = "png")

# Assess a linear relationship between returns and daily sentiment 

context_nvidia_data_full_interval <- context_nvidia_data_full_interval %>%
  mutate(return = (close - lag(close)) / lag(close)) %>%
  filter(!is.na(return))

fit_context_full_interval <- lm(return ~ mean.LM + number.of.news, data = context_nvidia_data_full_interval)
summary(fit_context_full_interval)

# Plot the data points
full_interval_context_plot <- ggplot(context_nvidia_data_full_interval, aes(x = mean.LM, y = return)) +
  geom_point(color = "blue", shape = 16) +  # Add points
  geom_smooth(method = "lm", color = "red", size = 1.5) +  # Add regression line
  labs(
    title = "NVIDIA Stock Return vs Sentiment Score", 
    subtitle = "Full interval, Nvidia Context sample news", 
    x = "Sentiment Score (mean.LM)", 
    y = "Return"
  ) +
  theme_minimal()  

ggsave("nvidia_stock_vs_sentiment_full_interval_context.png", plot = full_interval_context_plot, width = 8, height = 6, bg = "white")


### We apply the same time interval restriction as before 

stock_prices_restricted <- stock_prices %>%
  filter(date > as.Date("2023-01-01") & date < as.Date("2024-06-10"))

context_nvidia_data_restricted_interval <- context_daily_sentiment_LM_full_interval %>%
  select(date, mean.LM, number.of.news) %>%
  left_join(stock_prices_restricted %>% select(date, close), by = "date") %>%
  filter(!is.na(close))

# Reshape the data into long format
context_nvidia_data_restricted_interval_long <- context_nvidia_data_restricted_interval %>%
  pivot_longer(cols = c(mean.LM, number.of.news, close), 
               names_to = "variable", 
               values_to = "value")

# Plot with facets for each variable and add moving average line for mean.LM

# Recode the 'variable' column to have the desired labels
context_nvidia_data_restricted_interval_long <- context_nvidia_data_restricted_interval_long %>%
  mutate(variable = recode(variable,
                           "close" = "Close price",
                           "mean.LM" = "LM sentiment score",
                           "number.of.news" = "Number of news"))

# Recode the 'variable' column and set factor levels to control order
context_nvidia_data_restricted_interval_long <- context_nvidia_data_restricted_interval_long %>%
  mutate(variable = recode(variable,
                           "close" = "Close price",
                           "mean.LM" = "LM sentiment score",
                           "number.of.news" = "Number of news"),
         variable = factor(variable, levels = c("Close price", 
                                                "LM sentiment score", 
                                                "Number of news")))  

# Plot with a moving average line in orange and custom legend labels
restricted_interval_context_news <- ggplot(context_nvidia_data_restricted_interval_long, aes(x = date, y = value, color = variable)) +
  geom_line() +
  
  geom_ma(data = subset(context_nvidia_data_restricted_interval_long, variable == "LM sentiment score"), 
          aes(x = date, y = value), 
          n = 7, color = "blue", size = 0.5) + 
  
  # Use color scale without manually specifying colors
  scale_color_discrete(name = "Variables") +  # Set legend title to "Variables"
  
  facet_wrap(~ variable, scales = "free_y", ncol = 1) + 
  labs(x = "Date", y = "Value", title = "Nvidia: Daily sentiment, news count, and stock price", subtitle = "Pieces of news in which Nvidia is cited directly, restricted interval (2023-2024)") +
  theme_minimal()

ggsave("nvidia_restricted_interval_context_news.png", plot = restricted_interval_context_news, 
       width = 10, height = 8, dpi = 300, bg = "white", device = "png")

context_nvidia_data_restricted_interval <- context_nvidia_data_restricted_interval %>%
  mutate(return = (close - lag(close)) / lag(close)) %>%
  filter(!is.na(return))

fit_context_restricted_interval <- lm(return ~ mean.LM + number.of.news, data = context_nvidia_data_restricted_interval)
summary(fit_context_restricted_interval)

# Plot the data points
restricted_interval_context_plot <- ggplot(context_nvidia_data_restricted_interval, aes(x = mean.LM, y = return)) +
  geom_point(color = "blue", shape = 16) +  # Add points
  geom_smooth(method = "lm", color = "red", size = 1.5) +  # Add regression line
  labs(
    title = "NVIDIA Stock Return vs Sentiment Score", 
    subtitle = "Restricted interval (2023-2024), Nvidia context sample news", 
    x = "Sentiment Score (mean.LM)", 
    y = "Return"
  ) +
  theme_minimal()  # Optional: clean theme for better presentation

ggsave("nvidia_stock_vs_sentiment_restricted_interval_context.png", plot = restricted_interval_context_plot, width = 8, height = 6, bg = "white")


# Linear Regressions 
library(stargazer)

# Generate the regression table
output <- stargazer(fit_context_full_interval, 
                    fit_context_restricted_interval,
                    type = "latex",
                    report = "vc*t",  # Show coefficients, standard errors, t-values, and significance
                    title = "Regression Output: Returns vs SentimentLM Score",
                    covariate.labels = "SentimentLM Score",  # Rename covariate
                    dep.var.labels = "Stock Returns",  # Label dependent variable
                    column.labels = c("News Context Retained",
                                      "Interval Restricted, News Context Retained"),
                    omit.stat = c( "ser", "f"),  # Omit R2, Adjusted R2, Residual Std. Error, F Statistic
                    notes = "Each regression explores the impact of SentimentLM scores on Nvidia stock returns under different conditions.",
                    out = "output")


########################################################################################################################
########################################################################################################################
########################################################################################################################

### Task 4: MNIR

# Reason for price increase – context based approach: Which terms (uni- or bigrams) are associated
# with NVIDIA in times of a high stock price versus low stock price? - MNIR model estimation

stock_prices <- read_csv("NVIDIA_stock_prices.csv")

# Adjust the date column - chage to date format
stock_prices$Date <- as.Date(stock_prices$Date, format = "%m/%d/%Y")

# Join the Close/Last column from NVIDIA stock prices data frame to the corpus data by Date
nvidia_closing_prices <- stock_prices %>% select(Date, `Close/Last`)
nvidia_closing_prices <- nvidia_closing_prices %>% rename(close_last = `Close/Last`)
nvidia_closing_prices <- nvidia_closing_prices %>% rename(date = Date)

merged_data <- nvidia_context_news %>%
  left_join(nvidia_closing_prices, by = "date")

# Convert the prices to numerical values
merged_data$close_last <- as.numeric(gsub("[$]", "", merged_data$close_last))

# Check for NA values in merged_data 
sum(!is.finite(merged_data$close_last)) # 179 - weekends and some of national holidays

# Modify the dtm and merged_data to not use the rows where close_last are NA
dtm_mnir <- dtm[which(!is.na(merged_data$close_last)), ]
merged_data_mnir <- merged_data %>% filter(!is.na(close_last))

# Estimate the MNIR model with parallel session
cl <- makeCluster(5)

mnir <- dmr(cl, 
            covars = merged_data_mnir[, "close_last"], 
            counts = dtm_mnir, 
            bins=NULL, 
            gamma=10, 
            nlambda=10, 
            verb= 2)

stopCluster(cl)

# Get the coefficient
mnir.coef <- coef(mnir)
mnir.coef.df <- as.data.frame(as.matrix(mnir.coef))
mnir.coef.df <- t(mnir.coef.df)

mnir.coef.df <- as.data.frame(mnir.coef.df)
mnir.coef.df <- rownames_to_column(mnir.coef.df, var = "bigram")

mnir.coef.df <- mnir.coef.df %>%
  select(bigram, close_last) %>%
  arrange(desc(close_last))

# Bigrams associated with the highest prices
head(mnir.coef.df, 20)

# Bigrams associated with the lowest prices
tail(mnir.coef.df, 20)


########################################################################################################################
########################################################################################################################
########################################################################################################################

### Task 5: Topic model

# We will use unigrams for better better adaptability and flexibility 
# Generate unigrams
text_unigrams <- sapply(nvidia_context_news$cleaned_text_unique, function(text) {
  tokenize_words(text)
})

# Count total unigrams
unigrams_sum <- 0

for (i in seq_along(text_unigrams)) {
  no_unigrams <- length(text_unigrams[[i]])
  unigrams_sum <- unigrams_sum + no_unigrams
}
unigrams_sum

# After the preprocessing and cleaning we will create a corpus containing 245982 unigrams from 2340 documents
corpus <- Corpus(VectorSource(text_unigrams))
# Build DTM with refined bounds
dtm <- DocumentTermMatrix(Corpus(VectorSource(text_unigrams)),
                          control = list(bounds = list(global = c(20, 500))))

dtm <- dtm[row_sums(dtm) > 10,]

# Fit LDA model
topic <- LDA(dtm, 
             k = 30, 
             method = "Gibbs",
             control = list(
               seed = 1234, 
               burnin = 100,  
               iter = 800,  
               keep = 1,    
               save = F,    
               verbose = 10  
             ))

# inspect top 10 terms for all topics
apply(topic@beta, 1, function(x) head(topic@terms[order(x, decreasing = T)],10))

# Which topic fits the description? 

competitors <- 6
arm_deal <- 8
positive_forecast <- 9
gaming_mining <- 14
semiconductor <-  22
artificial_intelligence <-  26
magnificent_seven <-  30


str(beta)
beta <- exp(topic@beta)  # Ensure it's a numeric matrix

# wordcloud with term distribution for each topic
terms.top.40 <- head(topic@terms[order(beta[positive_forecast,], decreasing = T)], 40)
prob.top.40 <- head(sort(beta[positive_forecast,], decreasing = T), 40)
terms.top.40 <- gsub('"', '', terms.top.40)

# Specify the output PNG file
png(filename = "wordcloud_topic_positive_forecast.png", width = 800, height = 800, res = 150)

# Generate the word cloud
wordcloud(
  words = terms.top.40,
  freq = prob.top.40,
  random.order = FALSE,
  scale = c(4, 1)
)

# Close the PNG device to save the file
dev.off()







