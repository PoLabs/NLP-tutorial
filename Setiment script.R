
setwd('C:/Users/Po/Sync/MDMA NLP/NLP tutorial - kaggle')

# load in the libraries we'll need
library(tidyverse)
library(tidytext)
library(glue)
library(stringr)

DTreviews <- read_csv("trumptweets.csv")
DTreviews <- DTreviews[,5]
DTtweet <- ""
for (i in 1:32826){
  DTtweet <- paste0(DTtweet, " ", DTreviews[i,1])
}

fileConn<-file("DTtweets.txt")
writeLines(DTtweet, fileConn)
close(fileConn)
DTtweet <- read_file("DTtweets.txt")

DTtweet = gsub("&amp", " ", DTtweet)
DTtweet = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", " ", DTtweet)
DTtweet = gsub("@\\w+", " ", DTtweet)
DTtweet = gsub("[[:punct:]]", " ", DTtweet)
DTtweet = gsub("[[:digit:]]", " ", DTtweet)
DTtweet = gsub("http\\w+", " ", DTtweet)
DTtweet = gsub("[ \t]{2,}", " ", DTtweet)
DTtweet = gsub("^\\s+|\\s+$", " ", DTtweet) 
#get rid of unnecessary spaces
#DTtweet <- str_replace_all(DTtweet," "," ")
# Get rid of URLs
#DTtweet <- str_replace_all(DTtweet, "http://t.co/[a-z,A-Z,0-9]*{8}"," ")
DTtweet <- str_replace_all(DTtweet, "https://t.co/[a-z,A-Z,0-9]*","")
DTtweet <- str_replace_all(DTtweet, "http://t.co/[a-z,A-Z,0-9]*","")
# Take out retweet header, there is only one
DTtweet <- str_replace(DTtweet,"RT @[a-z,A-Z]*: "," ")
# Get rid of hashtags
DTtweet <- str_replace_all(DTtweet,"#[a-z,A-Z]*"," ")
# Get rid of references to other screennames
DTtweet <- str_replace_all(DTtweet,"@[a-z,A-Z]*"," ")  
DTtweet <- str_replace_all(DTtweet, "http", "")
DTtweet <- str_replace_all(DTtweet, "tinyurl", "")
DTtweet <- str_replace_all(DTtweet, "com", "")
DTtweet <- str_replace_all(DTtweet, "  ", " ")
DTtweet <- str_replace_all(DTtweet, "  ", " ")
DTtweet <- str_replace_all(DTtweet, "  ", " ")


# get a list of the files in the input directory
# files <- list.files("sentiment corpus")
# 
# # stick together the path to the file & 1st file name
# fileName <- glue("sentiment corpus/", files[1], sep = "")
# # get rid of any sneaky trailing spaces
# fileName <- trimws(fileName)
# 
# # read in the new file
# fileText <- glue(read_file(fileName))
# # remove any dollar signs (they're special characters in R)
# fileText <- gsub("\\$", "", fileText) 
fileText <- DTtweet

# tokenize
tokens <- data_frame(text = fileText) %>% unnest_tokens(word, text)

# get the sentiment from the first text: 
tokens %>%
  inner_join(get_sentiments("bing")) %>% # pull out only sentiment words
  count(sentiment) %>% # count the # of positive & negative words
  spread(sentiment, n, fill = 0) %>% # made data wide rather than narrow
  mutate(sentiment = positive - negative) # # of positive words - # of negative owrds

# negative positive sentiment
# <dbl>    <dbl>     <dbl>
#   1    16636    37488     20852



# write a function that takes the name of a file and returns the # of postive
# sentiment words, negative sentiment words, the difference & the normalized difference
GetSentiment <- function(file){
  # get the file
  fileName <- glue("", file, sep = "")
  # get rid of any sneaky trailing spaces
  #fileName <- trimws(fileName)
  
  # read in the new file
  fileText <- glue(read_file(fileName))
  # remove any dollar signs (they're special characters in R)
  #fileText <- gsub("\\$", "", fileText) 
  
  # tokenize
  tokens <- data_frame(text = fileText) %>% unnest_tokens(word, text)
  
  # get the sentiment from the first text: 
  sentiment <- tokens %>%
    inner_join(get_sentiments("bing")) %>% # pull out only sentimen words
    count(sentiment) %>% # count the # of positive & negative words
    spread(sentiment, n, fill = 0) %>% # made data wide rather than narrow
    mutate(sentiment = positive - negative) %>% # # of positive words - # of negative owrds
    mutate(file = file) #%>% # add the name of our file
    #mutate(year = as.numeric(str_match(file, "\\d{4}"))) %>% # add the year
    #mutate(president = str_match(file, "(.*?)_")[2]) # add president
  
  # return our sentiment dataframe
  return(sentiment)
}

# test: should return
# negative	positive	sentiment	file	year	president
# 117	240	123	Bush_1989.txt	1989	Bush
sentiments <- GetSentiment("DTtweets.txt")


# 
# # file to put our output in
# sentiments <- data_frame()
# 
# # get the sentiments for each file in our datset
# for(i in files){
#   sentiments <- rbind(sentiments, GetSentiment(i))
# }

# disambiguate Bush Sr. and George W. Bush 
# correct president in applicable rows
# bushSr <- sentiments %>% 
#   filter(president == "Bush") %>% # get rows where the president is named "Bush"...
#   filter(year < 2000) %>% # ...and the year is before 200
#   mutate(president = "Bush Sr.") # and change "Bush" to "Bush Sr."
# # remove incorrect rows
# sentiments <- anti_join(sentiments, sentiments[sentiments$president == "Bush" & sentiments$year < 2000, ])
# # add corrected rows to data_frame 
# sentiments <- full_join(sentiments, bushSr)

# summerize the sentiment measures
summary(sentiments)

sent.df <- data.frame(num=c('negative','positive','overall'), sents=c(16636,37488,20852) )
library(ggplot2)
g <- ggplot(data=sent.df, aes(num, sents)) + geom_bar(stat="identity")













# plot of sentiment over time & automatically choose a method to model the change
ggplot(sentiments, aes(x = as.numeric(year), y = sentiment)) + 
  geom_point(aes(color = president))+ # add points to our plot, color-coded by president
  geom_smooth(method = "auto") # pick a method & fit a model


# plot of sentiment by president
ggplot(sentiments, aes(x = president, y = sentiment, color = president)) + 
  geom_boxplot() # draw a boxplot for each president



# is the difference between parties significant?
# get democratic presidents & add party affiliation
democrats <- sentiments %>%
  filter(president == c("Clinton","Obama")) %>%
  mutate(party = "D")

# get democratic presidents & party add affiliation
republicans <- sentiments %>%
  filter(president != "Clinton" & president != "Obama") %>%
  mutate(party = "R")

# join both
byParty <- full_join(democrats, republicans)

# the difference between the parties is significant
t.test(democrats$sentiment, republicans$sentiment)

# plot sentiment by party
ggplot(byParty, aes(x = party, y = sentiment, color = party)) + geom_boxplot() + geom_point()





#Exercise 1: Normalizing for text length
# Rewrite the function GetSentiment so that it also returns the sentiment score
# divided by the number of words in each document.

# hint: you can use the function nrow() on your tokenized data_frame to find 
# the number of tokens in each document

# How does normalizing for text length change the outcome of the analysis?



