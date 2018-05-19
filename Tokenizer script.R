# load in libraries we'll need
library(tidyverse) #keepin' things tidy
library(tidytext) #package for tidy text analysis (Check out Julia Silge's fab book!)
library(glue) #for pasting strings
library(data.table) #for rbindlist, a faster version of rbind

# now let's read in some data & put it in a tibble (a special type of tidy dataframe)
file_info <- as_data_frame(read.csv("../input/guide_to_files.csv"))
head(file_info)



# stick together the path to the file & 1st file name from the information file
fileName <- glue("../input/", as.character(file_info$file_name[1]), sep = "")
# get rid of any sneaky trailing spaces
fileName <- trimws(fileName)

# read in the new file
fileText <- paste(readLines(fileName))
# and take a peek!
head(fileText)
# what's the structure?
str(fileText)




# "grep" finds the elements in the vector that contain the exact string *CHI:.
# (You need to use the double slashes becuase I actually want to match the character
# *, and usually that means "match any character"). We then select those indxes from
# the vector "fileText".
childsSpeech <- as_data_frame(fileText[grep("\\*CHI:",fileText)])
head(childsSpeech)


# use the unnest_tokens function to get the words from the "value" column of "child
childsTokens <- childsSpeech %>% unnest_tokens(word, value)
head(childsTokens)





# look at just the head of the sorted word frequencies
childsTokens %>% count(word, sort = T) %>% head



# anti_join removes any rows that are in the both dataframes, so I make a data_frame
# of 1 row that contins "chi" in the "word" column.
sortedTokens <- childsSpeech %>% unnest_tokens(word, value) %>% anti_join(data_frame(word = "chi")) %>% 
  count(word, sort = T)
head(sortedTokens)




# let's make a function that takes in a file & exactly replicates what we just did
fileToTokens <- function(filename){
  # read in data
  fileText <- paste(readLines(filename))
  # get child's speech
  childsSpeech <- as_data_frame(fileText[grep("\\*CHI:",fileText)])
  # tokens sorted by frequency 
  sortedTokens <- childsSpeech %>% unnest_tokens(word, value) %>% 
    anti_join(data_frame(word = "chi")) %>% 
    count(word, sort = T)
  # and return that to the user
  return(sortedTokens)
}





# we still have this fileName variable we assigned at the beginning of the tutorial
fileName

# so let's use that...
head(fileToTokens(fileName))
# and compare it to the data we analyzed step-by-step
head(sortedTokens)




# let's write another function to clean up file names. (If we can avoid 
# writing/copy pasting the same codew we probably should)
prepFileName <- function(name){
  # get the filename
  fileName <- glue("../input/", as.character(name), sep = "")
  # get rid of any sneaky trailing spaces
  fileName <- trimws(fileName)
  
  # can't forget to return our filename!
  return(fileName)
}

# make an empty dataset to store our results in
tokenFreqByChild <- NULL

# becuase this isn't a very big dataset, we should be ok using a for loop
# (these can be slow for really big datasets, though)
for(name in file_info$file_name){
  # get the name of a specific child
  child <- name
  
  # use our custom functions we just made!
  tokens <- prepFileName(child) %>% fileToTokens()
  # and add the name of the current child
  tokensCurrentChild <- cbind(tokens, child)
  
  # add the current child's data to the rest of it
  # I'm using rbindlist here becuase it's much more efficent (in terms of memory
  # usage) than rbind
  tokenFreqByChild <- rbindlist(list(tokensCurrentChild,tokenFreqByChild))
}

# make sure our resulting dataframe looks reasonable
summary(tokenFreqByChild)
head(tokenFreqByChild)



# let's plot the how many words get used each number of times 
ggplot(tokenFreqByChild, aes(n)) + geom_histogram()







#first, let's look at only the rows in our dataframe where the word is "um"
ums <- tokenFreqByChild[tokenFreqByChild$word == "um",]

# now let's merge our ums dataframe with our information file
umsWithInfo <- merge(ums, file_info, by.y = "file_name", by.x = "child")
head(umsWithInfo)







# see if there's a significant correlation
cor.test(umsWithInfo$n, umsWithInfo$months_of_english)

# and check the plot
ggplot(umsWithInfo, aes(x = n, y = months_of_english)) + geom_point() + 
  geom_smooth(method = "lm")



