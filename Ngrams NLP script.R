# Yes. Package NLP provides functionality to compute n-grams which can be used to construct a corresponding tokenizer. E.g.:
#library('NLP')
library("tm")
data("crude")

BigramTokenizer <-
  function(x)
    unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)

tdm <- TermDocumentMatrix(crude, control = list(tokenize = BigramTokenizer))
inspect(removeSparseTerms(tdm[, 1:10], 0.7))

