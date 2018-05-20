

library("readr")
library("dplyr")
library("plotly")
library("stringr")
library("stringi")
library("scales")
library("mxnet")


corpus_bucketed_train <- readRDS(file = "../data/train_buckets_one_to_one.rds")
corpus_bucketed_test <- readRDS(file = "../data/eval_buckets_one_to_one.rds")
vocab <- length(corpus_bucketed_test$dic)

### Create iterators
batch.size = 32
train.data <- mx.io.bucket.iter(buckets = corpus_bucketed_train$buckets, batch.size = batch.size, 
                                data.mask.element = 0, shuffle = TRUE)
eval.data <- mx.io.bucket.iter(buckets = corpus_bucketed_test$buckets, batch.size = batch.size,
                               data.mask.element = 0, shuffle = FALSE)
rnn_graph_one_one <- rnn.graph(num_rnn_layer = 3, 
                               num_hidden = 96,
                               input_size=vocab,
                               num_embed=64, 
                               num_decode =vocab,
                               dropout=0.2, 
                               ignore_label = 0,
                               cell_type="lstm",
                               masking = F,
                               output_last_state = T,
                               loss_output = "softmax",
                               config = "one-to-one")
graph.viz(rnn_graph_one_one, type = "graph", direction = "LR", 
          graph.height.px = 180, shape=c(100, 64))






devices <- mx.gpu(0)
initializer <- mx.init.Xavier(rnd_type = "gaussian", factor_type = "avg", magnitude = 3)
optimizer <- mx.opt.create("adadelta", rho = 0.9, eps = 1e-5, wd = 1e-8,
                           clip_gradient = 5, rescale.grad = 1/batch.size)

logger <- mx.metric.logger()
epoch.end.callback <- mx.callback.log.train.metric(period = 1, logger = logger)
batch.end.callback <- mx.callback.log.train.metric(period = 50)

mx.metric.custom_nd <- function(name, feval) {
  init <- function() {
    c(0, 0)
  }
  update <- function(label, pred, state) {
    m <- feval(label, pred)
    state <- c(state[[1]] + 1, state[[2]] + m)
    return(state)
  }
  get <- function(state) {
    list(name=name, value=(state[[2]]/state[[1]]))
  }
  ret <- (list(init=init, update=update, get=get))
  class(ret) <- "mx.metric"
  return(ret)
}

mx.metric.Perplexity <- mx.metric.custom_nd("Perplexity", function(label, pred) {
  label = mx.nd.reshape(label, shape = -1)
  label_probs <- as.array(mx.nd.choose.element.0index(pred, label))
  batch <- length(label_probs)
  NLL <- -sum(log(pmax(1e-15, as.array(label_probs)))) / batch
  Perplexity <- exp(NLL)
  return(Perplexity)
})
model <- mx.model.buckets(symbol = rnn_graph_one_one,
                          train.data = train.data, eval.data = eval.data, 
                          num.round = 20, ctx = devices, verbose = TRUE,
                          metric = mx.metric.Perplexity, 
                          initializer = initializer, optimizer = optimizer, 
                          batch.end.callback = NULL, 
                          epoch.end.callback = epoch.end.callback)
mx.model.save(model, prefix = "../models/model_one_to_one_lstm_gpu", iteration = 1)


# Inference on test data
# Setup inference data. Need to apply preprocessing to inference sequence and convert into a infer data iterator.
ctx <- mx.gpu(0)
batch.size <- 1

corpus_bucketed_train <- readRDS(file = "../data/train_buckets_one_to_one.rds")
dic <- corpus_bucketed_train$dic
rev_dic <- corpus_bucketed_train$rev_dic

infer_raw <- c("The United States are")
infer_split <- dic[strsplit(infer_raw, '') %>% unlist]
infer_length <- length(infer_split)

infer.data <- mx.io.arrayiter(data = matrix(infer_split), label = matrix(infer_split),  
                              batch.size = 1, shuffle = FALSE)
# 
# Inference with most likely term
# Here the predictions are performed by picking the character whose associated probablility is the highest.
model <- mx.model.load(prefix = "../models/model_one_to_one_lstm_gpu", iteration = 1)

internals <- model$symbol$get.internals()

sym_state <- internals$get.output(which(internals$outputs %in% "RNN_state"))
sym_state_cell <- internals$get.output(which(internals$outputs %in% "RNN_state_cell"))
sym_output <- internals$get.output(which(internals$outputs %in% "loss_output"))
symbol <- mx.symbol.Group(sym_output, sym_state, sym_state_cell)

predict <- numeric()

infer <- mx.infer.rnn.one(infer.data = infer.data, 
                          symbol = symbol,
                          arg.params = model$arg.params,
                          aux.params = model$aux.params,
                          input.params = NULL, 
                          ctx = ctx)

pred_prob <- mx.nd.slice.axis(infer$loss_output, axis=0, begin = infer_length-1, end = infer_length)
pred <- mx.nd.argmax(data = pred_prob, axis = 1, keepdims = T)
predict <- c(predict, as.numeric(as.array(pred)))

for (i in 1:100) {
  
  infer.data <- mx.io.arrayiter(data = as.matrix(pred), label = as.matrix(pred),  
                                batch.size = 1, shuffle = FALSE)
  
  infer <- mx.infer.rnn.one(infer.data = infer.data, 
                            symbol = symbol,
                            arg.params = model$arg.params,
                            aux.params = model$aux.params,
                            input.params = list(rnn.state = infer[[2]], 
                                                rnn.state.cell = infer[[3]]), 
                            ctx = ctx)
  
  pred <- mx.nd.argmax(data = infer$loss_output, axis = 1, keepdims = T)
  predict <- c(predict, as.numeric(as.array(pred)))
  
}

predict_txt <- paste0(rev_dic[as.character(predict)], collapse = "")
predict_txt_tot <- paste0(infer_raw, predict_txt, collapse = "")

# 
# Generated sequence: The United States are a standing the country and the challenges and the challenges and the challenges and the challenges a
# 
# Key ideas appear somewhat overemphasized.
# 
# Inference from random sample
# Noise is now inserted in the predictions by sampling each character based on their modeled probability.


set.seed(44)

infer_raw <- c("The United States are")
infer_split <- dic[strsplit(infer_raw, '') %>% unlist]
infer_length <- length(infer_split)

infer.data <- mx.io.arrayiter(data = matrix(infer_split), label = matrix(infer_split),  
                              batch.size = 1, shuffle = FALSE)

predict <- numeric()

infer <- mx.infer.rnn.one(infer.data = infer.data, 
                          symbol = symbol,
                          arg.params = model$arg.params,
                          aux.params = model$aux.params,
                          input.params = NULL, 
                          ctx = ctx)

pred_prob <- as.numeric(as.array(mx.nd.slice.axis(
  infer$loss_output, axis=0, begin = infer_length-1, end = infer_length)))
pred <- sample(length(pred_prob), prob = pred_prob, size = 1) - 1
predict <- c(predict, pred)

for (i in 1:100) {
  
  infer.data <- mx.io.arrayiter(data = as.matrix(pred), label = as.matrix(pred),  
                                batch.size = 1, shuffle = FALSE)
  
  infer <- mx.infer.rnn.one(infer.data = infer.data, 
                            symbol = symbol,
                            arg.params = model$arg.params,
                            aux.params = model$aux.params,
                            input.params = list(rnn.state = infer[[2]], 
                                                rnn.state.cell = infer[[3]]), 
                            ctx = ctx)
  
  pred_prob <- as.numeric(as.array(infer$loss_output))
  pred <- sample(length(pred_prob), prob = pred_prob, size = 1, replace = T) - 1
  predict <- c(predict, pred)
}

predict_txt <- paste0(rev_dic[as.character(predict)], collapse = "")
predict_txt_tot <- paste0(infer_raw, predict_txt, collapse = "")

