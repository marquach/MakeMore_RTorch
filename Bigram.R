# Bigram language model (Andrej Karpathy Youtube Video Make More)

#load data
names <- read.table(file = "names.txt")

# Bigramm model is based on two consequtives characters, so Bigram is 
# (in this case) on  character level.

# Use moving window of size two per word and then count them

# first approach but slow because of rbind
#bigram_old = matrix(, ncol = 2)
#for(j in seq_len(dim(names)[1])){
#  name <- c(".", unlist(strsplit(names[j,1], split = "")), ".")
#  for(i in 2:length(name)){
#    bigram_old <- rbind(bigram_old,c(name[i-1],name[i]))
#  }
#}
#bigram_old <- bigram_old[-1,]

bigram <- sapply(X = seq_len(dim(names)[1]), FUN = function(X){
  name <- c(".", unlist(strsplit(names[X,1], split = "")), ".")
  # nrow because of how unlist() works (iterates over columns)
  res <- matrix(NA, nrow = 2, ncol = length(name)-1)
  for(i in 2:length(name)){
    res[,i-1] <- c(name[i-1],name[i])
  }
  res
}, simplify = T)


bigram <- matrix(unlist(bigram), ncol = 2, byrow = T)
#all.equal(bigram, bigram_old)

test_set <- bigram
bigram <- apply(bigram, MARGIN = 1, function(x) paste(x, collapse = ""))
bigram_table <- table(bigram)
bigram_table[order(bigram_table, decreasing = T)] # to compare with makemore

N <- matrix(0, nrow = 26 + 1, ncol = 26 + 1) # 26 + 1 because of special char.

chars <- c(".", letters)
N_entries <- expand.grid(chars, chars, stringsAsFactors = F)
N_entries <- cbind(N_entries[,2], N_entries[,1])
N_entries<- apply(N_entries, MARGIN = 1, function(x) paste(x, collapse = ""))
#length(N_entries) == 27*27

head(bigram_table)
colnames(N) <- rownames(N) <- chars

stoi <- function(x){
  x <- unlist(strsplit(x, split = ""))
  c(which(x[1] == chars), which(x[2] == chars))
}

for(i in 1:length(bigram_table)){
  entry <- names(bigram_table)[i]
  entry_index <- stoi(entry)
  N[entry_index[1], entry_index[2]] <- as.integer(bigram_table[i])
}

library(tidyverse)
N %>% 
  as.data.frame() %>%
  rownames_to_column("f_id") %>%
  pivot_longer(-c(f_id), names_to = "samples", values_to = "counts") %>%
  ggplot(aes(x=samples, y=f_id, fill=counts)) + 
  geom_raster() + scale_y_discrete(limits = rev) +
  geom_text(aes(label = counts), size = 1, vjust = 0.75) +
  geom_text(aes(label = paste(f_id, samples)), size = 1, vjust = -0.75)+
  scale_fill_gradient(low = "white", high = "#1b98e0") +
  theme(legend.position = "none") + xlab("") + ylab("")

P <- (N)/rowSums(N)
# Comparison from jupyter notebook
P_python <- read.csv(file = "/Users/marquach/Desktop/Python/makemore-master/P_python.csv")
P_python <- P_python[,-1]
P_python <- as.matrix(P_python)
rownames(P_python) <- colnames(P_python) <- chars
all.equal(P, P_python) # almost same

itos <- function(x) chars[x]
  
set.seed(2147483647)
test_names <- c()
for (i in seq_len(5)) {
  test <- c()
  ix <- 1
  stop <- F
  while(!stop){
    p_multinomial <- P[ix,]
    ix <- which(rmultinom(n = 1, 1, p_multinomial)==1)
    if( ix == 1) break
    test <- paste(test,itos(ix), sep="")
    }
  test_names <- c(test_names, test)
  }
test_names

log_likelihood <- 0
n <- 0 
for(i in 1:length(bigram)){
    entry <- bigram[i]
    entry_index <- stoi(entry)
    prob <- P[entry_index[1], entry_index[2]]
    logprob <- log(prob)
    log_likelihood <- log_likelihood + logprob
    n <- n + 1
}
nll <- -log_likelihood
nll/n
loss_table <- nll/n


# Create training set of bigrams (x,y) for NN 
xs <- test_set[,1]
ys <- test_set[,2]

xs <- sapply(xs, FUN = stoi)
ys <- sapply(ys, FUN = stoi)

length(xs)
#prepare one hot encodings
xs_one_hot <- matrix(0, nrow = length(xs), ncol = 27)
for (i in seq_len(length(xs))) {
  xs_one_hot[i, xs[i]] <- 1
}
ys_one_hot <- matrix(0, nrow = length(xs), ncol = 27)
for (i in seq_len(length(ys))) {
  ys_one_hot[i, ys[i]] <- 1
}

plot(NA, xlim = c(1,27), ylim=c(5,1), type = "n")
points(apply(xs_one_hot[1:5,], MARGIN = 1, function(x) which(x == 1)), 1:5)

#first online nll for emma
xs <- xs[1:5]
ys <- ys[1:5]

#initialize random weight matrix
w <- matrix(data = rnorm(n = 27^2), ncol = 27, nrow = 27)
logits <- xs_one_hot[1:5,] %*% w
counts <- exp(logits)
probs <- counts/rowSums(counts)

nlls <- rep(0, 5)

for (i in seq_len(5)) {
  x <- xs[i]
  y <- ys[i]
  print(sprintf("bigram example %s: %s%s, (indexes: %s,%s)", i, itos(x), itos(y), x, y))
  print(sprintf("Input to the neural net %s", x))
  print(sprintf("label (acutal next character): %s", y))
  p <- probs[i, y]
  print(sprintf("probabilty assigned by the net to the correct character: %s", p))
  logp <- log(p)
  print(sprintf("log likelihood: %s", logp))
  nll <- -logp
  print(sprintf("negativ log likelihood: %s", nll))
  nlls[i] <- nll
  }
print(sprintf("average negativ log likelihood: %s", mean(nlls)))


# Start with Neural Networks
library(tidyverse)
library(keras)
xs <- test_set[,1]
ys <- test_set[,2]

xs <- sapply(xs, FUN = stoi)
ys <- sapply(ys, FUN = stoi)

xs_one_hot <- matrix(0, nrow = length(xs), ncol = 27)
for (i in seq_len(length(xs))) {
  xs_one_hot[i, xs[i]] <- 1
}
ys_one_hot <- matrix(0, nrow = length(xs), ncol = 27)
for (i in seq_len(length(ys))) {
  ys_one_hot[i, ys[i]] <- 1
}

logits <- xs_one_hot %*% w
counts <- exp(logits)
probs <- (counts/apply(counts, 1, sum))
loss <- rep(NA, dim(probs)[1])

for(i in 1:dim(probs)[1]){
  loss[i] <- probs[i, ys[i]]
}
loss <- -mean(log(loss))
loss # with random weights

# Setup model with keras
linear_model <- keras_model_sequential() %>%
  layer_dense(units = 27, input_shape = c(27), activation = "linear", use_bias = F)  %>%
  layer_activation_softmax()

linear_model %>% compile( optimizer = "adam",
                          loss = "categorical_crossentropy"
                          )

linear_model_hist <- linear_model %>% fit(xs_one_hot, ys_one_hot,
                                          epochs = 2,
                                          callbacks = callback_early_stopping(
                                            monitor = "loss",
                                            patience = 4,
                                            verbose = 0,
                                            mode = "min",
                                            restore_best_weights = T))
linear_model_hist$metrics$loss[2]
loss_table

set.seed(2147483647)

test_names_keras <- c()
for (i in seq_len(5)) {
  test <- c()
  ix <- 1
  stop <- F
  while(!stop){
    xenc <- xs_one_hot[ix,]
    p_multinomial <- linear_model(t(xenc))
    ix <- which(rmultinom(n = 1, 1, as.numeric(p_multinomial)) == 1)
    if( ix == 1) break
    test <- paste(test,itos(ix), sep = "")
  }
  test_names_keras <- c(test_names_keras, test)
}
test_names_keras
test_names

rbind(round(as.numeric(linear_model(t(xs_one_hot[1,]))),2),
round(P[1,],2))


# Now torch approach starts
library(torch)
library(luz)

xss <- torch_tensor(xs_one_hot)
yss <- torch_tensor(ys_one_hot)
# initalize random weights
w <- matrix(rnorm(27*27), ncol = 27)
# activate requires_grad um manuelles training durchzuführen 
w <- torch_tensor(w, requires_grad = T)
logits <- torch_matmul(xss, w)
counts <- torch_exp(logits)
probs <- counts/torch_sum(counts, dim = 2, keepdim = T)

torch_sum(probs, dim = 2, keepdim = T) # worked because rowsums have to be 1

# complicated way :) 

-torch_mean(torch_log(torch_gather(probs, 2, 
             torch_tensor(matrix(ys, nrow = length(ys), byrow=TRUE),
                                          dtype = torch_int64()))))
#
loss <- nnf_nll_loss(probs$log(), ys)
loss
# loss= 3.79686

# Now backprop
loss$grad_fn
w$grad # undefined
loss$backward()
w$grad # not undefined anymore
w <-  torch_tensor(w$data() - 0.1*w$grad, requires_grad = T)

# Second iteration
logits <- torch_matmul(xss, w)
counts <- torch_exp(logits)
probs <- counts/torch_sum(counts, dim = 2, keepdim = T)

loss2 <- nnf_nll_loss(probs$log(), ys)
loss2

# loss = 3.76769
loss2 < loss # Backprop worked 


#Now not "by hand" anymore
# torch in keras way

linear <- nn_sequential( 
  nn_linear(in_features = 27, 27, bias = F),
  nn_softmax(dim = 2)) # dim = 2 um über rows zu bilden
# and also an optimizer


### network parameters ---------------------------------------------------------
linear$parameters

# for adam, need to choose a much higher learning rate in this problem
optimizer <- optim_adam(params = linear$parameters, lr = 0.08)

### training loop --------------------------------------------------------------

for (t in 1:200) {
  
  ### -------- Forward pass -------- 
  
  y_pred <- linear(xss)
  
  ### -------- compute loss -------- 
  loss <- nnf_nll_loss(y_pred$log(), ys)
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")
  
  ### -------- Backpropagation -------- 
  
  # Still need to zero out the gradients before the backward pass, only this time,
  # on the optimizer object
  optimizer$zero_grad()
  
  # gradients are still computed on the loss tensor (no change here)
  loss$backward()
  
  ### -------- Update weights -------- 
  
  # use the optimizer to update model parameters
  optimizer$step()
} 


# Now compare with results from before
chars_onehot <- diag(27)

# now we can compare with P table
#torch model
round(as.array(linear(chars_onehot)), 4)[1,]
round(P[1,], 4)
# keras model
# by hand
softmax <- function(x){
  counts <- exp(x)
  counts/rowSums(counts)
}
round(softmax(chars_onehot %*% get_weights(linear_model)[[1]]), 3)[1,] # self fitted
round(as.array(linear_model(chars_onehot)[1,]), 3)

generator <- torch_generator()
generator$current_seed()
generator$set_current_seed(2147483647L) 

test_names_torch <- c()
for (i  in seq_len(5)) {
  test <- c()
  ix <- 1
  stop <- F
  while(!stop){
    xenc <- torch_tensor(matrix(xs_one_hot[as.array(ix),], ncol = 27))
    p_multinomial <- linear(xenc)
    ix <- torch_multinomial(p_multinomial, num_samples = 1, replacement = T,
                            generator = generator)
    if( as.array(ix) == 1) break
    test <- paste(test, itos(as.array(ix)), sep = "")
  }
  test_names_torch <- c(test_names_torch, test)
  
}
test_names_torch


# With modular approach (tensorflow style)
net <- nn_module(
  "bigram_nn",
  initialize = function() {
    self$fc1 <- nn_linear(in_features = 27, out_features = 27, bias = F)
  },
  
  forward = function(x) {
    x %>%  self$fc1() 
  }
)

model <- net()
model$parameters

optimizer <- optim_adam(model$parameters, lr = 0.1)
train_losses <- c() 

for (epoch in 1:200) {
  optimizer$zero_grad()
  output <- model(xss)
  loss <- nnf_cross_entropy(output, ys)
 #loss <- nnf_nll_loss(output$log(),ys)
  loss$backward()
  optimizer$step()
  train_losses <- c(train_losses, loss$item())
  print(loss$item())
}
train_losses
model(xss) 

# dataloader Versuch für luz
source("create_torch_dataset.R")

try <- data.frame(ys, xs_one_hot)
# we can now initialize an instance of our dataset.
# for example
xs_dataset <- df_dataset(df = try, response_variable = "ys")
# now we can get an item with
xs_dataset$.getitem(1)

xs_dl <- dataloader(xs_dataset, batch_size = 64, shuffle = F, num_workers = 4)
batch <- dataloader_make_iter(xs_dl) %>% dataloader_next()

torch_manual_seed(123)
# With modular approach
net <- nn_module(
  "bigram_nn",
  
  initialize = function() {
    self$fc1 <- nn_linear(in_features = 27, out_features = 27, bias = F)
  },
  
  forward = function(x) {
    x %>%  self$fc1() %>% nnf_softmax(dim = 2)
  }
)

fitted <- net %>%
  setup(
    loss = function(y_hat, y_true) nnf_nll_loss(y_hat$log(), as.integer(y_true)),
    optimizer = optim_adam
  ) %>% set_opt_hparams(lr = 0.01) %>%
  fit(data = list(x = xss, y = ys))

