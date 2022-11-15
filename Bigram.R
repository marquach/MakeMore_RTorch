# Bigram Model

#load data
names <- read.table(file = "names.txt")

# Bigramm model is based on two consequtives characters, so Bigram is 
# (in this case) on  character level.

# Use moving window of size two per word and then count them

test <- c(".", unlist(strsplit(names[1,1], split = "")), ".")

bigram = matrix(, ncol = 2)
for(j in seq_len(dim(names)[1])){
  
  name <- c(".", unlist(strsplit(names[j,1], split = "")), ".")
  
  for(i in 2:length(name)){
    bigram <- rbind(bigram,c(name[i-1],name[i]))
  }
}

bigram <- bigram[-1,]
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
  N[entry_index[1], entry_index[2]] <- bigram_table[i]
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

p <- (N)/rowSums(N)

itos <- function(x) chars[x]
  
set.seed(2147483647)
test_names <- c()
for (i in seq_len(5)) {
test <- c()
ix <- 1
stop <- F
while(!stop){
p_multinomial <- p[ix,]
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
    prob <- p[entry_index[1], entry_index[2]]
    logprob <- log(prob)
    log_likelihood <- log_likelihood + logprob
    n <- n + 1
}
nll <- -log_likelihood
nll/n

# Create training set of bigrams (x,y)
xs <- test_set[,1]
ys <- test_set[,2]

xs <- sapply(xs, FUN = stoi)
ys <- sapply(ys, FUN = stoi)

length(xs)
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
loss

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

set.seed(2147483647)
test_names

test_names_nn <- c()
for (i in seq_len(5)) {
  test <- c()
  ix <- 1
  stop <- F
  while(!stop){
    xenc <- xs_one_hot[ix,]
    logits <- xenc %*% get_weights(linear_model)[[1]]
    p_multinomial <- softmax(logits)
    ix <- which(rmultinom(n = 1, 1, p_multinomial) == 1)
    if( ix == 1) break
    test <- paste(test,itos(ix), sep = "")
  }
  test_names_nn <- c(test_names_nn, test)
}
test_names_nn

