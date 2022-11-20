# overall packages needed
library(torch)

################################################################################
########## 1. target: categorical classification with 3 classes
################################################################################
# data: iris 

#load and split
df <- datasets::iris
train_split <- 0.75
sample_indices <- sample(nrow(df)*train_split)

# convert df to matrix and create training and test data
x_train <- as.matrix(df[sample_indices, -5])
y_train <- as.numeric(df[sample_indices, 5])
x_test <- as.matrix(df[-sample_indices, -5])
y_test <- as.numeric(df[-sample_indices, 5])

# convert into tensors
x_train <- torch_tensor(x_train, dtype = torch_float())
y_train <- torch_tensor(y_train, dtype = torch_long())
x_test <- torch_tensor(x_test, dtype = torch_float())
y_test <- torch_tensor(y_test, dtype = torch_long())

# create model with 3 layers and relu as activation function
model <- nn_sequential(
  
  # Layer 1
  nn_linear(4, 8), 
  nn_relu(), 
  
  # Layer 2
  nn_linear(8, 16),
  nn_relu(),
  
  # Layer 3
  nn_linear(16,3)
)

# define cost and optimizer
criterion <- nn_cross_entropy_loss() # cross entropy for at least binary classficiation (we have 3 classes)
optimizer <- optim_adam(model$parameters, lr = 0.01) # lr=learning rate

epochs <- 200

# train neural net
# Train the net
for(i in 1:epochs){
  
  # setting the gradient to zero in each pass
  optimizer$zero_grad()
  
  y_pred = model(x_train)
  # calculating the error in the last layer
  loss = criterion(y_pred, y_train)
  # propagate the error through the network with backpropagation to calculate 
  # the error in each neuron
  loss$backward()
  # apply the gradients for optimzation
  optimizer$step()
  
  
  # Check Training
  if(i %% 10 == 0){
    
    winners = y_pred$argmax(dim=2)+1
    corrects = (winners == y_train)
    accuracy = corrects$sum()$item() / y_train$size()
    
    cat(" Epoch:", i,"Loss: ", loss$item()," Accuracy:",accuracy,"\n")
  }
  
}


################################################################################
########## 2. target: regression
################################################################################
# data: Gitters
# 1. linear model 
library(ISLR2)
Gitters <- na.omit(Hitters)
n <- nrow(Gitters)
set.seed(13)
ntest <- trunc(n / 3)
testid <- sample(1:n, ntest)

lfit <- lm(Salary ~ ., data = Gitters[-testid, ])
lpred <- predict(lfit, Gitters[testid, ])
with(Gitters[testid, ], mean(abs(lpred - Salary)))

# 2. LASSO
x <- scale(model.matrix(Salary ~ . - 1, data = Gitters))
y <- Gitters$Salary
library(glmnet)
cvfit <- cv.glmnet(x[-testid, ], y[-testid],
                   type.measure = "mae")
cpred <- predict(cvfit, x[testid, ], s = "lambda.min")
mean(abs(y[testid] - cpred))


# 3. Neuronales Netz
# aber hier stimmt das ergebnis noch nicht os ganz (selbst zusammengebastelt)
# Load data 
library(ISLR2)
Gitters <- na.omit(Hitters)
n <- nrow(Gitters)
set.seed(13)
ntest <- trunc(n / 3)
testid <- sample(1:n, ntest)

x_train <- as.matrix(Gitters[-testid, -19])
y_train <- as.numeric(Gitters[-testid, 19])
x_test <- as.matrix(Gitters[testid, -19])
y_test <- as.numeric(Gitters[testid, 19])

Gitters_data <- Gitters

# convert data into torch_tensor
prepare_Gitters_data = function(input) {
  
  input <- input %>%
    mutate(across(.fns = as.factor)) 
  
  target_col <- input$Salary %>% 
    as.double %>%
    as.matrix()
  
  categorical_cols <- input %>% 
    select(-Salary) %>%
    select(where(function(x) nlevels(x) != 2)) %>%
    mutate(across(.fns = as.integer)) %>%
    as.matrix()
  
  numerical_cols <- input %>%
    select(-Salary) %>%
    select(where(function(x) nlevels(x) == 2)) %>%
    mutate(across(.fns = as.integer)) %>%
    as.matrix()
  
  data_set <- torch_tensor(cbind(categorical_cols,numerical_cols), dtype=torch_float())
  list(data_set, target_col)
}

set.seed(123)
train_indices <- sample(1:nrow(Gitters_data), size = floor(0.8 * nrow(Gitters_data)))
valid_indices <- setdiff(1:nrow(Gitters_data), train_indices)

train_data <- Gitters[train_indices,]
test_data <- Gitters[valid_indices,]

train_torch <- prepare_Gitters_data(train_data)
test_torch <- prepare_Gitters_data(test_data)


# Model 

# single hidden layer with 50 hidden units and relu activation function
# with a drop out layer in which a random of 40% of 50 activations from the previous layer 
# are set to zero during each iteration of stochastic gradient descent
modnn <- nn_module(
  initialize = function(input_size) {
    self$hidden <- nn_linear(input_size, 50)
    self$activation <- nn_relu()
    self$dropout <- nn_dropout(0.4)
    self$output <- nn_linear(50, 1)
  },
  forward = function(x) {
    x %>% 
      self$hidden() %>% 
      self$activation() %>% 
      self$dropout() %>% 
      self$output()
  }
)

# control the fitting algorithm and minimize squared error loss
modnn <- modnn %>% 
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_rmsprop,
    metrics = list(luz_metric_mse())
  ) %>% 
  set_hparams(input_size = ncol(train_torch[[1]])) # specify arguments which should be passed to initialize method of modnn

# fit the model 
fitted <- modnn %>% 
  fit(
    data = list(train_torch[[2]],train_torch[[1]]),
    valid_data = list(test_torch[[2]], test_torch[[1]]),
    epochs = 5
  )

plot(fitted)

npred <- predict(fitted, test_torch[[2]])
mean(abs(test_torch[[1]] - npred))


################################################################################
########## 3. target: binary classification
################################################################################
# data: mushroom
# inspo zur datenkonvertierung bei strings und numerischen features 

library(torch)
library(purrr)
library(readr)
library(dplyr)
library(ggplot2)
library(ggrepel)

# get the data
download.file(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
  destfile = "agaricus-lepiota.data"
)

mushroom_data <- read_csv(
  "agaricus-lepiota.data",
  col_names = c(
    "poisonous",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-type",
    "ring-number",
    "spore-print-color",
    "population",
    "habitat"
  ),
  col_types = rep("c", 23) %>% paste(collapse = "")
) %>%
  # can as well remove because there's just 1 unique value
  select(-`veil-type`)

mushroom_dataset <- dataset(
  name = "mushroom_dataset",
  
  initialize = function(indices) {
    data <- self$prepare_mushroom_data(mushroom_data[indices, ])
    self$xcat <- data[[1]][[1]]
    self$xnum <- data[[1]][[2]]
    self$y <- data[[2]]
  },
  
  .getitem = function(i) {
    xcat <- self$xcat[i, ]
    xnum <- self$xnum[i, ]
    y <- self$y[i, ]
    
    list(x = list(xcat, xnum), y = y)
  },
  
  .length = function() {
    dim(self$y)[1]
  },
  
  prepare_mushroom_data = function(input) {
    
    input <- input %>%
      mutate(across(.fns = as.factor)) 
    
    target_col <- input$poisonous %>% 
      as.integer() %>%
      `-`(1) %>%
      as.matrix()
    
    categorical_cols <- input %>% 
      select(-poisonous) %>%
      select(where(function(x) nlevels(x) != 2)) %>%
      mutate(across(.fns = as.integer)) %>%
      as.matrix()
    
    numerical_cols <- input %>%
      select(-poisonous) %>%
      select(where(function(x) nlevels(x) == 2)) %>%
      mutate(across(.fns = as.integer)) %>%
      as.matrix()
    
    list(list(torch_tensor(categorical_cols), torch_tensor(numerical_cols)),
         torch_tensor(target_col))
  }
)

train_indices <- sample(1:nrow(mushroom_data), size = floor(0.8 * nrow(mushroom_data)))
valid_indices <- setdiff(1:nrow(mushroom_data), train_indices)

train_ds <- mushroom_dataset(train_indices)
train_dl <- train_ds %>% dataloader(batch_size = 256, shuffle = TRUE)

valid_ds <- mushroom_dataset(valid_indices)
valid_dl <- valid_ds %>% dataloader(batch_size = 256, shuffle = FALSE)

# Model: modularize
embedding_module <- nn_module(
  
  initialize = function(cardinalities) {
    self$embeddings = nn_module_list(lapply(cardinalities, function(x) nn_embedding(num_embeddings = x, embedding_dim = ceiling(x/2))))
  },
  
  forward = function(x) {
    embedded <- vector(mode = "list", length = length(self$embeddings))
    for (i in 1:length(self$embeddings)) {
      embedded[[i]] <- self$embeddings[[i]](x[ , i])
    }
    torch_cat(embedded, dim = 2)
  }
)

net <- nn_module(
  "mushroom_net",
  
  initialize = function(cardinalities,
                        num_numerical,
                        fc1_dim,
                        fc2_dim) {
    self$embedder <- embedding_module(cardinalities)
    self$fc1 <- nn_linear(sum(map(cardinalities, function(x) ceiling(x/2)) %>% unlist()) + num_numerical, fc1_dim)
    self$fc2 <- nn_linear(fc1_dim, fc2_dim)
    self$output <- nn_linear(fc2_dim, 1)
  },
  
  forward = function(xcat, xnum) {
    embedded <- self$embedder(xcat)
    all <- torch_cat(list(embedded, xnum$to(dtype = torch_float())), dim = 2)
    all %>% self$fc1() %>%
      nnf_relu() %>%
      self$fc2() %>%
      self$output() %>%
      nnf_sigmoid()
  }
)

cardinalities <- map(
  mushroom_data[ , 2:ncol(mushroom_data)], compose(nlevels, as.factor)) %>%
  keep(function(x) x > 2) %>%
  unlist() %>%
  unname()

num_numerical <- ncol(mushroom_data) - length(cardinalities) - 1

fc1_dim <- 16
fc2_dim <- 16

model <- net(
  cardinalities,
  num_numerical,
  fc1_dim,
  fc2_dim
)

device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"

model <- model$to(device = device)

# Training 
optimizer <- optim_adam(model$parameters, lr = 0.1)

for (epoch in 1:20) {
  
  model$train()
  train_losses <- c()  
  
  coro::loop(for (b in train_dl) {
    optimizer$zero_grad()
    output <- model(b$x[[1]]$to(device = device), b$x[[2]]$to(device = device))
    loss <- nnf_binary_cross_entropy(output, b$y$to(dtype = torch_float(), device = device))
    loss$backward()
    optimizer$step()
    train_losses <- c(train_losses, loss$item())
  })
  
  model$eval()
  valid_losses <- c()
  
  coro::loop(for (b in valid_dl) {
    output <- model(b$x[[1]]$to(device = device), b$x[[2]]$to(device = device))
    loss <- nnf_binary_cross_entropy(output, b$y$to(dtype = torch_float(), device = device))
    valid_losses <- c(valid_losses, loss$item())
  })
  
  cat(sprintf("Loss at epoch %d: training: %3f, validation: %3f\n", epoch, mean(train_losses), mean(valid_losses)))
}

# Evaluation
model$eval()

test_dl <- valid_ds %>% dataloader(batch_size = valid_ds$.length(), shuffle = FALSE)
iter <- test_dl$.iter()
b <- iter$.next()

output <- model(b$x[[1]]$to(device = device), b$x[[2]]$to(device = device))
preds <- output$to(device = "cpu") %>% as.array()
preds <- ifelse(preds > 0.5, 1, 0)

comp_df <- data.frame(preds = preds, y = b[[2]] %>% as_array())
num_correct <- sum(comp_df$preds == comp_df$y)
num_total <- nrow(comp_df)
accuracy <- num_correct/num_total
accuracy










































