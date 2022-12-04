# Not so cute examples
# In dem Skript werden verschiedene Modelle mittels Torch implementiert
# Zur Kontrolle werden die R internen Funktionen genutzt.
# lm mit lm(), glm mit glm() und (g)am mit gam()

# Schwierigkeiten können beim Erstellen der Z Matrix der GAM auftreten. Ich
# habe hier auf den model.matrix() Aufruf zurückgegriffen. 
# splines mit splinedesign() bzw. mgcv mit cSplineDes()
# sollten eher verwendet werden.

# Start LM 
# data: hitters
# 1. linear model 
library(ISLR2)
hitters <- na.omit(Hitters)
n <- nrow(hitters)
set.seed(13)
ntest <- trunc(n / 3)
testid <- sample(1:n, ntest)

lfit <- lm(Salary ~ ., data = hitters[-testid, ])

terms(lfit)
?model.response
x_hitters_train <- model.matrix(lfit)
y_hitters_train <- lfit$model$Salary

library(torch)
x_hitters_train_torch <- torch_tensor(x_hitters_train)
y_hitters_train_torch <- torch_tensor(y_hitters_train)
linear <- nn_sequential( 
  nn_linear(in_features = ncol(x_hitters_train), 1, bias = F)
  )

# and also need an optimizer
# for adam, need to choose a much higher learning rate in this problem
optimizer <- optim_adam(params = linear$parameters, lr = 0.05)

### network parameters ---------------------------------------------------------
linear$parameters # random 

### training loop --------------------------------------------------------------
for (t in 1:3000) {
  ### -------- Forward pass -------- 
  
  y_pred <- linear(x_hitters_train_torch)
  
  ### -------- compute loss -------- 
  loss <- nnf_mse_loss(y_pred$view(nrow(x_hitters_train_torch)),
                       y_hitters_train_torch)
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
linear$parameters 
rbind(as.array(linear$parameters$`0.weight`),coef(lfit))

mean((y_hitters_train-lfit$fitted.values)^2) # LM loss: 92693.9
loss # NN loss nach paar durchläufen 92710.2

# With more modular approach (tensorflow style)
net <- nn_module(
  "linear_nn",
  initialize = function() {
    self$fc1 <- nn_linear(in_features = ncol(x_hitters_train_torch),
                          out_features = 1, bias = F)
  },
  
  forward = function(x) {
    x %>%  self$fc1() 
  }
)


model <- net()
model$parameters
optimizer <- optim_adam(model$parameters, lr = 0.05)
train_losses <- c() 

for (epoch in 1:7500) {
  optimizer$zero_grad()
  output <- model(x_hitters_train_torch)
  loss <- nnf_mse_loss(output$view(176), y_hitters_train_torch)
  loss$backward()
  optimizer$step()
  train_losses <- c(train_losses, loss$item())
  if (t %% 10 == 0)
    cat("Epoch: ", epoch, "   Loss: ", loss$item(), "\n")
}
model$parameters
coef(lfit)
loss

loss # 92697.5 with training several more times
mean((y_hitters_train-lfit$fitted.values)^2) # LM loss
# 92693.9
# almost same

# LASSO

lasso_model <- glmnet::glmnet(x = x_hitters_train[,-1], y = y_hitters_train, lambda = 0.1)
coef(lasso_model)

linear <- nn_sequential( 
  nn_linear(in_features = ncol(x_hitters_train), 1, bias = F)
)

# and also need an optimizer
# for adam, need to choose a much higher learning rate in this problem
optimizer <- optim_adam(params = linear$parameters, lr = 0.05)

### network parameters ---------------------------------------------------------
linear$parameters # random 

### training loop --------------------------------------------------------------
for (t in 1:3000) {
  ### -------- Forward pass -------- 
  
  y_pred <- linear(x_hitters_train_torch)
  
  ### -------- compute loss -------- 
  loss <- nnf_mse_loss(y_pred$view(nrow(x_hitters_train_torch)),
                       y_hitters_train_torch) + linear$parameters$`0.weight`$abs()$sum()
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
linear$parameters 
as.array(linear$parameters$`0.weight`)
coef(lasso_model)
mean((y_hitters_train-predict(lasso_model, x_hitters_train[,-1]))^2) +
  sum(abs(lasso_model$beta)) # LASSO loss: 92962.42
loss # NN loss nach paar durchläufen 93131.1



# GLM; logistic regression
# Logistic regression example
data <- ISLR::Default
#make this example reproducible
set.seed(42)
#Use 70% of dataset as training set and remaining 30% as testing set
sample <- sample(c(TRUE, FALSE),
                 nrow(data), replace=TRUE, prob=c(0.7,0.3))
train <- data[sample, ]
test <- data[!sample, ]  

# normalize train, test otherwise will not work
train[c(3,4)] <- scale(train[c(3,4)])

#fit logistic regression model
glm_model <- glm(default~., family="binomial", data=train)
#view model summary
round(coef(glm_model), 3)

x_glm_train <- model.matrix(glm_model)
y_glm_train <- glm_model$y

x_glm_train_torch <- torch_tensor(x_glm_train)
y_glm_train_torch <- torch_tensor(y_glm_train)


# With more modular approach (tensorflow style)
net <- nn_module(
  "logisticRegression_nn",
  initialize = function() { # hier x um hardcode zu umgehen
    self$fc1 <- nn_linear(in_features = 4, # hardcode bäh 
                          out_features = 1, bias = F) # Bias=F weil model.matrix
  },
  
  forward = function(x) {
    x %>%  self$fc1() %>% nnf_sigmoid()
  }
)

model <- net()
model$parameters
optimizer <- optim_adam(model$parameters, lr = 0.05)
train_losses <- c() 
for (epoch in 1:2000) {
  optimizer$zero_grad()
  output <- model(x_glm_train_torch)
  loss <- nnf_binary_cross_entropy(output$view(6979),
                                   y_glm_train_torch)
  loss$backward()
  optimizer$step()
  train_losses <- c(train_losses, loss$item())
  if (t %% 10 == 0)
    cat("Epoch: ", epoch, "   Loss: ", loss$item(), "\n")
}
round(as_array(model$parameters$fc1.weight), 3)
round(coef(glm_model), 3)
loss # loss after first round: 0.07333746
nnf_binary_cross_entropy(torch_tensor(glm_model$fitted.values),
                         y_glm_train_torch)
# GLM loss: 0.0733374

# jetzt GAM B-Spline

data(mcycle, package = 'MASS')
ggplot(mcycle, aes(x = times, y = accel)) +
  geom_point() +
  labs(x = "Miliseconds post impact", y = "Acceleration (g)",
       title = "Simulated Motorcycle Accident",
       subtitle = "Measurements of head acceleration")

m1 <- gam(accel ~ s(times), data = mcycle, family = gaussian)
gam_splines <- model.matrix(m1) # Hier ien bisschen getrickst
dim(gam_splines)

# With more modular approach (tensorflow style)
net <- nn_module(
  "spline_nn",
  initialize = function() {
    self$fc1 <- nn_linear(in_features = 2,
                          out_features = 1, bias = F)
  },
  
  forward = function(x) {
    x %>%  self$fc1() 
  }
)

spline_approach <- net()
# and also need an optimizer
# for adam, need to choose a much higher learning rate in this problem
optimizer <- optim_adam(params = spline_approach$parameters, lr = 0.08)

### network parameters ---------------------------------------------------------
spline_approach$parameters # random 
times_torch <- torch_tensor(gam_splines)
accel_torch <- torch_tensor(mcycle$accel)
### training loop --------------------------------------------------------------
for (t in 1:3000) {
  ### -------- Forward pass -------- 
  
  y_pred <- spline_approach(gam_splines)
  ### -------- compute loss -------- 
  loss <- nnf_mse_loss(y_pred$view(133),
                       accel_torch) +
    0.01*spline_approach$parameters$fc1.weight$abs()$sum() 
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
spline_approach$parameters 
coef(m1)

loss
mean((m1$fitted.values - mcycle$accel)^2)

plot(mcycle$times, mcycle$accel)
lines(mcycle$times , spline_approach(gam_splines), col = "green")
lines(mcycle$times, m1$fitted.values, col = "red")

# Beispiel aus mgcv für spline design matrix
## create some x's and knots...
n <- 200
x <- 0:(n-1)/(n-1);
k <- 0:5/5
X <- cSplineDes(x,k)

plot(x,X[,1],type="l"); for (i in 2:5) lines(x,X[,i],col=i)
## check that the ends match up....


