library(torch)
library(R6)
library(ggplot2)
library(progress)
source(file = 'utils.r')
setOldClass("torch_tensor")

log_P_WgivenY <- function(Y, O, covariates, W,C,B_zero){
  q <- dim(W)[length(dim(W))]
  len <- length(dim(W))
  if (len == 2){
    CW <- torch_matmul(W,torch_transpose(C, 2,1))
  }
  else{
    CW <- torch_matmul(C$unsqueeze(1)$unsqueeze(2), W$unsqueeze(4))$squeeze() 
  }
  A <- O + CW + torch_matmul(covariates,B_zero)
  gaussian_term <- -q/2 *log(2*pi) - 1/2*torch_norm(W, dim = -1)**2
  poisson_term <- torch_sum(-torch_exp(A) + torch_multiply(A,Y)-log_stirling(Y), dim = -1)
  return(poisson_term + gaussian_term)
}



log_gaussian_density <- function(W, mu_p, Sigma_p){
  dimension <- tail(dim(W),1)
  log_const <- dimension/2*torch_log(2*pi) + torch_log(torch_det(Sigma_p))/2
  Wmoinsmu = W-mu_p$unsqueeze(1)
  inv_Sig = torch_inverse(Sigma_p)
  log_d = -1 / 2 * torch_matmul(torch_matmul(inv_Sig$unsqueeze(1),Wmoinsmu$unsqueeze(4))$squeeze()$unsqueeze(3),Wmoinsmu$unsqueeze(4))
  return(log_d$squeeze() - log_const)
}

sample_gaussians <- function(N_samples, mu, sqrt_Sigma){
  q <- dim(sqrt_Sigma)[2]
  W_p <- torch_matmul(sqrt_Sigma$unsqueeze(1), torch_randn(N_samples,1,q,1))$squeeze() + mu$unsqueeze(1)
  return(W_p)  
}
#d = 2L
#n = 200L
#p = 15L
#q = 5L
#N_samples = 200L
### Sampling some data ###
#O <-  torch_tensor(matrix(1,nrow = n, ncol = p))
#covariates <- torch_tensor(matrix(rnorm(n),nrow = n, ncol = d))
#true_Theta <- torch_tensor(matrix(rnorm(d*p),nrow = d, ncol = p))
#true_C <- torch_tensor(matrix(rnorm(p*q), nrow = p, ncol = q) )
#true_Sigma <- torch_matmul(true_C,torch_transpose(true_C, 2,1))
#B_zero <- torch_tensor(matrix(rnorm(n),nrow = d, ncol = p))
#Y <- sample_PLN(true_C,true_Theta,O,covariates, B_zero,ZI= FALSE)

full_C_from_Sigma <- function(Sigma,q){
  USV <- linalg_svd(Sigma)
  U <- USV[[1]]
  S <- USV[[2]]
  V <- USV[[3]]
  S[(q+1):S$shape[1]] <- 0   
  return(torch_multiply(U, torch_sqrt(S)))
  
}


C_from_Sigma <- function(Sigma, q){
  USV <- linalg_svd(Sigma)
  U <- USV[[1]]
  S <- USV[[2]]
  V <- USV[[3]]
  S[(q+1):S$shape[1]] <- 0   
  return(torch_multiply(U[,1:q], torch_sqrt(S[1:q])))
}

##### IMPS_PLN class ######
IMPS_PLN <- R6Class("IMPS_PLN",
              public = list(
                Y = NULL,
                O = NULL,
                covariates = NULL,
                q = NULL,
                n = NULL,
                p = NULL, 
                d = NULL, 
                C = NULL, 
                Theta = NULL,
                mode = NULL,
                Sigma_prop = NULL,
                sqrt_Sigma_prop = NULL, 
                const = NULL,
                samples = NULL,
                weights = NULL, 
                batch_grad_Theta = NULL,
                batch_grad_C = NULL,
                initialize = function(Y, O, covariates,q){
                  self$Y <- Y
                  self$O <- O 
                  self$covariates <- covariates
                  self$q <- q
                  self$p <- Y$shape[2]
                  self$n <- Y$shape[1]
                  self$d <- covariates$shape[2]
                  self$C <- torch_randn(self$p,self$q, requires_grad = TRUE)
                  self$Theta <- torch_randn(self$d,self$p, requires_grad = TRUE)
                  self$mode <- torch_randn(self$n,self$q, requires_grad = TRUE)
                },
                
                find_mode = function(N_iter_max, lr, eps = 0.001){
                  un_log_posterior <- function(W){
                    return(log_P_WgivenY(self$Y,self$O,self$covariates,W,self$C,self$Theta))
                  }
                optimizer = optim_rprop(self$mode, lr = lr)
                for (i in 1:N_iter_max){
                   optimizer$zero_grad()
                   loss = -torch_mean(un_log_posterior(self$mode))
                   loss$backward()
                   optimizer$step()
                  }
                },
                
                get_best_var = function(){
                  batch_matrix <- torch_matmul(self$C$unsqueeze(3),self$C$unsqueeze(2))$unsqueeze(1)
                  CW <- torch_matmul(self$C$unsqueeze(1),self$mode$unsqueeze(3))$squeeze()
                  common <- torch_exp(self$O + torch_matmul(self$covariates, self$Theta)+ CW)$unsqueeze(3)$unsqueeze(4)
                  prod = torch_multiply(batch_matrix, common)
                  # The hessian of the posterior
                  Hess_post = torch_sum(prod, dim=2) + torch_eye(self$q)
                  self$Sigma_prop = torch_inverse(Hess_post)
                  # Add a term to avoid non-invertible matrix.
                  eps = torch_diag(torch_full(c(self$q, 1), 1e-8)$squeeze())
                  self$sqrt_Sigma_prop = torch_cholesky(self$Sigma_prop + eps)
                },
                
                get_weights = function(){
                  self$samples <- self$samples#*0 + 1
                  log_f <- log_P_WgivenY(self$Y, self$O, self$covariates, self$samples, self$C, self$Theta)
                  log_g <- log_gaussian_density(self$samples, self$mode, self$Sigma_prop)
                  diff_log <- log_f - log_g 
                  self$const <- torch_max(diff_log, dim = 1)[[1]]
                  diff_log = torch_sub(diff_log, self$const)
                  self$weights <- torch_exp(diff_log)
                },
                grad_log_post_C = function(){
                  XB = torch_matmul(
                    self$covariates$unsqueeze(2),
                    self$Theta$unsqueeze(1))$squeeze()
                  CV = torch_matmul(
                    self$C$reshape(c(1, 1, self$p, 1, self$q)),
                    self$samples$unsqueeze(3)$unsqueeze(5)
                  )$squeeze()
                  Ymoinsexp = self$Y - torch_exp(self$O + XB + CV)
                  outer = torch_matmul(Ymoinsexp$unsqueeze(4), self$samples$unsqueeze(3))
                  return(outer)
                },
                get_grad_C = function(){
                  grad_log_post <- self$grad_log_post_C()
                  num = torch_multiply(self$weights$unsqueeze(3)$unsqueeze(4), grad_log_post)
                  denum = torch_sum(self$weights, dim=1)
                  batch_grad = torch_sum(num/(denum$unsqueeze(1)$unsqueeze(3)$unsqueeze(4)), dim=1)
                  self$batch_grad_C <- batch_grad 
                  return(torch_mean(batch_grad, dim = 1))
                },
                
                get_grad_Theta = function(){
                  grad_log_post <- self$grad_log_post_Theta()
                  numerator <- torch_sum(torch_multiply(self$weights$unsqueeze(3)$unsqueeze(4),
                                                        grad_log_post), dim=1)
                  denominator <- (torch_sum(self$weights,
                                            dim=1)$unsqueeze(2)$unsqueeze(3))
                  self$batch_grad_Theta <- numerator/denominator
                  return(torch_mean(self$batch_grad_Theta, dim = 1))
                },
                grad_log_post_Theta = function(){
                  XY = torch_matmul(self$covariates$unsqueeze(3),
                                    self$Y$unsqueeze(2))
                  XB = torch_matmul(self$covariates$unsqueeze(2),
                                    self$Theta$unsqueeze(1))$squeeze()
                  CV = torch_matmul(self$C$reshape(c(1, 1, self$p, 1, self$q)),
                                    self$samples$unsqueeze(3)$unsqueeze(5))$squeeze()
                  Xexp = torch_matmul(self$covariates$unsqueeze(1)$unsqueeze(4),
                                      torch_exp(self$O + XB + CV)$unsqueeze(3))
                  return(XY$unsqueeze(1) - Xexp)
                },
                
                fit = function(N_iter_max, lr = 0.1, acc = 0.01){
                  N_samples <- as.integer(1/acc)
                  model_optimizer <- optim_rprop(c(self$Theta, self$C), lr = lr)
                  
                  pb <- progress_bar$new(total=N_iter_max, width=60, clear=F,
                                         format = " (:spin) [:bar] :percent IN :elapsed")
                  for (j in 1:N_iter_max){
                    pb$tick()
                    self$find_mode(200,0.1)
                    self$get_best_var()
                    self$samples <- sample_gaussians(N_samples, self$mode, self$sqrt_Sigma_prop)
                    self$get_weights()
                    log_like <- torch_mean(torch_log(torch_mean(self$weights, dim = 1)) + self$const)
                    model_optimizer$zero_grad()
                    self$Theta$set_grad_(-self$get_grad_Theta()$detach())
                    self$C$set_grad_(-self$get_grad_C()$detach())
                    model_optimizer$step()
                    pr('log like', log_like)
                    pr('MSE', MSE(torch_matmul(self$C, torch_transpose(self$C, 2,1)) -true_Sigma))
                    }
                },
                get_one_p_theta = function(i){
                  return(torch_log(torch_mean(self$weights[,i])) + self$const[i])
                },
                get_p_theta = function(){
                  slow_log_like <- 0
                  for (i in 1:(self$n)){
                    slow_log_like = slow_log_like + self$get_one_p_theta(i)
                  }
                  return(slow_log_like/(self$n))
                },
                get_one_grad_log_Theta = function(i){
                  return((self$batch_grad_Theta)[i,,])
                },
                get_grad_log_Theta = function(){
                  grad_Theta <- torch_zeros(self$d, self$p)
                  for (i in (1:(self$n))){
                    grad_Theta = grad_Theta + self$get_one_grad_log_Theta(i)
                    #pr('added:', self$get_one_grad_log_Theta(i))
                  }
                  return(grad_Theta/(self$n))
                },
                get_one_outer_product_Theta = function(i){
                  vec_grad_i <- self$get_one_grad_log_Theta(i)$flatten()
                  #pr('res : ', torch_outer(vec_grad_i,vec_grad_i))
                  return(torch_outer(vec_grad_i,vec_grad_i))
                },
                outer_product_Theta = function(){
                  res <- torch_zeros((self$d)*(self$p),(self$d)*(self$p)) 
                  for (i in 1:(self$n)){
                    res <- res + self$get_one_outer_product_Theta(i)
                  }
                  return(res/(self$n))
                },
                grad_log_outer_product_Theta = function(){
                  vec_grad_Theta <- torch_flatten(self$batch_grad_Theta, start_dim = 2) 
                  return(torch_matmul(vec_grad_Theta$unsqueeze(3), vec_grad_Theta$unsqueeze(2)))
                },
                get_one_one_hessian = function(i,W){
                  # here for a double check of the vectorized version get_one_hessian
                  x_i = (self$covariates[i,])
                  kron_left <- torch_kron(torch_eye(self$p), x_i$unsqueeze(2))
                  kron_right <- torch_kron(torch_eye(self$p), x_i$unsqueeze(1))
                  diag <- torch_diag(torch_exp((self$O[i,]+ torch_matmul(x_i$unsqueeze(1), self$Theta) + torch_matmul(self$C, W$unsqueeze(2))$squeeze())$squeeze()))
                  Hess <- -torch_matmul(torch_matmul(kron_left, diag),kron_right)   
                  return(Hess)              
                },
                get_one_hessian = function(i,W){
                  # here for the vectorized version  get_hessian
                  x_i = (self$covariates[i,])
                  kron_left <- torch_kron(torch_eye(self$p), x_i$unsqueeze(2))
                  kron_right <- torch_kron(torch_eye(self$p), x_i$unsqueeze(1))
                  inside_exp <- (self$O[i,]+ torch_matmul(x_i$unsqueeze(1), self$Theta))$squeeze() + torch_matmul(self$C$unsqueeze(1), W$unsqueeze(3))$squeeze() 
                  diag <- torch_diag_embed(torch_exp(inside_exp))
                  vect_hess <- -torch_matmul(torch_matmul(kron_left$unsqueeze(1), diag), kron_right$unsqueeze(1))
                  return(vect_hess)
                },
                get_hessian = function(){
                  XB = torch_matmul(self$covariates$unsqueeze(2),self$Theta$unsqueeze(1))$squeeze()
                  kron_left <- torch_kron(torch_eye(self$p)$unsqueeze(1), self$covariates$unsqueeze(3))
                  kron_right <- torch_kron(torch_eye(self$p)$unsqueeze(1), self$covariates$unsqueeze(2))
                  CV <- torch_matmul(self$C$unsqueeze(1)$unsqueeze(2), self$samples$unsqueeze(4))$squeeze()
                  inside_exp <- (self$O + XB)$unsqueeze(1) + CV  
                  diag <- torch_diag_embed(torch_exp(inside_exp))
                  hess <- - torch_matmul(torch_matmul(kron_left$unsqueeze(1), diag), kron_right$unsqueeze(1))
                  return(hess)
                },
                compute_hess_p_theta = function(){
                  hess_log <- self$get_hessian()
                  vec_outer_prod <- self$grad_log_post_outer_product_Theta()
                  return(torch_mean(torch_multiply(hess_log + vec_outer_prod, self$weights$unsqueeze(3)$unsqueeze(4)), dim = 1)) 
                }, 
                compute_hess_log_p_theta = function(){
                  hess_p_theta <- self$compute_hess_p_theta()
                  outer_prod <- self$grad_log_outer_product_Theta()
                  pr('outer_prod ', outer_prod$shape)
                  pr('hess_p_theta', hess_p_theta$shape)
                  pr('const', self$const$shape)
                  pr('weights', torch_mean(self$weights, dim = 1) + self$const)
                  grad_p_theta_div_p_theta <- torch_div(hess_p_theta, (torch_mean(self$weights, dim =  1) + self$const)$unsqueeze(2)$unsqueeze(3))
                  hess_log_p_theta <- grad_p_theta_div_p_theta - outer_prod
                  return(torch_mean(hess_log_p_theta, dim = 1))
                },
                grad_log_post_outer_product_Theta = function(){
                  vec_grad_log_post <- self$grad_log_post_Theta()$flatten(start_dim = 3)
                  return(torch_matmul(vec_grad_log_post$unsqueeze(4), vec_grad_log_post$unsqueeze(3)))
                },
                log_p_theta = function(){
                  num <- self$compute_hess_p_theta()
                  denum <- torch_mean(self$weights, dim = 1)$unsqueeze(2)$unsqueeze(3)
                  return(torch_mean(num/denum -  self$grad_log_outer_product_Theta(), dim = 1))
                }
                
                
                
              ),
              private = list(
              )
      )


imps <- IMPS_PLN$new(Y,O,covariates,q)
imps$C <- torch_clone(true_C) + 0*torch_randn(true_C$shape)
imps$Theta <- true_Theta + 0*torch_randn(true_Theta$shape)
imps$fit(30L, acc = 0.008)
how_much()

vizmat(as.matrix(torch_matmul(imps$C, torch_transpose(imps$C, 2,1))))
vizmat(as.matrix(torch_matmul(true_C, torch_transpose(true_C,2,1))))

X <- how_much()
torch_float64(X)
torch_sum(torch_tensor((X< 1.96)))/60
torch_sum(X)
hess_log <- imps$compute_hess_log_p_theta()
C_hess <- C_from_Sigma(-hess_log, 30)


how_much <- function(){
  vec_imps_Theta <- imps$Theta$flatten()
  np <- vec_imps_Theta$shape[1]
  vec_Theta <- true_Theta$flatten()
  hess_log <- imps$compute_hess_log_p_theta()
  C_hess <- full_C_from_Sigma(-hess_log, 30)
  X <- torch_abs(torch_sqrt(imps$n)*torch_matmul(C_hess, vec_imps_Theta - vec_Theta))
  #percentage_inside <- torch_mean(X < 1.96)
  return(torch_sum(X<1.96)/np)
}

torch_mean(torch_matmul(C_hess, torch_transpose(C_hess, 2,1)) + hess_log)

q = 5L
Y <- torch_tensor(as.matrix(read.csv('Y.csv')))
O <- torch_tensor(as.matrix(read.csv('O.csv')))
covariates <- torch_tensor(as.matrix(read.csv('covariates.csv')))
true_Sigma <- torch_tensor(as.matrix(read.csv('true_5_Sigma.csv')))
true_Theta <- torch_tensor(as.matrix(read.csv('true_beta.csv')))
true_C <- C_from_Sigma(true_Sigma,q)
vizmat(as.matrix(torch_matmul(true_C, torch_transpose(true_C, 2,1))))



hess <- imps$compute_hess_p_theta()
vec_imps_Theta <- imps$Theta$flatten()
vec_Theta <- true_Theta$flatten()
vec_imps_Theta - vec_Theta
pr('MSE', MSE(torch_matmul(imps$C, torch_transpose(imps$C, 2,1)) - torch_matmul(C, torch_transpose(C, 2,1))))

imps$weights

test = torch_randn(20,20)
torch_eigvalsh(test)

torch_svd(test)
vec_imps_Theta
vec_Theta

Y <- sample_PLN(C,Theta,O,covariates, B_zero,ZI= FALSE)



W_vect_vect <- torch_randn(200,n,q)
covariates[1,]

imps$vect_outer_product_B_zero()
imps$outer_product_B_zero()

torch_mean(imps$batch_grad_B_zero, dim = 1) 
res<- torch_zeros(B_zero$shape)
for (i in 1:n){
  pr('i', i )
  #res = res + imps$get_one_grad_log_B_zero(i)
}
pr('res', res/n)
imps$get_grad_log_B_zero()


imps$get_grad_log_B_zero()
imps$B_zero$grad


outer <- imps$outer_product_beta()
outer
new_outer <- imps$vect_outer_product_beta()
new_outer
outer
imps$get_one_grad_log_B_zero(1)


vizmat(as.matrix(Sigma))


#### test with fixed simulated data #####


Y <- torch_tensor(as.matrix(read.csv('Y_test')))
O <- torch_tensor(as.matrix(read.csv('O_test')))
covariates <- torch_tensor(as.matrix(read.csv('cov_test')))
true_Sigma <- torch_tensor(as.matrix(read.csv('true_Sigma_test')))
true_C <- torch_cholesky(true_Sigma)
true_B_zero <- torch_tensor(as.matrix(read.csv('true_beta_test')))




###### Check if we find the right mode. Works if n = q = 1#######
imps <- IMPS_PLN$new(Y,O,covariates,q)
imps$find_mode(100, 0.01)
imps$mode
len <- 100 
W_ <- torch_tensor(((-len/2):(len/2-1))/100+ imps$mode)
W_ <- torch_transpose(W_$unsqueeze(2), 3,1)
res = list()
for (i in 1:len){
  pr('W', W_[i,,])
  pr('log', log_P_WgivenY(imps$Y,imps$O,imps$covariates,W_[i,,], imps$C, imps$B_zero)$squeeze()$item())
  res <- append(res,log_P_WgivenY(Y,O,covariates,W_[i,,], imps$C, imps$B_zero)$squeeze()$item())
}
plot(y = unlist(res), x = W_$squeeze())
########################


y = torch_randn(30,4,6)
y
y_vect <- y$flatten(start_dim = 2)
y_vect
torch_matmul(y_vect$unsqueeze(3), y_vect$unsqueeze(2))

new_y <- torch_reshape(y, c(4,6,30))

