library('reshape2')
library('torch')
vizmat <- function(x){
  m1 = min(x);   m2 = max(x)
  x <- melt(t(x))
  
  g1 = ggplot(data = x, aes(x=Var1, Var2, fill=value))  +  geom_tile(color = "white")  +  
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(m1, m2), space = "Lab",name="")  +  ylab("")  +   xlab("")  +
    theme(axis.text= element_text(size=22),  axis.title = element_text(size=28))  +
    theme(legend.key.height=unit(0.5, "in"), legend.text = element_text(size=22), plot.title=element_text(size=28,hjust=0.5))
  g1
}


pr <- function(str, var){
  print(str)
  print(var)
}


sample_PLN <- function(C,Theta,O,covariates, B_zero = None, ZI = FALSE){
  #Sample Poisson log Normal variables. If ZI is True, then the model will
  #be zero inflated.
  #The sample size n is the the first size of O, the number p of variables
  #considered is the second size of O. The number d of covariates considered
  #is the first size of beta.
  
  #Args:
  #  C: torch.tensor of size (p,q). The matrix c of the PLN model
  #  Theta: torch.tensor of size (d,p).
  #  0: torch.tensor. Offset, size (n,p)
  #  covariates : torch.tensor. Covariates, size (n,d)
  #  B_zero: torch.tensor of size (d,p), optional. If ZI is True,
  #      it will raise an error if you don't set a value. Default is None.
  #      ZI: Bool, optional. If True, the model will be Zero Inflated. Default is False.
  #  Returns :
  #      Y: torch.tensor of size (n,p), the count variables.
  #      Z: torch.tensor of size (n,p), the gaussian latent variables.
  #      ksi: torch.tensor of size (n,p), the bernoulli latent variables.
  
  n_ <- dim(covariates)[1] 
  q_ <- dim(C)[2]
  
  XB = torch_matmul(covariates$unsqueeze(2),Theta$unsqueeze(1))$squeeze()
  
  Z <- torch_matmul(torch_randn(n,q), torch_transpose(C, 2,1)) + XB + O
  parameter = torch_exp(Z)
  if (ZI == TRUE){
    ZI_covariates <- as.matrix(covariates,Theta)
    ksi <- torch_tensor(matrix(rbinom(n = n*p,prob = 1/(1+ exp(-as.matrix(ZI_covariates))), size = 1), nrow = n, ncol = p))
  }
  else{
    ksi <- 0
  }
  Y = torch_tensor((1-ksi)*matrix(rpois(n = n*p, lambda = as.matrix(parameter)), nrow = n, ncol = p))
  return(Y)
}

MSE <- function(t){
  return(mean(t**2))
}

log_stirling <- function(n_){
  n_ <- n_+ (n_==0)
  return(torch_log(torch_sqrt(2*pi*n_)) + n_*log(n_/exp(1))) 
}

C_from_Sigma <- function(Sigma, q){
  USV <- linalg_svd(Sigma)
  U <- USV[[1]]
  S <- USV[[2]]
  V <- USV[[3]]
  S[(q+1):S$shape[1]] <- 0   
  return(torch_multiply(U[,1:q], torch_sqrt(S[1:q])))
}

how_much <- function(Theta, true_Theta, hess, nb_minor,n){
  vec_Theta <- Theta$flatten()
  np <- vec_Theta$shape[1]
  vec_true_Theta <- true_Theta$flatten()
  C_hess <- full_C_from_Sigma(hess, nb_minor)
  X <- torch_abs(torch_sqrt(n)*torch_matmul(C_hess, vec_Theta - vec_true_Theta))
  return(torch_sum(X<1.96)/np)
}


full_C_from_Sigma <- function(Sigma,q){
  USV <- linalg_svd(Sigma)
  U <- USV[[1]]
  S <- USV[[2]]
  V <- USV[[3]]
  S[(q+1):S$shape[1]] <- 0   
  return(torch_multiply(U, torch_sqrt(S)))
}


## Initialisation for the PLN model

Poisson_reg <- function( Y, O, covariates, Niter_max=300,
                         tol=0.001, lr=0.005, verbose=FALSE){
  #Run a gradient ascent to maximize the log likelihood, using
  #      pytorch autodifferentiation. The log likelihood considered is
  #      the one from a poisson regression model. It is roughly the
  #      same as PLN without the latent layer Z.
  
  #      Args:
  #          Y: torch.tensor. Counts with size (n,p)
  #          0: torch.tensor. Offset, size (n,p)
  #          covariates: torch.tensor. Covariates, size (n,d)
  #          Niter_max: int, optional. The maximum number of iteration.
  #              Default is 300.
  #          tol: non negative float, optional. The tolerance criteria.
  #              Will stop if the norm of the gradient is less than
  #              or equal to this threshold. Default is 0.001.
  #          lr: positive float, optional. Learning rate for the gradient ascent.
  #              Default is 0.005.
  #          verbose: bool, optional. If True, will print some stats.
  
  #        Returns : None. Update the parameter beta. You can access it
  #              by calling self.beta.
  
  # Initialization of beta of size (d,p)
  beta = torch_randn(covariates$shape[2],
                     Y$shape[2],requires_grad=TRUE)
  optimizer = optim_rprop(c(beta), lr=lr)
  i = 0
  grad_norm = 2 * tol  # Criterion
  while (as.logical((i < Niter_max) & (grad_norm > tol))){
    optimizer$zero_grad()
    loss = -compute_poiss_loglike(Y, O, covariates, beta)
    loss$backward()
    optimizer$step()
    grad_norm = torch_norm(beta$grad)
    i = i + 1
    if (verbose){
      if(i %% 10 == 0){
        pr('log like : ', -loss)
        pr('grad_norm : ', grad_norm)
      }
    }
    
  }
  if (verbose){
    if (i < Niter_max){
      print(paste('Tolerance reached in ', as.character(i),' iterations'))
    }
    else{
      pr('Maxium number of iterations reached : ', Niter_max)   
      pr('grad norm', grad_norm)
    }
  }
  return(beta)
}


compute_poiss_loglike = function(Y, O, covariates, beta){
  # Compute the log likelihood of a Poisson regression
  # Matrix multiplication of X and beta.
  XB = torch_matmul(covariates$unsqueeze(2), beta$unsqueeze(1))$squeeze()
  # Returns the formula of the log likelihood of a poisson regression model.
  return(torch_sum(-torch_exp(O + XB) + torch_multiply(Y, O + XB)))
}

init_Sigma <- function(Y, O, covariates, beta){
  # Initialization for Sigma for the PLN model. Take the log of Y
  #   (careful when Y=0), remove the covariates effects X@beta and
  #   then do as a MLE for Gaussians samples.
  #   Args :
  #           Y: torch.tensor. Samples with size (n,p)
  #           0: torch.tensor. Offset, size (n,p)
  #           covariates: torch.tensor. Covariates, size (n,d)
  #           beta: torch.tensor of size (d,p)
  #   Returns : torch.tensor of size (p,p).
  #   
  # Take the log of Y, and be careful when Y = 0. If Y = 0,
  # then we set the log(Y) as 0.
  log_Y = torch_log(Y + (Y == 0) * exp(-2))
  # we remove the mean so that we see only the covariances
  log_Y_c = log_Y - torch_matmul(covariates$unsqueeze(2), beta$unsqueeze(1))$squeeze()
  # MLE in a Gaussian setting
  n = Y$shape[1]
  Sigma_hat = 1/(n-1)*(torch_mm(log_Y_c$t(),log_Y_c))
  return(Sigma_hat)
}


init_C <- function(Y, O, covariates, beta, q){
  # Inititalization for C for the PLN model. Get a first
  #   guess for Sigma that is easier to estimate and then takes
  #   the q largest eigenvectors to get C.
  #   Args :
  #       Y: torch.tensor. Samples with size (n,p)
  #       0: torch.tensor. Offset, size (n,p)
  #       covarites: torch.tensor. Covariates, size (n,d)
  #       beta: torch.tensor of size (d,p)
  #       q: int. The dimension of the latent space, i.e. the reducted dimension.
  #   Returns :
  #       torch.tensor of size (p,q). The initialization of C.
  # 
  # get a guess for Sigma
  Sigma_hat = init_Sigma(Y, O, covariates, beta)$detach()
  # taking the q largest eigenvectors
  C = C_from_Sigma(Sigma_hat, q)
  return(C)
}






