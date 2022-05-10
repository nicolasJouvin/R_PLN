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



