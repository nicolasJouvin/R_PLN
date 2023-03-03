library(purrr)
source("utils.r")
n = 7
K = 2
q = 4
p = 20

Y = matrix(rpois(n * p, 4), n, p)
M = torch_randn(c(n, q))
S = torch_randn(c(n, q))
Mu = torch_randn(c(K, q))
Tau = torch_randn(c(K,n))
Pi = torch_randn(c(K))
Pi = Pi / Pi$sum()
Lambda = torch_randn(c(K, q, q))
for (k in 1:K) Lambda[k,,] = Lambda[k,,] + Lambda[k,,]$t()
invLambda = Lambda$inverse()
C = torch_randn(c(p, q))

k = 2
torch_equal(M - Mu[k, NULL], M - Mu[k,])

centeredM = M - Mu[k, NULL]
centeredMk = Tau[k,]$t()[,NULL] * centeredM
centeredMk_bis = centeredM$multiply(Tau[k, NULL]$t())
centeredMk$t()$matmul(centeredM)
Tau * torch_log(Pi)[, NULL]

SrondS = S$multiply(S)
Tau[k,]$t()[, NULL] * SrondS
(Tau[k,]$t()[, NULL] * SrondS)$sum(dim=1)$diag()


MMtk = centeredMk$t()$mm(centeredM)
SSk = (Tau[k,]$t()[, NULL] * SrondS)$sum(dim=1)$diag()


cl = kmeans(Y, centers = K, nstart = 10)$cl
projY = Y %*% as.matrix(C) 
Mu = sapply(projY %>% as.data.frame() %>% split(cl), colMeans) %>% t() 
Mu = torch_tensor(Mu, requires_grad = TRUE)
Mu

Lambda = projY %>% as.data.frame() %>% split(cl) %>% map(cov) %>% simplify2array() %>% aperm(c(3, 1, 2))
Lambda[k,,]
dim(Lambda)


Pi = torch_tensor(table(cl) / n, requires_grad=TRUE)

Tau = torch_tensor(as_indicator(cl) %>% t())
Tau$sum(2)
print(Tau)
torch_sum(Tau * torch_log(Pi)[, NULL])
torch_nansum(Tau * torch_log(Tau)) 

# essaie broadcating
x = Tau[,,NULL]$mul(M[NULL,,])$divide(Tau$sum(2)[,NULL,NULL])
dim(x)
dim(x$sum(2)) == c(K, q)
