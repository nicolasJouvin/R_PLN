library(torch)
library(dplyr)
n = 3
X = torch_rand(c(n,n))

lower_tri_idxs = torch_tril_indices(n,n)
cat('Indexing should start at 1 in R torch but starts at ', as.numeric(lower_tri_idxs$min()), '\n')
# Thus this throws an error : X[lower_tri_idxs]

# I don't know the workaround, I tried the Pythonistic way
X[lower_tri_idxs + 1L]
# But this returns the copy of the tensor because R torch is
# coded avec les pieds 

# This R version of cbind(1:n, 1:n) doesn't work as expected either
# diag_idx =  torch_vstack(c(torch_arange(1, n+1, dtype = torch_long()),
#                            torch_arange(1, n+1, dtype = torch_long())))$t()
# Only workaround I found : create an nxn matrix with T/F...
diag_idx = matrix(as.logical(diag(n)), n,n)
X[diag_idx] = 2 # affection ok


lower_tri_idxs = matrix(FALSE, n, n)
lower_tri_idxs[(torch_tril_indices(n,n, -1) + 1L)$t() %>% as.matrix()] = TRUE
X[lower_tri_idxs] = -1
X



# In R --------------------------------------------------------------------

R_X = X %>% as.matrix()
R_X[cbind(1:n, 1:n)]
