library(FNN)



## SOMM function
SOMM <- function(df, n, K, a){
  # Step 1
  max_df <- apply(df[,names(df)!="class"], 2, max)
  min_df <- apply(df[,names(df)!="class"], 2, min)
  df_n <- df

  df_n[,names(df)!="class"] <- apply(df[,names(df)!="class"], 2, FUN = function(x)(if((max(x)-min(x))!=0){(x-min(x))/(max(x)-min(x))}else{x-min(x)}))
  
  A <- df_n[df_n$class!=a,names(df_n)!="class"]
  B <- df_n[df_n$class==a,names(df_n)!="class"]
  # Step 2
  U <- apply(B, 2, max)
  L <- apply(B, 2, min)
  # Step 3
  S <- data.frame(matrix(NA, 0, ncol = ncol(B)))
  names(S) <- names(B)
  Snew <- data.frame(matrix(NA, 0, ncol = ncol(B)))
  names(Snew) <- names(B)
  i <- 1
  while (i <= n) {
    for (j in 1:ncol(B)) {
      S[i, j] <- runif(1, L[j], U[j])
    }
    nn <- get.knnx(df_n[,names(df_n)!="class"], S[i,], k=K)
    nn_df <- df_n[nn$nn.index,]
    
    if(nrow(nn_df[nn_df$class==a,])==0){i <- i-1}else if(nn_df$class[1]==a){Snew[i,] <- S[i,]*(max_df - min_df) + min_df}else{
      B_index <- which(nn_df$class==a)[1]
      nn_A <- nn_df[(1:B_index-1),names(df)!="class"]
      nn_B <- nn_df[B_index,names(df)!="class"]
      disA <- apply(nn_A, 1, FUN = function(x) (dist(rbind(S[i,], x))))
      disB <- dist(rbind(S[i,], nn_B))
      dirB <- nn_B - S[i,]
      disAonB <- disA
      for (k in 1:length(disA)) {
        dirA <- nn_A[k,] - S[i,]
        disAonB[k] <- disA[k]*sum(dirA*dirB)/(norm(dirA, type = "2")*norm(dirB, type = "2"))
      }
      MdisAonB <- max(max(disAonB),0)
      xB <- runif(1, MdisAonB, disB)
      Snew[i,] <- (S[i,]+dirB*xB)*(max_df - min_df) + min_df #  sd_df + mu_df
    }
    i <- i + 1
  }
  
  Snew$class <- a
  return(Snew)
}

# df is data frame
# n is the number of synthetic samples to be generated 
# k is the number of nearest neighbours
# a is the minority class label



