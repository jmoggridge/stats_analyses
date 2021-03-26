# compute pca projection of dataframe (must be numeric df)
pca <- function(df) {
  # get new PC_j vector for df from: Eigenvector_j * ea. row in df
  get_pc <- function(E_j, df){
    apply(df, 1, function(df_i) as.numeric(t(E_j) %*% df_i))
  }
  E <- eigen(var(df))$vectors 
  # get PC columns
  scores <- apply(E, 2, get_pc, df = df)
  colnames(scores) <- c(paste0("pc", 1:dim(df)[2]))
  return(as.data.frame(scores))
}