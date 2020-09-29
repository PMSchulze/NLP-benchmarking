install.packages("rjson")
library("rjson")
# ------------------------------------------------------------------------

# read json file with PTB sample data:
PTB <- fromJSON(file = "./desktop/Masterarbeit/data/PTB.json") 

PTB <- sapply(PTB, function(x) x[[1]]) # extract the text (drop tags)
n <- length(PTB) # store number of sentences
idx_train <- sample(1:n, 0.8*n) # sample indices of training sentences
train <- PTB[idx_train] # create training set
test <- PTB[setdiff(1:n,idx_train)] # create test set

# store training and test data 
# newline is added after each sentence, ...
# ...such that LineByLineTextDataset interprets them as documents
cat(train, file = "./desktop/Masterarbeit/data/ptb_train.txt", sep = "\n")
cat(test, file = "./desktop/Masterarbeit/data/ptb_test.txt", sep = "\n")
