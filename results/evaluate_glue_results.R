library("rjson")
library("tidyverse")

# ***********************************************************************************************
# **** NOTE: Please adjust the 'mountpoint' default argument of the function 'get_glue_score' ***
# ***********************************************************************************************

# If plots should be saved, overwrite this variable with desired folder
plot_outpath <- ""

# --------------------------------------------------------------------------------------------------
# 1. define helper functions
# --------------------------------------------------------------------------------------------------

# get all GLUE scores simultaneously from individual textfiles
# *** NOTE: CHANGE MOUNTPOINT DEFAULT ARGUMENT TO YOUR SPECIFIC FOLDER ***
get_glue_score<-function(model, version, mountpoint = "/Volumes/ru87tow/") {
  filenames<-Sys.glob(file.path(
    mountpoint, "fine_tuned", model, version, "glue", "*/eval*.txt"
  ))
  row = if(model=="gpt2") 1 else 2
  second_row <- !(grepl("QQP", filenames) | grepl("STS.B", filenames) | grepl("MRPC", filenames))
  scores<-sapply(filenames[second_row], function(x) read.table(x)[row,3])
  scores<-c(scores,sapply(filenames[!second_row], function(x) read.table(x)[row+1,3]))
  names(scores) <- str_match(names(scores), paste(version,"(.*?)/eval", sep="/"))[,2]
  scores
}

# calculate "model size" (non-embedding parameters)
calc_N <- function(n_layer,d_model) 12*n_layer*(d_model)^2
# calculate total number of parameters (i.e., with embedding parameters)
calc_all <- function(n_layer,d_model,n_ctx,n_vocab) calc_N(n_layer,d_model)+(n_vocab+n_ctx)*d_model
# calculate "true" number of FLOPs per token and forward pass (i.e., with context-dependent term)
calc_flops <- function(model,n_layer,d_model,n_ctx) {
  C <- NULL
  if (model == "gpt2") {
    C <- 2*calc_N(n_layer,d_model) + 2*n_layer*n_ctx*d_model
  } else {
    C <- 2*calc_N(n_layer,d_model) + 4*n_layer*n_ctx*d_model
  }
  return(C)
}
# calculate model size, true parameters, FLOPs at once 
calc_size_and_compute <- function(model, version) {
  specs <- strsplit(version, "_")[[1]]
  d_model <- as.integer(specs[1])
  n_layer <- as.integer(specs[2])
  n_ctx <- 512
  n_vocab <- 30000
  flops <- calc_flops(model, n_layer,d_model,n_ctx)
  model_size <- calc_N(n_layer,d_model)
  total_params <- calc_all(n_layer,d_model,n_ctx,n_vocab)
  out<-c(flops, model_size, total_params)
  names(out) <- c("flops", "model_size", "total_params")
  return(out)
}

# main function, which downloads all specified versions for a given model class
## and creates a table with most important model specifications
download_glue_table <- function(model, version){
  specifications <- strsplit(version, "_")
  
  glue_table <- data.frame(do.call("rbind", map(
    version, function(x) get_glue_score(model, x)
  )))
  task_names <- c("cola", "mnli", "mnli1", "qnli", "rte", "sst2", "stsb", "wnli", "mrpc", "qqp")
  if(model == "gpt2") task_names[c(2, 3)] <- c("mnli-m", "mnli-mm") else task_names[c(2, 3)] <- c("mnli-mm", "mnli-m")
  names(glue_table) <- task_names
  score_large_pair <- rowMeans(glue_table[,c("mnli-m","qnli","qqp")])
  glue_table$score_large_pair <- score_large_pair
  glue_table$embedding_dimension <- as.integer(sapply(specifications, function(x) x[1]))
  glue_table$number_layers <- as.integer(sapply(specifications, function(x) x[2]))
  glue_table$number_heads <- as.integer(sapply(specifications, function(x) x[3]))
  glue_table$feed_forward_dimension <- as.integer(sapply(specifications, function(x) x[4]))
  glue_table$num_epochs <- as.integer(sapply(specifications, function(x) x[5]))
  glue_table$batch_size <- as.integer(sapply(specifications, function(x) if(!is.na(tmp <- x[6])) tmp else 64))
  size_and_compute <- data.frame(
    do.call("rbind", map(version, function(x) calc_size_and_compute(model, x)
  )))
  res <- cbind(model, version, glue_table, size_and_compute)
  return(res)
}

# --------------------------------------------------------------------------------------------------
# 2. load the data
# --------------------------------------------------------------------------------------------------

# BERT: scaling embedding size H
glue_bert_embedding <- download_glue_table(
  model = "bert", 
  version = c("128_2_2_512_6", "192_2_2_768_6", "288_2_2_1152_6", "384_2_2_1536_6", "544_2_2_2176_6")
)

# BERT: scaling number of layers L
glue_bert_layer <- download_glue_table(
  model = "bert", 
  version = c("128_2_2_512_6", "128_5_2_512_6", "128_10_2_512_6", "128_18_2_512_6", "128_36_2_512_6")
)

# BERT: scaling embedding size H & number of layers L
glue_bert_embedding_layer <- download_glue_table(
  model = "bert", 
  version = c("128_2_2_512_6", "204_7_2_816_6", "256_9_2_1024_6")
)

# BERT: scaling embedding size H & number of layers L
glue_bert_embedding_layer <- download_glue_table(
  model = "bert", 
  version = c("128_2_2_512_6", "204_7_2_816_6", "256_9_2_1024_6")
)

# BERT: scaling embedding size H & number of attention heads A
glue_bert_embedding_heads <- download_glue_table(
  model = "bert", 
  version = c("544_2_2_2176_6", "544_2_8_2176_6")
)

# BERT: effect of batch size and number of training steps
glue_bert_batchsize_steps <- download_glue_table(
  model = "bert", 
  version = c("256_9_2_1024_3", "256_9_2_1024_3_32")
)

# BERT: grid search (calculation of model size not working here due to 'grid_search' in version name, 
## but results are found in Appendix)
glue_bert_gridsearch <- download_glue_table(
  model = "bert", 
  version = c("grid_search/128_2_2_512_6", "grid_search/104_3_2_416_6", "grid_search/90_4_2_360_6")
)

# BERT: strategic scaling
glue_bert_strategic <- download_glue_table(
  model = "bert", 
  version = c("469_4_7_1876_5", "585_5_9_2340_5", "832_5_13_3328_5")
)

# -----------------------------

# RoBERTa: scaling embedding size H
glue_roberta_embedding <- download_glue_table(
  model = "roberta", 
  version = c("128_2_2_512_10", "192_2_2_768_10", "288_2_2_1152_10", "384_2_2_1536_10", "544_2_2_2176_10")
)

# RoBERTa: scaling number of layers L
glue_roberta_layer <- download_glue_table(
  model = "roberta", 
  version = c("128_2_2_512_10", "128_5_2_512_10", "128_10_2_512_10", "128_18_2_512_10", "128_36_2_512_10")
)

# RoBERTa: scaling embedding size H & number of layers L
glue_roberta_embedding_layer <- download_glue_table(
  model = "roberta", 
  version = c("128_2_2_512_10", "204_7_2_816_10", "256_9_2_1024_10")
)

# RoBERTa: scaling embedding size H & number of layers L
glue_roberta_embedding_layer <- download_glue_table(
  model = "roberta", 
  version = c("128_2_2_512_10", "204_7_2_816_10", "256_9_2_1024_10")
)

# RoBERTa: scaling embedding size H & number of attention heads A
glue_roberta_embedding_heads <- download_glue_table(
  model = "roberta", 
  version = c("544_2_2_2176_10", "544_2_8_2176_10")
)

# RoBERTa: effect of batch size and number of training steps
glue_roberta_batchsize_steps <- download_glue_table(
  model = "roberta", 
  version = c("256_9_2_1024_5", "256_9_2_1024_5_32")
)

# -----------------------------

# GPT-2: scaling embedding size H
glue_gpt2_embedding <- download_glue_table(
  model = "gpt2", 
  version = c("128_2_2_512_10", "192_2_2_768_10", "288_2_2_1152_10", "384_2_2_1536_10", "544_2_2_2176_10")
)

# GPT-2: scaling number of layers L
glue_gpt2_layer <- download_glue_table(
  model = "gpt2", 
  version = c("128_2_2_512_10", "128_5_2_512_10", "128_10_2_512_10", "128_18_2_512_10", "128_36_2_512_10")
)

# GPT-2: scaling embedding size H & number of layers L
glue_gpt2_embedding_layer <- download_glue_table(
  model = "gpt2", 
  version = c("128_2_2_512_10", "204_7_2_816_10", "256_9_2_1024_10")
)

# GPT-2: scaling embedding size H & number of attention heads A
glue_gpt2_embedding_heads <- download_glue_table(
  model = "gpt2", 
  version = c("544_2_2_2176_10", "544_2_8_2176_10")
)

# --------------------------------------------------------------------------------------------------
# 3. create plots
# --------------------------------------------------------------------------------------------------

# Comparison of different pre-training tasks when increasing number of layers

## Create table for layer scaling
table_layer_scaling <- rbind(
  data.frame(
    model_size = glue_bert_layer$model_size, 
    score_large_pair = glue_bert_layer$score_large_pair, 
    score_cola = glue_bert_layer$cola,
    score_sst2 = glue_bert_layer$sst2,
    type = "BERT-style",
    label = paste0("L=", glue_bert_layer$number_layers)
  ),
  data.frame(
    model_size = glue_roberta_layer$model_size, 
    score_large_pair = glue_roberta_layer$score_large_pair,
    score_cola = glue_roberta_layer$cola,
    score_sst2 = glue_roberta_layer$sst2,
    type = "RoBERTa-style", 
    label = paste0("L=", glue_roberta_layer$number_layers)
  ),
  data.frame(
    model_size = glue_gpt2_layer$model_size, 
    score_large_pair = glue_gpt2_layer$score_large_pair, 
    score_cola = glue_gpt2_layer$cola,
    score_sst2 = glue_gpt2_layer$sst2,
    type = "GPT-2-style",
    label = paste0("L=",glue_gpt2_layer$number_layers)
  )
)

## Combined Score (Three Largest GLUE Tasks):
pdf(file = paste0(plot_outpath,"number_layers.pdf"), 
    width = 6, 
    height = 5)

ggplot(table_layer_scaling, 
       aes(model_size/1000000, score_large_pair*100, 
           group = type, shape=type, color=type, label = label)) +
  geom_point(size = 2.5) +
  geom_line(size = 0.8) +
  geom_text(nudge_y = 1, check_overlap = TRUE, show.legend = FALSE, size = 3.5, fontface = "bold") +
  geom_label(aes(x=c(rep(0,14),6.3), y=c(rep(0,14),80), label = "A=2, H=128"),
             show.legend = FALSE, size = 5, color = "black")+
  scale_color_manual(name="System", values=c("#1B9E77", "#7570B3", "#E7298A")) +
  scale_shape_manual(name="System", values=c(15,17,19))+
  scale_y_continuous(breaks = scales::pretty_breaks(n = 10), limits = c(59, 80))+
  scale_x_continuous(breaks = scales::pretty_breaks(n = 7))+
  labs(x="Non-Embedding Parameters (Millions)", y="Average Score", colour = "System") +
  theme(legend.position = c(0.14, 0.85))

dev.off()


# Comparison of Embedding Size

## Create table for embedding size scaling
table_width_scaling <- rbind(
  data.frame(
    model_size = glue_bert_embedding$model_size, 
    score_large_pair = glue_bert_embedding$score_large_pair, 
    score_cola = glue_bert_embedding$cola,
    score_sst2 = glue_bert_embedding$sst2,
    type = "BERT-style",
    label = paste0("H=",glue_bert_embedding$embedding_dimension)
  ),
  data.frame(
    model_size = glue_roberta_embedding$model_size, 
    score_large_pair = glue_roberta_embedding$score_large_pair,
    score_cola = glue_roberta_embedding$cola,
    score_sst2 = glue_roberta_embedding$sst2,
    type = "RoBERTa-style", 
    label = paste0("H=",glue_roberta_embedding$embedding_dimension)
  ),
  data.frame(
    model_size = glue_gpt2_embedding$model_size, 
    score_large_pair = glue_gpt2_embedding$score_large_pair, 
    score_cola = glue_gpt2_embedding$cola,
    score_sst2 = glue_gpt2_embedding$sst2,
    type = "GPT-2-style",
    label = paste0("H=", glue_gpt2_embedding$embedding_dimension)
  )
)

## Combined Score (Three Largest GLUE Tasks):
pdf(file = paste0(plot_outpath,"embedding_size.pdf"), 
    width = 6, 
    height = 5)

ggplot(table_width_scaling, 
       aes(model_size/1000000, score_large_pair*100, 
           group = type, shape=type, color=type, label = label)) +
  geom_point(size = 2.5) +
  geom_line(size = 0.8) +
  geom_text(nudge_y = c(rep(1,5), -1, -1, -1, -1, 1, rep(1,5)), 
            show.legend = FALSE, alpha = 10, size = 3.5, fontface = "bold") +
  geom_label(aes(x=c(rep(0,14),6.5), y=c(rep(0,14),80), label = "A=2, L=2"),
             show.legend = FALSE, size = 5, color = "black")+
  scale_color_manual(name="System", values=c("#1B9E77", "#7570B3", "#E7298A")) +
  scale_shape_manual(name="System", values=c(15,17,19))+
  scale_y_continuous(breaks = scales::pretty_breaks(n = 10), limits = c(59, 80))+
  scale_x_continuous(breaks = scales::pretty_breaks(n = 7))+
  labs(x="Non-Embedding Parameters (Millions)", y="Average Score", colour = "System") +
  theme(legend.position = c(0.14, 0.85))

dev.off()


# comparison single vs. multi scaling, glue score RoBERTa
table_multi_comparison_roberta<- rbind(
  data.frame(
    model_size = glue_roberta_embedding_layer$model_size,
    score_large_pair = glue_roberta_embedding_layer$score_large_pair,
    type = "L & H",
    label = paste0("(L,H)=(",glue_roberta_embedding_layer$number_layers,",",
                   glue_roberta_embedding_layer$embedding_dimension,")")
  ),
  data.frame(
    model_size = glue_roberta_layer$model_size[-c(2,3)],
    score_large_pair = glue_roberta_layer$score_large_pair[-c(2,3)],
    type = "L",
    label = paste0("(L,H)=(",glue_roberta_layer$number_layers[-c(2,3)],",",
                   glue_roberta_layer$embedding_dimension[-c(2,3)],")")
  ),
  data.frame(
    model_size = glue_roberta_embedding$model_size[-c(2,3)],
    score_large_pair = glue_roberta_embedding$score_large_pair[-c(2,3)],
    type = "H",
    label = paste0("(L,H)=(",glue_roberta_embedding$number_layers[-c(2,3)],",",
                   glue_roberta_embedding$embedding_dimension[-c(2,3)],")")
  )
)

pdf(file = paste0(plot_outpath,"multi_roberta.pdf"), 
    width = 6, 
    height = 5)

ggplot(table_multi_comparison_roberta, 
       aes(model_size/1000000, score_large_pair*100, 
           group = type, shape=type, color=type, label = label)) +
  geom_point(size = 2.5) +
  geom_line(size = 0.8) +
  geom_text(nudge_y = c(100, 1, 1, -1.5, -0.8, -1.5 , rep(-1,3)),
            nudge_x = c(0, -0.5, 0, 100, 0.6, 0.08, 0, 0.3, 0),
            check_overlap = TRUE, show.legend = FALSE, size = 3.5, fontface = "bold") +
  geom_label(aes(x = c(6, rep(0,8)), y=c(60, rep(0,8)), label = "RoBERTa-Style, A=2"), 
             color = "black", size = 5)+
  scale_color_manual(name="Scaling Method", values=c("#1B9E77", "#7570B3", "#E7298A")) +
  scale_shape_manual(name="Scaling Method", values=c(15,17,19))+
  scale_y_continuous(breaks = scales::pretty_breaks(n = 10), limits = c(59, 80))+
  scale_x_continuous(breaks = scales::pretty_breaks(n = 7), limits = c(0,7.6))+
  labs(x="Non-Embedding Parameters (Millions)", y="Average Score", colour = "System") +
  theme(legend.position = c(0.137, 0.85))

dev.off()

# comparison single vs. multi scaling, glue score BERT
table_multi_comparison_bert<- rbind(
  data.frame(
    model_size = glue_bert_embedding_layer$model_size,
    score_large_pair = glue_bert_embedding_layer$score_large_pair,
    type = "L & H",
    label = paste0("(L,H)=(", glue_bert_embedding_layer$number_layers, ",", 
                   glue_bert_embedding_layer$embedding_dimension, ")")
  ),
  data.frame(
    model_size = glue_bert_layer$model_size[-c(2,3)],
    score_large_pair = glue_bert_layer$score_large_pair[-c(2,3)],
    type = "L",
    label = paste0("(L,H)=(",glue_bert_layer$number_layers[-c(2,3)], ",",
                   glue_bert_layer$embedding_dimension[-c(2,3)], ")")
  ),
  data.frame(
    model_size = glue_bert_embedding$model_size[-c(2,3)],
    score_large_pair = glue_bert_embedding$score_large_pair[-c(2,3)],
    type = "H",
    label = paste0("(L,H)=(",glue_bert_embedding$number_layers[-c(2,3)], ",",
                   glue_bert_embedding$embedding_dimension[-c(2,3)], ")")
  )
)

pdf(file = paste0(plot_outpath,"multi_bert.pdf"), 
    width = 6, 
    height = 5)

ggplot(table_multi_comparison_bert, 
       aes(model_size/1000000, score_large_pair*100, 
           group = type, shape=type, color=type, label = label)) +
  geom_point(size = 2.5) +
  geom_line(size = 0.8) +
  geom_text(nudge_y = c(-100, -0.8, -0.8, -100, -0.8 , -0.8, -0.8, -0.8, -0.8),
            nudge_x = c(0, 0.7, 0, 0, 0.7, 0, 0, 0.5,0),
            check_overlap = TRUE, show.legend = FALSE, size = 3.5, fontface = "bold") +
  geom_label(aes(x = c(6, rep(0,8)), y=c(60, rep(0,8)), label = "BERT-Style, A=2"), 
             color = "black", size = 5) +
  scale_color_manual(name="Scaling Method", values=c("#1B9E77", "#7570B3", "#E7298A")) +
  scale_shape_manual(name="Scaling Method", values=c(15,17,19))+
  scale_y_continuous(breaks = scales::pretty_breaks(n = 10), limits = c(59, 80))+
  scale_x_continuous(breaks = scales::pretty_breaks(n = 7), limits = c(0,7.6))+
  labs(x="Non-Embedding Parameters (Millions)", y="Average Score", colour = "System") +
  theme(legend.position = c(0.137, 0.85))

dev.off()
