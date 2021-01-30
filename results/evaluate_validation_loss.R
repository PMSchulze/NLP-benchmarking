library("rjson")
library("tidyverse")

# ***********************************************************************************************
# **** NOTE: Please adjust the 'mountpoint' default argument of the function 'get_eval_loss *****
# **** and 'get_eval_loss_grid_search' **********************************************************
# ***********************************************************************************************

# If plots should be saved, overwrite this variable with desired folder
plot_outpath <- ""

# --------------------------------------------------------------------------------------------------
# 1. define helper functions
# --------------------------------------------------------------------------------------------------

# get all loss-logfiles simultaneously
# *** NOTE: CHANGE MOUNTPOINT DEFAULT ARGUMENT TO YOUR SPECIFIC FOLDER ***
get_eval_loss<-function(model, version, len, mountpoint = "/Volumes/ru87tow/") {
  path_logfile <- Sys.glob(file.path(
    mountpoint, "models", model, version, len, "checkpoint*", "trainer_state.json"
  ))
  path_eval_loss_final <- Sys.glob(file.path(
    mountpoint, "models", model, version, len, "eval_loss_final.txt"
  ))
  path_time <- Sys.glob(file.path(
    mountpoint, "models", model, version, len, "time.txt"
    ))
  time <- levels(read.table(path_time)[1,3])
  time_final <- (as.numeric(substr(time, 1,2))*3600 + 
    as.numeric(substr(time, 4,5))*60 + as.numeric(substr(time, 7,10)))
  logfile <- fromJSON(file = path_logfile)
  log <- data.frame(step = NULL, eval_loss = NULL)
  time_epoch <- 0 
  for (i in 1:length(logfile$log_history)){
    if("eval_loss" %in% names(logfile$log_history[[i]])){
      step <- logfile$log_history[[i]]$step
      eval_loss <- logfile$log_history[[i]]$eval_loss
      time_epoch <- time_epoch + logfile$log_history[[i]]$eval_runtime
      log <- rbind(log, data.frame(step = step, eval_loss = eval_loss, time = round(time_epoch)))
    }
  }
  eval_loss_final <- read.table(path_eval_loss_final)[1,2]
  log <- rbind(log, data.frame(step = log$step[nrow(log)]+log$step[1], 
                               eval_loss = eval_loss_final, time = round(time_final)))
  return(log)
}

# get all loss-logfiles of grid_search simultaneously
# *** NOTE: CHANGE MOUNTPOINT DEFAULT ARGUMENT TO YOUR SPECIFIC FOLDER ***
get_eval_loss_grid_search <- function(version, mountpoint = "/Volumes/ru87tow/"){
  path_logfile <- Sys.glob(file.path(
    mountpoint, "models", "bert", "grid_search",
    version, "short_range", "checkpoint*", "trainer_state.json"
  ))
  logfile <- fromJSON(file = path_logfile)
  logfile$log_history[[189]]$eval_loss
}

# ----------------------------------------------------------------------------------------
# BERT
# ----------------------------------------------------------------------------------------

# 1. SHORT RANGE

# ----------------------------------------------------------------------------------------

# obtain eval loss for different layers
bert_128_2_2_512_6 <- get_eval_loss("bert", "128_2_2_512_6", "short_range")
bert_128_5_2_512_6 <- get_eval_loss("bert", "128_5_2_512_6", "short_range")
bert_128_10_2_512_6 <- get_eval_loss("bert", "128_10_2_512_6", "short_range")
bert_128_18_2_512_6 <- get_eval_loss("bert", "128_18_2_512_6", "short_range")
bert_128_36_2_512_6 <- get_eval_loss("bert", "128_36_2_512_6", "short_range")

# obtain eval loss for different embedding sizes
bert_128_2_2_512_6 <- get_eval_loss("bert", "128_2_2_512_6", "short_range")
bert_192_2_2_768_6 <- get_eval_loss("bert", "192_2_2_768_6", "short_range")
bert_288_2_2_1152_6 <- get_eval_loss("bert", "288_2_2_1152_6", "short_range")
bert_384_2_2_1536_6 <- get_eval_loss("bert", "384_2_2_1536_6", "short_range")
bert_544_2_2_2176_6 <- get_eval_loss("bert", "544_2_2_2176_6", "short_range")

# obtain eval loss when scaling width + depth
bert_204_7_2_816_6 <- get_eval_loss("bert", "204_7_2_816_6", "short_range")
bert_256_9_2_1024_6 <- get_eval_loss("bert", "256_9_2_1024_6", "short_range")

# create tables for plotting
bert_embedding <- rbind(
  data.frame(bert_288_2_2_1152_6, d = 288, n_layer = 2), 
  data.frame(bert_384_2_2_1536_6, d = 384, n_layer = 2), 
  data.frame(bert_544_2_2_2176_6, d = 544, n_layer = 2)
)
bert_layer <- rbind(
  data.frame(bert_128_10_2_512_6, d = 128, n_layer = 10), 
  data.frame(bert_128_18_2_512_6, d = 128, n_layer = 18), 
  data.frame(bert_128_36_2_512_6, d = 128, n_layer = 36)
)
bert_single <- rbind(bert_embedding, bert_layer)
key <- paste0(bert_single$d, "_", bert_single$n_layer)
bert_single$key <- factor(key, levels = c("128_36", "128_18", "128_10", 
                                          "544_2", "384_2", "288_2"))
  
bert_single$lab_x <- rep(c(12700, 14000, 17700, 14000, 17000, 24000), each = 6)
bert_single$lab_y <- rep(c(4.05, 3.8, 3.5, 5, 4.35, 4), each = 6)
bert_single$label <- c("2.0M Parameters", rep("",5),
                       "3.5M Parameters", rep("",5),
                       "7.1M Parameters", rep("",5),
                       "2.0M Parameters", rep("",5),
                       "3.5M Parameters", rep("",5),
                       "7.1M Parameters", rep("",5))

bert_multiple <- rbind(
  data.frame(bert_204_7_2_816_6, d = 204, n_layer = 7), 
  data.frame(bert_256_9_2_1024_6, d = 256, n_layer = 9)
)
bert_multiple$lab_x <- rep(c(15000, 17700), each = 6)
bert_multiple$lab_y <- rep(c(3.8, 3.4), each = 6)
bert_multiple$label <- c("3.5M Parameters", rep("",5),
                       "7.1M Parameters", rep("",5))
key <- paste0(bert_multiple$d, "_", bert_multiple$n_layer)
bert_multiple$key <- factor(key, levels = c(key[12], key[1]))

# plot single-dimension scaling losss
pdf(file = paste0(plot_outpath, "bert_loss_single.pdf"), 
    width = 6, 
    height = 5)

ggplot(bert_single) +
  geom_line(aes(time, eval_loss, group = key, color = as.factor(key)), size = 1.25) +
  ylim(3,7) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 8), limits = c(0,26000)) +
  scale_color_manual(values=c("#890f4d", "#e7298a", "#ef75b3","#383563", "#7570B3", "#9e9bc9"), 
                     labels=c("A=2, H=288, L=2", "A=2, H=384, L=2", "A=2, H=544, L=2", 
                              "A=2, H=128, L=10", "A=2, H=128, L=18","A=2, H=128, L=36"),
                     breaks=c("288_2", "384_2", "544_2", "128_10", "128_18", "128_36")) +
  labs(x = "Wall Clock (Seconds)", y = "Validation Loss BERT-Style", colour = "Shape") +
  geom_text(aes(lab_x, lab_y, label=label, color = as.factor(key)), 
            size = 3, alpha = 10, fontface = "bold", show.legend = FALSE)+
  theme(legend.position = c(0.8, 0.7))

dev.off()

# plot multi-dimension scaling losss
pdf(file = paste0(plot_outpath, "bert_loss_multi.pdf"), 
    width = 6, 
    height = 5)
  
ggplot(bert_multiple) +
  geom_line(aes(time, eval_loss, group = key, color = as.factor(key)), size = 1.25) +
  ylim(3,7) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 8), limits = c(0,26000)) +
  scale_color_manual(values=c("#0e5741", "#1B9E77"), 
                     labels=c("A=2, H=204, L=7", "A=2, H=256, L=9"), 
                     breaks=c("204_7", "256_9")) +
  labs(x = "Wall Clock (Seconds)", y = "Validation Loss BERT-Style", colour = "Shape") +
  geom_text(aes(lab_x, lab_y, label=label, color = as.factor(key)), 
            size = 3, alpha = 10, fontface = "bold", show.legend = FALSE)+
  theme(legend.position = c(0.8, 0.7))

dev.off()

bert_469_4_7_1876_5 <- get_eval_loss("bert", "469_4_7_1876_5", "short_range")
bert_585_5_9_2340_5 <- get_eval_loss("bert", "585_5_9_2340_5", "short_range")
bert_832_5_13_3328_5 <- get_eval_loss("bert", "832_5_13_3328_5", "short_range")

bert_multiple_strategic <- rbind(
  data.frame(bert_469_4_7_1876_5, d = 469, n_layer = 4),
  data.frame(bert_585_5_9_2340_5, d = 585, n_layer = 5),
  data.frame(bert_832_5_13_3328_5, d = 832, n_layer = 5)
)
bert_multiple_strategic$lab_x <- c(rep(19000, 5), rep(26300, 5), rep(34000, 5))
bert_multiple_strategic$lab_y <- c(rep(3.25, 5), rep(3.05, 5), rep(2.8, 5))
bert_multiple_strategic$label <- c("10.6M Parameters", rep("",4), 
                                   "20.6M Parameters", rep("",4), 
                                   "41.6M Parameters", rep("",4))
key <- paste0(bert_multiple_strategic$d, "_", bert_multiple_strategic$n_layer)
bert_multiple_strategic$key <- factor(key, levels = c(key[13], key[8], key[1]))

# plot strategic scaling losss
pdf(file = paste0(plot_outpath, "bert_strategic.pdf"), 
    width = 6, 
    height = 5)

ggplot(bert_multiple_strategic) +
  geom_line(aes(time, eval_loss, group = key, color = as.factor(key)), size = 1.25) +
  ylim(2.5,5) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 8), limits = c(0,37000)) +
  scale_color_manual(values=c("#0e5741", "#1B9E77", "#25d9a4"), 
                     labels=c("A=7, H=469, L=4", "A=9, H=585, L=5", "A=13, H=832, L=5"),
                     breaks=c("469_4", "585_5", "832_5")) +
  labs(x = "Wall Clock (Seconds)", y = "Validation Loss BERT-Style", colour = "Shape") +
  geom_text(aes(lab_x, lab_y, label=label, color = as.factor(key)), 
            size = 3, alpha = 10, fontface = "bold", show.legend = FALSE)+
  theme(legend.position = c(0.8, 0.7))

dev.off()
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------

# 2. LONG RANGE

# ----------------------------------------------------------------------------------------

# obtain eval loss for different layers
bert_128_2_2_512_6_long <- get_eval_loss("bert", "128_2_2_512_6", "long_range")
bert_128_5_2_512_6_long <- get_eval_loss("bert", "128_5_2_512_6", "long_range")
bert_128_10_2_512_6_long <- get_eval_loss("bert", "128_10_2_512_6", "long_range")
bert_128_18_2_512_6_long <- get_eval_loss("bert", "128_18_2_512_6", "long_range")
bert_128_36_2_512_6_long <- get_eval_loss("bert", "128_36_2_512_6", "long_range")

# obtain eval loss for different embedding sizes
bert_128_2_2_512_6_long <- get_eval_loss("bert", "128_2_2_512_6", "long_range")
bert_192_2_2_768_6_long <- get_eval_loss("bert", "192_2_2_768_6", "long_range")
bert_288_2_2_1152_6_long <- get_eval_loss("bert", "288_2_2_1152_6", "long_range")
bert_384_2_2_1536_6_long <- get_eval_loss("bert", "384_2_2_1536_6", "long_range")
bert_544_2_2_2176_6_long <- get_eval_loss("bert", "544_2_2_2176_6", "long_range")

# obtain eval loss when scaling width + depth
bert_204_7_2_816_6_long <- get_eval_loss("bert", "204_7_2_816_6", "long_range")
bert_256_9_2_1024_6_long <- get_eval_loss("bert", "256_9_2_1024_6", "long_range")

# create tables for plotting
bert_embedding_long <- rbind(
  data.frame(bert_288_2_2_1152_6_long, d = 288, n_layer = 2), 
  data.frame(bert_384_2_2_1536_6_long, d = 384, n_layer = 2), 
  data.frame(bert_544_2_2_2176_6_long, d = 544, n_layer = 2)
)
bert_layer_long <- rbind(
  data.frame(bert_128_10_2_512_6_long, d = 128, n_layer = 10), 
  data.frame(bert_128_18_2_512_6_long, d = 128, n_layer = 18), 
  data.frame(bert_128_36_2_512_6_long, d = 128, n_layer = 36)
)
bert_single_long <- rbind(bert_embedding_long, bert_layer_long)
bert_single_long$key <- paste0(bert_single_long$d, "_", bert_single_long$n_layer)
bert_single_long$lab_x <- rep(c(6600, 7300, 8900, 7500, 10000, 15400), each = 6)
bert_single_long$lab_y <- rep(c(4.1, 3.87, 3.65, 4.6, 4.38, 4.05), each = 6)
bert_single_long$label <- c("2.0M Parameters", rep("",5),
                       "3.5M Parameters", rep("",5),
                       "7.1M Parameters", rep("",5),
                       "2.0M Parameters", rep("",5),
                       "3.5M Parameters", rep("",5),
                       "7.1M Parameters", rep("",5))
bert_multiple_long <- rbind(
  data.frame(bert_204_7_2_816_6_long, d = 204, n_layer = 7), 
  data.frame(bert_256_9_2_1024_6_long, d = 256, n_layer = 9)
)
bert_multiple_long$lab_x <- rep(c(7900, 9000), each = 6)
bert_multiple_long$lab_y <- rep(c(3.67, 3.25), each = 6)
bert_multiple_long$label <- c("3.5M Parameters", rep("",5),
                         "7.1M Parameters", rep("",5))
bert_multiple_long$key <- paste0(bert_multiple_long$d, "_", bert_multiple_long$n_layer)

# plot single-dimension scaling losss
pdf(file = paste0(plot_outpath, "bert_loss_single_long.pdf"), 
    width = 6, 
    height = 5)

ggplot(bert_single_long) +
  geom_line(aes(time, eval_loss, group = key, color = as.factor(key)), size = 1.25) +
  ylim(3,7) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 8), limits = c(0,18000)) +
  scale_color_manual(values=c("#890f4d", "#e7298a", "#ef75b3","#383563", "#7570B3", "#9e9bc9"), 
                     labels = c("A=2, H=128, L=10", "A=2, H=128, L=18","A=2, H=128, L=36",
                                "A=2, H=288, L=2", "A=2, H=384, L=2", "A=2, H=544, L=2")) +
  labs(x = "Wall Clock (Seconds)", y = "Validation Loss BERT-Style", colour = "Shape") +
  geom_text(aes(lab_x, lab_y, label=label, color = as.factor(key)), 
            size = 3, alpha = 10, fontface = "bold", show.legend = FALSE)+
  theme(legend.position = c(0.8, 0.7))

dev.off()

# plot multi-dimension scaling losss
pdf(file = paste0(plot_outpath, "bert_loss_multi_long.pdf"), 
    width = 6, 
    height = 5)

ggplot(bert_multiple_long) +
  geom_line(aes(time, eval_loss, group = key, color = as.factor(key)), size = 1.25) +
  ylim(3,7) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 8), limits = c(0,18000)) +
  scale_color_manual(values=c("#0e5741", "#1B9E77"), 
                     labels=c("A=2, H=204, L=7", "A=2, H=256, L=9")) +
  labs(x = "Wall Clock (Seconds)", y = "Validation Loss BERT-Style", colour = "Shape") +
  geom_text(aes(lab_x, lab_y, label=label, color = as.factor(key)), 
            size = 3, alpha = 10, fontface = "bold", show.legend = FALSE)+
  theme(legend.position = c(0.8, 0.7))

dev.off()


bert_469_4_7_1876_5_long <- get_eval_loss("bert", "469_4_7_1876_5", "long_range")
bert_585_5_9_2340_5_long <- get_eval_loss("bert", "585_5_9_2340_5", "long_range")
bert_832_5_13_3328_5_long <- get_eval_loss("bert", "832_5_13_3328_5", "long_range")

bert_multiple_strategic_long <- rbind(
  data.frame(bert_469_4_7_1876_5_long, d = 469, n_layer = 4),
  data.frame(bert_585_5_9_2340_5_long, d = 585, n_layer = 5),
  data.frame(bert_832_5_13_3328_5_long, d = 832, n_layer = 5)
)
bert_multiple_strategic_long$lab_x <- c(rep(9300, 5), rep(6300, 3), rep(8100, 3))
bert_multiple_strategic_long$lab_y <- c(rep(3.12, 5), rep(2.91, 3), rep(2.78, 3))
bert_multiple_strategic_long$label <- c("10.6M Parameters", rep("",4), 
                                   "20.6M Parameters", rep("",2), 
                                   "41.6M Parameters", rep("",2))
key <- paste0(bert_multiple_strategic_long$d, "_", bert_multiple_strategic_long$n_layer)
bert_multiple_strategic_long$key <- factor(key, levels = c("832_5", "585_5", "469_4"))

# plot strategic scaling losss
pdf(file = paste0(plot_outpath, "bert_loss_strategic_long.pdf"), 
    width = 6, 
    height = 5)

ggplot(bert_multiple_strategic_long) +
  geom_line(aes(time, eval_loss, group = key, color = as.factor(key)), size = 1.25) +
  ylim(2.5,5) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 8), limits = c(0,12000)) +
  scale_color_manual(values=c("#0e5741", "#1B9E77", "#25d9a4"), 
                     labels=c("A=7, H=469, L=4", "A=9, H=585, L=5", "A=13, H=832, L=5"),
                     breaks=c("469_4", "585_5", "832_5")) +
  labs(x = "Wall Clock (Seconds)", y = "Validation Loss BERT-Style", colour = "Shape") +
  geom_text(aes(lab_x, lab_y, label=label, color = as.factor(key)), 
            size = 3, alpha = 10, fontface = "bold", show.legend = FALSE)+
  theme(legend.position = c(0.8, 0.7))

dev.off()
# ----------------------------------------------------------------------------------------

# obtain validation loss for grid search
version_gridsearch <- c("104_3_2_416_6", "128_2_2_512_6", "46_16_2_184_6", "48_14_2_192_6",  
                        "52_12_2_208_6", "58_10_2_232_6", "64_8_2_256_6", "74_6_2_296_6", "90_4_2_360_6")
loss_gridsearch <- data.frame(
  version = version_gridsearch,
  eval_loss = map_dbl(version_gridsearch, get_eval_loss_grid_search)
)
