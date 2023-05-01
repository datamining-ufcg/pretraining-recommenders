library(tidyverse)
library(broom)
library(here)
library(scales)
library(gridExtra)
library(grid)
library(cowplot)
library(stringr)
library(ggthemes)

## Loading Data

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
file_ext <- args[2]

## Helper functions

rename_fn <- function(df, new_method) {
  new_df <- df %>% dplyr::mutate(method = new_method)
  new_df
}

load_and_rename <- function(path, filename, new_method) {
  ctypes <- "fffdddd"
  dataset <- readr::read_csv(
    here::here(paste(path, filename, sep = "/")),
    col_types = ctypes
  )
  dataset <- rename_fn(dataset, new_method)
  dataset
}

get_legend <- function(myggplot){
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}

## Script

prefix <- if_else(endsWith(path, "ml100k"), "transfer", "leakage_transfer")

ml100k_random <- load_and_rename(
  path,
  str_interp("${prefix}_${file_ext}_random.csv"),
  "Pretrained with Random"
)
ml100k_pca <- load_and_rename(
  path,
  str_interp("${prefix}_${file_ext}_pca.csv"),
  "Pretrained with PCA"
)
ml100k_w2v <- load_and_rename(
  path,
  str_interp("${prefix}_${file_ext}_word2vec.csv"),
  "Pretrained with Word2Vec"
)

ml100k <- rbind(rbind(ml100k_random, ml100k_pca), ml100k_w2v)

ml100k$method <- ordered(
  ml100k$method,
  levels = c(
    "Pretrained with Random",
    "Pretrained with PCA",
    "Pretrained with Word2Vec"
  )
)

ml100k$init <- ordered(
  ml100k$init,
  levels = c("random", "pca", "word2vec")
)


# plotting

xlab_lines <- "Proportion of transferred item embeddings"

joint_ml100k <- ml100k %>%
  ggplot(aes(x = limit, y = rmse_mean, color = init)) +
  geom_line(linewidth = .5) +
  geom_point(size = 1) +
  facet_wrap(~method, scales = "free_x", strip.position = "bottom") +
  xlab(xlab_lines) +
  ylab("RMSE") + scale_color_brewer(palette = "Dark2") +
  labs(color = "Target Fallback Initialization") +
  theme_bw() +
  theme(legend.direction = "horizontal")

leg <- get_legend(joint_ml100k)

joint_ml100k <- joint_ml100k + theme(legend.position = "none")

jml100k <- arrangeGrob(joint_ml100k, leg, heights = c(10, 1))

ggsave(
  filename = here::here(path, str_interp("ml${file_ext}_scatter.pdf")),
  jml100k,
  width = 8,
  height = 3
)
