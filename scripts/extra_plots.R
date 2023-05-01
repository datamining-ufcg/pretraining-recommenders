library(tidyverse)
library(broom)
library(here)
library(scales)
library(gridExtra)
library(grid)
library(cowplot)
library(stringr)
library(ggthemes)

process <- function(file_ext) {
    ml_random <- read_csv(here::here(
        str_interp("results/target_${file_ext}/leakage_user_25m/leakage_transfer_${file_ext}_random.csv")
        ), col_types = "fffdddd")
    ml_pca <- read_csv(here::here(
        str_interp("results/target_${file_ext}/leakage_user_25m/leakage_transfer_${file_ext}_pca.csv")
        ), col_types = "fffdddd")
    ml_w2v <- read_csv(here::here(
        str_interp("results/target_${file_ext}/leakage_user_25m/leakage_transfer_${file_ext}_word2vec.csv")
        ), col_types = "fffdddd")

    ml_all <- rbind(
      ml_pca %>% filter(limit == 1) %>% mutate(source = "PCA"),
      ml_w2v %>% filter(limit == 1) %>% mutate(source = "Word2vec")
    )

    ml_all <- rbind(
      ml_random %>% filter(limit == 1) %>% mutate(source = "Random"),
      ml_all
    )

    ml_all <- rbind(
      ml_random %>% filter(limit == 0 & init == "random") %>% mutate(source = "Baseline", init = "Baseline"),
      ml_all
    )

    ml_all <- ml_all %>%
    mutate(scenario = if_else(
        source != "Baseline",
        paste(source, str_to_title(init), sep=" to "),
        source
    ))

    ml_all <- ml_all %>% filter(init == "random" || init == "Baseline")

    ml_all$scenario <- ml_all$scenario %>% replace_na("Baseline")
    ml_all$scenario <- as.factor(ml_all$scenario)

    min_val <- 0.75
    max_val <- if_else(file_ext == "1m", 0.85, 0.9)

    p_all <- ml_all %>% 
      ggplot(aes(x = fct_inorder(source), y=rmse_mean)) +
      geom_bar(position="dodge", stat="identity", fill="#619CFF") +
      geom_linerange(
          aes(ymin = rmse_mean - rmse_std, ymax = rmse_mean + rmse_std)
      ) +
      xlab("Source model initialized with") +
      ylab("RMSE") + 
      scale_y_continuous(limits=c(min_val, max_val), oob = rescale_none) +
      coord_flip() +
      theme_hc() +
      theme(legend.direction = "horizontal")

    return(p_all)
}

p_100k <- process("100k")
p_1m <- process("1m")
leg <- cowplot::get_legend(p_100k)

p_100k <- p_100k + theme(legend.position = "none") + ggtitle("ML100k")
p_1m <- p_1m + theme(
    legend.position = "none",
    axis.text.y=element_blank(), #remove x axis labels
    axis.ticks.y=element_blank(), #remove x axis ticks
) + ggtitle("ML1M")

lay <- rbind(
  c(1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2),
  c(1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2),
  c(1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2),
  c(1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2),
  c(1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2)
)
plots <- arrangeGrob(p_100k, p_1m, layout_matrix = lay)

ggsave(
  filename = here::here(
    str_interp("documents/dual_supervised.pdf")
  ),
  plots,
  width = 8,
  height = 4
)

local_minima <- function(path, filename, title, left_title=NULL) {
  fp <- str_interp("${path}/${filename}.csv")  
  df <- read_csv(here::here(fp), col_types = "ddd")
  
  left <- df %>% 
    ggplot(aes(x = alphas, y = rmse_train)) +
    geom_point(shape="x", size=1, color="blue") +
    geom_line(color="blue") +
    labs(x="", y="") +
    scale_x_continuous(breaks=c(0, 0.2, 0.4, 0.6, 0.8, 1)) +
    theme_bw() +
    theme(
      plot.margin = unit(c(0, 0.1, -0.5, 0), "cm")
    )
  
  right <- df %>% 
    ggplot(aes(x = alphas, y = rmse_test)) +
    geom_point(shape="x", size=1, color="darkgreen") +
    geom_line(color="darkgreen") +
    labs(x="", y="") +
    scale_x_continuous(breaks=c(0, 0.2, 0.4, 0.6, 0.8, 1)) +
    scale_y_continuous(position="right") +
    theme_bw() +
    theme(
      plot.margin = unit(c(0, 0, -0.5, 0.1), "cm")
    )
  
  my_plot <- arrangeGrob(
    left, right, ncol=2,
    top=textGrob(title, gp=gpar(fontsize=10, font=8)),
    left=textGrob(left_title, rot=90, gp=gpar(fontsize=10, font=8)),
    bottom=NULL, right=NULL
  )
}

# INTERPOLATE TARGET 2 TARGET
path <- "results/interpolation"
r1a <- local_minima(path, "100k_random2pca", "Random to PCA")
r1b <- local_minima(path, "100k_random2word2vec", "Random to Word2Vec")

r2a <- local_minima(path, "1m_random2pca", "Random to PCA")
r2b <- local_minima(path, "1m_random2word2vec", "Random to Word2Vec")

r3a <- local_minima(path, "10m_random2pca", "Random to PCA")
r3b <- local_minima(path, "10m_random2word2vec", "Random to Word2Vec")

r4a <- local_minima(path, "20m_random2pca", "Random to PCA")
r4b <- local_minima(path, "20m_random2word2vec", "Random to Word2Vec")

r5a <- local_minima(path, "25m_random2pca", "Random to PCA")
r5b <- local_minima(path, "25m_random2word2vec", "Random to Word2Vec")

ml_plus_lm <- arrangeGrob(
  r1a, r1b, r2a, r2b, r3a, r3b, r4a, r4b, r5a, r5b,
  nrow=5, ncol=2, bottom="\u03B7", left = "RMSE"
)

ggsave(
  filename = here::here(
    str_interp("documents/target2target_interpolation.pdf")
  ),
  ml_plus_lm,
  width=8,
  height=6,
  cairo_pdf
)

# INTERPOLATE SOURCE 2 TARGET
path <- "results/target_1m/leakage_user_25m"
row3 <- local_minima(path, "interpolate_random", "Random to Random")
row4 <- local_minima(path, "interpolate_pca", "PCA to Random")
row5 <- local_minima(path, "interpolate_word2vec", "Word2Vec to Random")

ml100k_lm <- arrangeGrob(
  row3, row4, row5,
  nrow=2, ncol=2, bottom="\u03B7", left = "RMSE",
  layout_matrix=rbind(c(1,1,2,2), c(NA,3,3,NA))
)

ggsave(
  filename = here::here(
    str_interp("documents/interpolation_25m.pdf")
  ),
  ml100k_lm,
  width=8,
  height=3,
  cairo_pdf
)
