library(tidyverse)
library(here)
library(glue)
library(latex2exp)
library(reticulate)
source("comparison/stats-helper.R")

np <- import("numpy")
oc_dir <- here("evals/gram-winoground/openclip/")
# oc_files <- list.files(oc_dir)

## make human data
get_human_data_wg <- function(manifest_file = "assets/gram-winoground/manifest.csv",
                              data_file = "evals/gram-winoground/human.jsonl") {
  included_trials <- read_csv(manifest_file) |> 
    mutate(trial = as.numeric(str_extract(image1, "(?<=ex_)[0-9]+(?=_img)")) + 1) |> 
    pull(trial)
  human_data <- read_lines(data_file) |> 
    str_remove_all('\\{\\"label\\"\\: \\"') |> 
    str_replace_all('\\", \\"score\\"\\: ', ',') |> 
    str_remove_all('\\}') |> 
    I() |> 
    read_csv(col_names = c("label", "score")) |> 
    separate_wider_delim(label, delim = "_", names = c("trial", "text", "image")) |> 
    mutate(trial = as.numeric(trial) + 1,
           image = (str_sub(image, 2) |> as.numeric()) + 1,
           text = (str_sub(text, 2) |> as.numeric()) + 1,
           pair = glue("image{image}text{text}")) |> 
    select(-image, -text) |> 
    filter(trial %in% included_trials) |> 
    mutate(trial = sapply(trial, \(t) which(included_trials == t)))
}

human_data_wg <- get_human_data_wg()

## comparison fxn
compare_wg <- function(model_data, human_data) {
  human_data_by_text <- human_data |> 
    separate_wider_regex(cols = pair,
                         patterns = c(image = "(?:image[12])", 
                                      text = "(?:text[12])")) |> 
    pivot_wider(names_from = image,
                values_from = score) |> 
    mutate(rowsum = rowSums(across(starts_with("image"))),
           across(starts_with("image"), \(x) x / rowsum),
           trial = glue("{trial}_{text}")) |> 
    select(-rowsum)
  
  model_data_by_text <- model_data |> 
    pivot_longer(cols = starts_with("image"),
                 names_to = c("image", "text"),
                 names_pattern = "(image[12])(text[12])",
                 values_to = "score") |> 
    pivot_wider(names_from = image,
                values_from = score) |> 
    mutate(trial = glue("{trial}_{text}"))
  
  opt_kl = get_opt_kl(human_data_by_text, model_data_by_text)
  tibble(kl = opt_kl$objective,
         beta = opt_kl$solution,
         iters = opt_kl$iterations)
}

## comparisons for openclip
openclip_div_wg <- lapply(oc_files, \(ocf) {
  epoch <- str_match(ocf, "[0-9]+")[1] |> as.numeric()
  res <- np$load(here(oc_dir, ocf)) |> 
    as_tibble() |> 
    `colnames<-`(value = c("image1text1", "image2text1", "image1text2", "image2text2")) |> 
    mutate(trial = seq_along(image1text1))
  kls <- compare_wg(res, human_data_wg) |> 
    mutate(epoch = epoch)
}) |> bind_rows()

wg_oc <- ggplot(openclip_div_wg, aes(x = log(epoch), y = kl)) +
  geom_point() +
  geom_smooth(col = my_palette[5]) +
  scale_colour_continuous() +
  # theme_classic() +
  labs(x = "log(Epoch)",
       y = TeX("Model–human dissimilarity ($D^*_{KL}$)"))

ggsave("comparison/gram-wg-oc.png", 
       wg_oc, 
       width = 6.2, height = 4.2, units = "in")

# mod_wg <- lm(cor ~ log(epoch), data = openclip_cors) |> 
#   tidy()

## comparisons for other models
wg_files <- c(list.files("evals/gram-winoground", pattern = "*.npy"), "openclip/openclip_epoch_256.npy")

other_res_wg <- lapply(wg_files, \(wgf) {
  res <- np$load(here("evals/gram-winoground", wgf)) |> 
    drop()
  if (dim(res)[2] == 4) {
    res <- (res[,,1] - res[,,2])
  }
  res <- res |> 
    as_tibble() |> 
    `colnames<-`(value = c("image1text1", "image2text1", "image1text2", "image2text2")) |> 
    mutate(trial = seq_along(image1text1))
  acc <- res |> 
    mutate(correct = ((image1text1 > image1text2) + (image2text2 > image2text1))/2) |> 
    summarise(accuracy = mean(correct)) |> 
    pull(accuracy)
  kls <- compare_wg(res, human_data_wg) |> 
    mutate(model = wgf |> str_remove_all("wg_") |> str_remove_all(".npy"),
           accuracy = acc)
}) |> bind_rows() |> 
  mutate(model = str_replace_all(model, model_rename))

write_csv(other_res_wg, "other_res_wg.csv")

wg_all <- ggplot(other_res_wg, 
                 aes(x = accuracy, y = kl, shape = model)) + 
  geom_point() + 
  scale_shape_manual(values = c(16, 1, 17, 15, 18, 0, 2, 3)) +
  # theme_classic() +
  labs(x = "Accuracy",
       y = TeX("Model–human dissimilarity ($D^*_{KL}$)"),
       shape = "Model")

ggsave("comparison/gram-wg-all.png", 
       wg_all, 
       width = 6.2, height = 4.2, units = "in")

wg_feats <- other_res_wg |> 
  left_join(model_feats, by = join_by(model == Model)) |> 
  mutate(n_params = str_replace_all(`# params`, size_fix) |> as.numeric(),
         n_images = str_replace_all(`# images`, size_fix) |> as.numeric()) |> 
  summarise(accuracy = cor(kl, accuracy),
            size = cor(kl, log(n_params)),
            training = cor(kl, log(n_images)))

## item-level comparisons
item_res_wg <- lapply(wg_files, \(wgf) {
  res <- np$load(here("evals/gram-winoground", wgf)) |> 
    as_tibble()
  res <- res |> 
    `colnames<-`(value = c("image1text1", "image2text1", "image1text2", "image2text2")) |> 
    mutate(trial = seq_along(image1text1)) |> 
    pivot_longer(cols = starts_with("image"),
                 names_to = c("image", "text"),
                 names_pattern = "(image[12])(text[12])",
                 values_to = "score") |> 
    pivot_wider(names_from = image,
                values_from = score) |> 
    mutate(trial = glue("{trial}_{text}"))
  model_name = wgf |> str_remove_all("wg_") |> str_remove_all(".npy") |> 
    str_replace_all(model_rename)
  beta = other_res_wg |> filter(model == model_name) |> pull(beta)
  human_data_nested <- human_data_wg |> 
    separate_wider_regex(cols = pair,
                         patterns = c(image = "(?:image[12])", 
                                      text = "(?:text[12])")) |> 
    pivot_wider(names_from = image,
                values_from = score) |> 
    mutate(rowsum = rowSums(across(starts_with("image"))),
           across(starts_with("image"), \(x) x / rowsum),
           trial = glue("{trial}_{text}")) |> 
    select(-rowsum) |> 
    select(trial, starts_with("image"))
  kl <- get_mean_kl_img(human_data_nested, softmax_images(res, beta), return_distribs = TRUE) |> 
    # select(-distribs) |> 
    mutate(kl_z = scale(kl)[,1]) |> 
    mutate(model = model_name)
}) |> bind_rows()

item_dif_wg <- item_res_wg |> 
  separate_wider_delim(trial, delim = "_", names = c("trial", "text")) |> 
  group_by(trial) |> 
  summarise(kl_z = mean(kl_z))
  
