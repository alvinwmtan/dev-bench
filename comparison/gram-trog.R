library(tidyverse)
library(here)
library(glue)
library(latex2exp)
library(lubridate)
library(reticulate)
source("comparison/stats-helper.R")

np <- import("numpy")
oc_dir <- here("evals/gram-trog/openclip/")
# oc_files <- list.files(oc_dir)

## make human data
get_human_data_trog <- function(manifest_file = "assets/gram-trog/manifest.csv",
                                data_file = "evals/gram-trog/syntax-trials-2024-05-15-full-clean.csv") {
  manifest <- read_csv(manifest_file) |> 
    mutate(trial = seq_along(text1)) |> 
    pivot_longer(cols = starts_with("image"),
                 names_to = "option",
                 values_to = "image")
  human_data <- read_csv(data_file) |> 
    mutate(age_yr = interval(parse_date_time(paste(user.birthYear, user.birthMonth, "01"), orders = "ymd"), 
                             user.CreateTime) |> as.numeric('years') |> floor()) |> 
    filter(age_yr <= 12) |> 
    select(text1 = item, answer, response, age_yr) |> 
    group_by(text1, answer) |> 
    count(response) |> 
    mutate(prop = n / sum(n, na.rm = TRUE),
           text1 = tolower(text1) |> str_replace("\\.", ""),
           image = glue("images/{response}.png")) |> 
    left_join(manifest, by = join_by(text1, image)) |> 
    arrange(trial, option) |> 
    filter(!is.na(option)) |> 
    ungroup() |> 
    select(trial, text1, option, prop) |> 
    pivot_wider(names_from = option, values_from = prop) |> 
    mutate(across(starts_with("image"), \(i) replace_na(i, 0)))
}

human_data_trog <- get_human_data_trog()

## comparison fxn
compare_trog <- function(model_data, human_data) {
  opt_kl = get_opt_kl(human_data |> select(trial, starts_with("image")), model_data)
  tibble(kl = opt_kl$objective,
         beta = opt_kl$solution,
         iters = opt_kl$iterations)
}

## comparisons for openclip
openclip_div_trog <- lapply(oc_files, \(ocf) {
  epoch <- str_match(ocf, "[0-9]+")[1] |> as.numeric()
  res <- np$load(here(oc_dir, ocf)) |> 
    as_tibble() |> 
    `colnames<-`(value = c("image1", "image2", "image3", "image4")) |> 
    mutate(trial = seq_along(image1))
  kls <- compare_trog(res, human_data_trog) |> 
    mutate(epoch = epoch)
}) |> bind_rows()

trog_oc <- ggplot(openclip_div_trog, aes(x = log(epoch), y = kl)) +
  geom_point() +
  geom_smooth(col = my_palette[5], span = 1) +#, method = "lm") +
  scale_colour_continuous() +
  # theme_classic() +
  labs(x = "log(Epoch)",
       y = TeX("Model–human dissimilarity ($D^*_{KL}$)"))

ggsave("comparison/gram-trog-oc.png", 
       trog_oc, 
       width = 6.2, height = 4.2, units = "in")

## comparisons for other models
trog_files <- c(list.files("evals/gram-trog", pattern = "*.npy"), "openclip/openclip_epoch_256.npy")

other_res_trog <- lapply(trog_files, \(trogf) {
  res <- np$load(here("evals/gram-trog", trogf))  |> 
    drop()
  if (length(dim(res)) == 3) {
    res <- (res[,,1] - res[,,2])
  }
  res <- res |> 
    as_tibble() |> 
    `colnames<-`(value = c("image1", "image2", "image3", "image4")) |> 
    mutate(trial = seq_along(image1))
  acc <- res |> 
    mutate(correct = image1 > image2 & image1 > image3 & image1 > image4) |> 
    summarise(accuracy = mean(correct)) |> 
    pull(accuracy)
  kls <- compare_trog(res, human_data_trog) |> 
    mutate(model = trogf |> str_remove_all("trog_") |> str_remove_all(".npy"),
           accuracy = acc)
}) |> bind_rows() |> 
  mutate(model = str_replace_all(model, model_rename))

write_csv(other_res_trog, "other_res_trog.csv")

trog_all <- ggplot(other_res_trog, 
                   aes(x = accuracy, y = kl, shape = model)) + 
  geom_point() + 
  scale_shape_manual(values = c(16, 1, 17, 15, 18, 0, 4, 3, 5, 2)) +
  # theme_classic() +
  labs(x = "Accuracy",
       y = TeX("Model–human dissimilarity ($D^*_{KL}$)"),
       shape = "Model")

ggsave("comparison/gram-trog-all.png", 
       trog_all, 
       width = 6.2, height = 4.2, units = "in")

trog_feats <- other_res_trog |> 
  left_join(model_feats, by = join_by(model == Model)) |> 
  mutate(n_params = str_replace_all(`# params`, size_fix) |> as.numeric(),
         n_images = str_replace_all(`# images`, size_fix) |> as.numeric()) |> 
  summarise(accuracy = cor(kl, accuracy),
            size = cor(kl, log(n_params)),
            training = cor(kl, log(n_images)))

## item-level comparisons
item_res_trog <- lapply(trog_files, \(trogf) {
  res <- np$load(here("evals/gram-trog", trogf)) |> 
    as_tibble()
  res <- res |> 
    `colnames<-`(value = c("image1", "image2", "image3", "image4")) |> 
    mutate(trial = seq_along(image1))
  model_name = wgf |> str_remove_all("winoground_") |> str_remove_all(".npy") |> 
    str_replace_all(model_rename)
  beta = other_res_wg |> filter(model == model_name) |> pull(beta)
  human_data_nested <- human_data_trog |> 
    select(trial, starts_with("image"))
  kl <- get_mean_kl_img(human_data_nested, softmax_images(res, beta), return_distribs = TRUE) |> 
    select(-distribs) |> 
    mutate(kl_z = scale(kl)[,1]) |> 
    mutate(model = model_name)
}) |> bind_rows()

item_dif_trog <- item_res_trog |> 
  group_by(trial) |> 
  summarise(kl_z = mean(kl_z))
