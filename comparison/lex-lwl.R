library(tidyverse)
library(here)
library(glue)
library(latex2exp)
library(reticulate)
library(peekbankr)
source("comparison/stats-helper.R")

np <- import("numpy")
oc_dir <- here("evals/lex-lwl/openclip/")
# oc_files <- list.files(oc_dir)

## make human data
get_human_data_lwl <- function(
    manifest_file = "assets/lex-lwl/manifest.csv",
    use_cached = TRUE,
    cache_file = "evals/lex-lwl/human.csv") {
  if (use_cached && file.exists(cache_file)) return(read_csv(cache_file))
  
  manifest <- read_csv(manifest_file) |> 
    mutate(trial = seq_along(text1))
  
  ## get Adams data
  aoi_tp <- get_aoi_timepoints(dataset_name = "adams_marchman_2018")
  admins <- get_administrations(dataset_name = "adams_marchman_2018")
  subjects <- get_subjects()
  trials <- get_trials(dataset_name = "adams_marchman_2018")
  trial_types <- get_trial_types(dataset_name = "adams_marchman_2018")
  stimuli <- get_stimuli(dataset_name = "adams_marchman_2018")
  aoi_joined <- aoi_tp |> 
    left_join(admins) |> 
    left_join(subjects) |> 
    left_join(trials) |> 
    left_join(trial_types) |> 
    left_join(stimuli |> select(target_id = stimulus_id, target_image = lab_stimulus_id)) |> 
    left_join(stimuli |> select(distractor_id = stimulus_id, distractor_image = lab_stimulus_id))
  
  adams_data <- aoi_joined |> 
    filter(t_norm >= 300,
           t_norm <= 4000,
           aoi %in% c("target", "distractor"),
           age >= 17,
           age <= 19) |> 
    group_by(administration_id, trial_id, subject_id, target_image) |> 
    count(aoi) |> 
    mutate(prop = n / sum(n),
           age_bin = 1.5) |> 
    filter(aoi == "target") |> 
    group_by(age_bin, target_image) |> 
    summarise(prop = mean(prop)) |> 
    mutate(image1 = glue("images_adams/{target_image}.png") |> 
             str_replace("birdie", "birdy")) |> 
    select(-target_image)
  
  ## get Frank data
  frank_data <- read_csv("evals/lex-lwl/experiment_info/eye.tracking.csv") |>
    filter(word.type == "Familiar-Familiar",
           age.grp == 2) |>
    group_by(age.grp, word) |>
    summarise(prop = mean(prop)) |> 
    rename(age_bin = age.grp,
           text1 = word) |> 
    left_join(manifest |> 
                filter(str_detect(image1, "frank")) |> 
                select(text1, image1)) |>
    select(-text1)
  
  ## get Donnelly data
  manifest_lr <- manifest |> 
    mutate(lr = ifelse(trial %% 2 == 1, 
                       glue("{text1}_left.wmv"),
                       glue("{text1}_right.wmv"))) |> 
    filter(str_detect(image1, "donnelly"))
  donnelly_data <- read_csv("evals/lex-lwl/LT_30mo.csv") |> 
    select(target_image = Target,
           lr = MediaName,
           participant_id = ParticipantName,
           prop = Prop) |> 
    group_by(target_image, lr) |> 
    summarise(prop = mean(prop)) |> 
    mutate(age_bin = 2.5) |> 
    left_join(manifest_lr |> select(lr, image1), by = join_by(lr)) |> 
    ungroup() |> 
    select(-lr, -target_image)
  
  all_data <- bind_rows(adams_data, frank_data, donnelly_data) |> 
    mutate(image1 = str_replace(image1, "doggy", "doggie")) |> 
    left_join(manifest |> select(trial, image1)) |> 
    select(-image1)
  
  write_csv(all_data, cache_file)
}

human_data_lwl <- get_human_data_lwl()

## comparison fxn
compare_lwl <- function(model_data, human_data) {
  model_data_correct <- model_data |> 
    mutate(correct = image1 > image2)
  # 
  # model_data_by_text <- model_data |> 
  #   pivot_longer(cols = starts_with("image"),
  #                names_to = c("image", "text"),
  #                names_pattern = "(image[12])(text[12])",
  #                values_to = "score") |> 
  #   pivot_wider(names_from = image,
  #               values_from = score) |> 
  #   mutate(trial = glue("{trial}_{text}"),
  #          correct = ifelse(text == "text1", image1 > image2, image2 > image1)) |> 
  #   filter(trial %in% human_data$trial)
  
  human_data_nested <- human_data |> 
    rename(image1 = prop) |> 
    mutate(image2 = 1. - image1) |> 
    nest(data = -age_bin) |> 
    mutate(opt_kl = lapply(data, \(d) {get_opt_kl(d, model_data_correct)}),
           kl = sapply(opt_kl, \(r) {r$objective}),
           beta = sapply(opt_kl, \(r) {r$solution}),
           iters = sapply(opt_kl, \(r) {r$iterations}),
           accuracy = mean(model_data_correct$correct, na.rm = TRUE)) |> 
    select(-data, -opt_kl)
}

## comparisons for openclip
openclip_div_lwl <- lapply(oc_files, \(ocf) {
  epoch <- str_match(ocf, "[0-9]+")[1] |> as.numeric()
  res <- np$load(here(oc_dir, ocf)) |> 
    as_tibble() |> 
    `colnames<-`(value = c("image1", "image2")) |> 
    mutate(trial = seq_along(image1))
  kls <- compare_lwl(res, human_data_lwl) |> 
    mutate(epoch = epoch)
}) |> bind_rows()

lwl_oc <- ggplot(openclip_div_lwl, 
                 aes(x = log(epoch), y = kl, col = as.factor(age_bin))) +
  geom_point() +
  geom_smooth(span = 1) +#, method = "lm") +
  # scale_colour_continuous() +
  # theme_classic() +
  labs(x = "log(Epoch)",
       y = TeX("Model–human dissimilarity ($D^*_{KL}$)"),
       col = "Age") +
  guides(colour = guide_legend(position = "inside")) +
  coord_cartesian(ylim = c(0, 0.06)) +
  theme(legend.position.inside = c(0.9, 0.8))

ggsave("comparison/lex-lwl-oc.png", 
       lwl_oc, 
       width = 6.2, height = 4.2, units = "in")

# mod_lwl <- lm(cor ~ log(epoch) * age, data = openclip_cors) |> 
#   tidy()

## comparisons for other models
lwl_files <- c(list.files("evals/lex-lwl", pattern = "*.npy"), "openclip/openclip_epoch_256.npy")

other_res_lwl <- lapply(lwl_files, \(lwlf) {
  res <- np$load(here("evals/lex-lwl", lwlf)) |> 
    drop()
  if (length(dim(res)) == 3) {
    res <- (res[,,1] - res[,,2])
  }
  res <- res |> 
    as_tibble() |> 
    `colnames<-`(value = c("image1", "image2")) |> 
    mutate(trial = seq_along(image1))
  acc <- res |>
    mutate(correct = image1 > image2) |>
    summarise(accuracy = mean(correct, na.rm = TRUE)) |>
    pull(accuracy)
  kls <- compare_lwl(res, human_data_lwl) |> 
    mutate(model = lwlf |> str_remove_all("lwl_") |> str_remove_all("_final") |> str_remove_all(".npy"),
           accuracy = acc)
}) |> bind_rows() |> 
  mutate(model = str_replace_all(model, model_rename))

write_csv(other_res_lwl, "other_res_lwl.csv")

lwl_all <- ggplot(other_res_lwl, 
                  aes(x = accuracy, y = kl, col = as.factor(age_bin), shape = model)) + 
  geom_point() + 
  scale_shape_manual(values = c(16, 1, 17, 15, 18, 0, 4, 3, 5, 2)) +
  # theme_classic() +
  labs(x = "Accuracy",
       y = TeX("Model–human dissimilarity ($D^*_{KL}$)"),
       shape = "Model",
       col = "Age")

ggsave("comparison/lex-lwl-all.png", 
       lwl_all, 
       width = 6.2, height = 4.2, units = "in")

lwl_devt <- other_res_lwl |> 
  group_by(model) |> 
  summarise(cor = cor(age_bin, kl, method = "spearman"))

lwl_feats <- other_res_lwl |> 
  left_join(model_feats, by = join_by(model == Model)) |> 
  mutate(n_params = str_replace_all(`# params`, size_fix) |> as.numeric(),
         n_images = str_replace_all(`# images`, size_fix) |> as.numeric()) |> 
  group_by(age_bin) |> 
  summarise(accuracy = cor(kl, accuracy),
            size = cor(kl, log(n_params)),
            training = cor(kl, log(n_images)))

