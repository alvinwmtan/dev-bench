library(tidyverse)
library(here)
library(latex2exp)
library(reticulate)
library(psych)
source("comparison/stats-helper.R")

np <- import("numpy")
oc_dir <- here("evals/sem-wat/openclip/")
# oc_files <- list.files(oc_dir)

## make human data
get_human_data_wat <- function() {
  child_1 <- read_csv(here("assets/sem-wat/entwisle_norms.csv"),
                      show_col_types = FALSE) |> 
    pivot_longer(cols = -c(cue, response),
                 names_to = "age_group",
                 values_to = "count") |> 
    filter(!is.na(count)) |> 
    mutate(
      age = case_when(
        age_group == "kindergarten" ~ 5,
        age_group == "first" ~ 6,
        age_group == "third" ~ 8,
        age_group == "fifth" ~ 10), 
      n = case_when(
        age_group == "kindergarten" ~ 200,
        age_group == "first" ~ 280,
        age_group == "third" ~ 280,
        age_group == "fifth" ~ 280)) |> 
    rename(target = response)
  
  # child_2 <- read_tsv(here("assets/sem-wat/TA_data_formatted.txt"),
  #                     show_col_types = FALSE) |> 
  #   count(Experimenter_Word, Child_Word, Age2) |> 
  #   group_by(Experimenter_Word, Age2) |> 
  #   mutate(count = n,
  #          n = sum(n),
  #          age = case_when(
  #            Age2 == "Younger" ~ 4,
  #            Age2 == "Older" ~ 7,
  #            .default = 25)) |> 
  #   filter(count != 1) |> 
  #   rename(cue = Experimenter_Word,
  #          target = Child_Word) |> 
  #   ungroup()
  
  child_union <- child_1 |> select(age, cue, target, count, n) # |> 
    # bind_rows(child_2 |> select(age, cue, target, count, n))
  
  adult_loc <- here("assets/sem-wat/adult/")
  adult <- lapply(list.files(adult_loc), \(f) {
    read_lines(here(adult_loc, f), skip = 3) |> 
      str_remove_all("[¥�]") |> 
      I() |> 
      read_csv(show_col_types = FALSE)
  }) |> bind_rows() |> 
    filter(CUE %in% toupper(child_union$cue))
  
  adult_union <- adult |> 
    mutate(age = 25,
           cue = tolower(CUE),
           target = tolower(TARGET)) |> 
    filter(`#P` > 2) |> 
    select(age, cue, target, count = `#P`, n = `#G`)
  
  all_union <- child_union |> 
    bind_rows(adult_union) |> 
    mutate(across(c(cue, target), \(w) str_replace_all(w, "_", " "))) |> 
    group_by(age, cue, target) |> 
    summarise(count = sum(count),
              n = sum(n),
              .groups = "drop_last") |> 
    mutate(prop = count / n,
           prop_norm = count / sum(count)) |> 
    ungroup()
  
  ## legacy code for generating the manifest
  # all_text <- c(all_union$cue, all_union$target) |> unique()
  # write_csv(tibble(text1 = all_text), "assets/sem-wat/manifest.csv")
}

human_data_wat <- get_human_data_wat()

## comparison fxn
compare_wat <- function(model_data, human_data) {
  set.seed(42)
  MAX_N = 20
  
  all_text <- c(human_data$cue, human_data$target) |> unique()
  comparison <- human_data |> 
    select(age, cue, target, human = prop) |> 
    nest(data = -c(age, cue)) |> 
    mutate(n_targets = sapply(data, nrow),
           data = pmap(list(data, cue, n_targets), \(d, c, n) {
             opts <- setdiff(all_text, c(c, d$target))
             rand <- tibble(target = sample(opts, MAX_N - n),
                            human = 0)
             bind_rows(d, rand)
           })) |> 
    unnest(cols = data) |> 
    mutate(similarity = map2_dbl(cue, target, \(c, t) {
      embeds <- rbind(model_data[which(all_text == c),],
                      model_data[which(all_text == t),])
      philentropy::distance(embeds, method = "cosine",
                            mute.message = TRUE)
    })) |> 
    select(age, cue, target, human, model = similarity) |> 
    nest(data = -age) |> 
    mutate(opt_kl = lapply(data, \(d) {get_opt_kl_txt(d)}),
           kl = sapply(opt_kl, \(r) {r$objective}),
           beta = sapply(opt_kl, \(r) {r$solution}),
           iters = sapply(opt_kl, \(r) {r$iterations})) |> 
    select(-data, -opt_kl)
}

## comparisons for openclip
openclip_div_wat <- lapply(oc_files, \(ocf) {
  epoch <- str_match(ocf, "[0-9]+")[1] |> as.numeric()
  res <- np$load(here(oc_dir, ocf))
  kls <- compare_wat(res, human_data_wat) |> 
    mutate(epoch = epoch)
}) |> bind_rows()

wat_oc <- ggplot(openclip_div_wat, aes(x = log(epoch), y = kl, col = as.factor(age))) +
  geom_point() +
  geom_smooth(span = 1) +
  # theme_classic() +
  labs(x = "log(Epoch)",
       y = TeX("Model–human dissimilarity ($D^*_{KL}$)"),
       col = "Age") +
  guides(colour = guide_legend(position = "inside")) +
  # coord_cartesian(ylim = c(0, 0.06)) +
  theme(legend.position.inside = c(0.9, 0.3))

ggsave("comparison/sem-wat-oc.png", 
       wat_oc, 
       width = 6.2, height = 4.2, units = "in")

## comparisons for other models
wat_files <- c(list.files("evals/sem-wat", pattern = "*.npy"), "openclip/openclip_epoch_256.npy")

other_res_wat <- lapply(wat_files, \(watf) {
  res <- np$load(here("evals/sem-wat", watf))
  kls <- compare_wat(res, human_data_wat) |> 
    mutate(model = watf |> str_remove_all("wat_") |> str_remove_all(".npy"))
}) |> bind_rows() |> 
  mutate(model = str_replace_all(model, model_rename))

write_csv(other_res_wat, "other_res_wat.csv")

wat_all <- ggplot(other_res_wat |> filter(!is.na(kl)),
                  aes(x = fct_reorder(model, desc(kl)), y = kl, fill = as.factor(age))) + 
  geom_col(position = "dodge") + 
  # scale_shape_manual(values = c(16, 1, 17, 15, 18, 0, 2, 3)) +
  # theme_classic() +
  labs(x = "Model",
       y = TeX("Model–human dissimilarity ($D^*_{KL}$)"),
       fill = "Age") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("comparison/sem-wat-all.png", 
       wat_all, 
       width = 6.2, height = 4.2, units = "in")

wat_devt <- other_res_wat |> 
  group_by(model) |> 
  summarise(cor = cor(age, kl, method = "spearman"))

wat_feats <- other_res_wat |> 
  left_join(model_feats, by = join_by(model == Model)) |> 
  mutate(n_params = str_replace_all(`# params`, size_fix) |> as.numeric(),
         n_images = str_replace_all(`# images`, size_fix) |> as.numeric()) |> 
  group_by(age) |> 
  summarise(size = cor(kl, log(n_params)),
            training = cor(kl, log(n_images)))
