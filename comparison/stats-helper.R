library(tidyverse)
library(glue)
library(extrafont)
library(philentropy)
library(nloptr)
loadfonts()

model_rename <- c("blip" = "BLIP",
                  "bridgetower" = "BridgeTower",
                  "clip_base" = "CLIP-base",
                  "clip_large" = "CLIP-large",
                  "cvcl" = "CVCL",
                  "flava" = "FLAVA",
                  "openclip/openclip_epoch_256" = "OpenCLIP",
                  "vilt" = "ViLT",
                  "siglip" = "SigLIP",
                  "openclip_random" = "Random (OpenCLIP)",
                  "kosmos" = "Kosmos-2",
                  "llava" = "LLaVA",
                  "tiny" = "Tiny",
                  "cogvlm" = "CogVLM")
model_feats <- read_csv("model_feat_vals.csv")
size_fix <- c("K" = "e3",
              "M" = "e6",
              "B" = "e9")

epoch_set <- c(seq(1, 10),
               seq(12, 20, 2),
               seq(25, 50, 5),
               seq(60, 100, 10),
               122, 140, 160, 180, 200, 256)

oc_files <- glue("openclip_epoch_{epoch_set}.npy")

theme_set(theme_classic() +
            theme(text = element_text(family = "Source Sans Pro")))
my_palette = c("#f0653e", "#f0cf3e", "#a6f03e", "#3ef0e1", "#3e7ff0", "#a63ef0", "#f03e94")

assign("scale_colour_discrete", 
       function(..., values = my_palette) scale_colour_manual(..., values = values), globalenv())
assign("scale_fill_discrete", 
       function(..., values = my_palette) scale_fill_manual(..., values = values), globalenv())

## RSA functions
rsa <- function(mat1, mat2, method = "spearman") {
  mat1_lower <- mat1[lower.tri(mat1)]
  mat2_lower <- mat2[lower.tri(mat2)]
  cor(mat1_lower, mat2_lower, use = "pairwise.complete.obs", method = method)
}

get_permutations <- function(mat1, mat2, method = "spearman", nsim = 1000, seed = 42) {
  set.seed(seed)
  sims <- sapply(1:nsim, \(sim) {
    idx <- sample(nrow(mat1))
    mat1_perm <- mat1[idx, idx]
    rsa(mat1_perm, mat2, method = method)
  })
}

calc_permuted_p <- function(sim_cors, obs_cor) {
  sum(abs(obs_cor) < abs(sim_cors)) / length(sim_cors)
}

## image functions
softmax_images <- function(data, beta = 1) {
  data |> 
    mutate(across(starts_with("image"), \(i) exp(beta * i)),
           rowsum = rowSums(across(starts_with("image"))),
           across(starts_with("image"), \(i) i/rowsum)) |> 
    select(-rowsum)
}

get_mean_kl_img <- function(human_probs_wide, model_probs_wide, return_distribs = FALSE) {
  combined_distribs <- bind_rows(human_probs_wide, model_probs_wide) |> 
    mutate(across(starts_with("image"), \(i) replace_na(i, 0))) |> 
    nest(distribs = -trial) |> 
    filter(lapply(distribs, nrow) == 2) |> 
    mutate(kl = sapply(distribs, \(d) {
      d |> select(starts_with("image")) |> 
        as.matrix() |> 
        {\(m) quietly(KL)(m, unit = "log")["result"][[1]]}()
    }))
  if (return_distribs) return(combined_distribs)
  combined_distribs |> 
    pull(kl) |> 
    mean(na.rm = TRUE)
}

mean_kl_img <- \(beta) {get_mean_kl_img(human_probs_wide, 
                                        softmax_images(model_logits_wide, beta))}

## text functions
softmax_texts <- function(data, beta = 1) {
  data |> 
    group_by(cue) |> 
    mutate(model = exp(beta * model),
           model_sum = sum(model),
           model = model/model_sum) |> 
    select(-model_sum)
}

get_mean_kl_txt <- function(combined_probs) {
  combined_distribs <- combined_probs |> 
    select(-target) |> 
    nest(distribs = -cue) |> 
    mutate(distribs = lapply(distribs, \(d) t(d)),
           kl = sapply(distribs, \(d) {
             d |> as.matrix() |> 
               {\(m) quietly(KL)(m, unit = "log")["result"][[1]]}()
           }))
  combined_distribs |> 
    pull(kl) |> 
    mean(na.rm = TRUE)
}

## optimisation functions
get_opt_kl <- function(human_probs_wide, model_logits_wide) {
  mean_kl <- \(beta) {get_mean_kl_img(human_probs_wide, 
                                      softmax_images(model_logits_wide, beta))}
  res <- nloptr(x0 = 1, 
                eval_f = mean_kl, 
                lb = 0.025,
                ub = 40,
                opts = list(algorithm = "NLOPT_GN_DIRECT_L",
                            ftol_abs = 1e-4,
                            maxeval = 200))
}

get_opt_kl_txt <- function(combined_data) {
  mean_kl <- \(beta) {get_mean_kl_txt(softmax_texts(combined_data, beta))}
  res <- nloptr(x0 = 1, 
                eval_f = mean_kl, 
                lb = 0.025,
                ub = 40,
                opts = list(algorithm = "NLOPT_GN_DIRECT_L",
                            ftol_abs = 1e-4,
                            maxeval = 200))
}

get_reg_kl <- function(human_probs_wide, model_logits_wide, beta = 1) {
  res <- get_mean_kl(human_probs_wide, softmax_images(model_logits_wide, beta))
  list(objective = res,
       solution = beta,
       iterations = 0)
}

get_opt_kl_allage <- function(human_data, model_logits_wide, beta) {
  human_data |> 
    select(age_bin, trial, starts_with("image")) |> 
    nest(data = -age_bin) |> 
    mutate(kl = sapply(data, \(d) get_mean_kl(d, softmax_images(model_logits_wide, beta)))) |> 
    pull(kl) |> 
    mean(na.rm = TRUE)
}

get_opt_kl_allepoch <- function(human_probs_wide, model_data, beta) {
  model_data_nested <- model_data |> 
    select(epoch, trial, starts_with("image")) |> 
    nest(data = -epoch) |> 
    mutate(kl = sapply(data, \(d) get_mean_kl(human_probs_wide, softmax_images(d, beta)))) |> 
    pull(kl) |> 
    mean(na.rm = TRUE)
}
