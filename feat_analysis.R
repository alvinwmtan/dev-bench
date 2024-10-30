library(tidyverse)
library(patchwork)

feats <- read_csv("model_feat_analysis.csv") |> 
  mutate(Age = ifelse(Age == 25, 16, Age),
         Task = as.factor(Task) |> fct_inorder()) |> 
  pivot_longer(cols = c(Accuracy, Size, Training),
               names_to = "Feature",
               values_to = "Correlation")

ft <- ggplot(feats, aes(x = Age, y = Correlation, col = Task)) +
  geom_point() +
  geom_line() +
  geom_hline(yintercept = 0, lty = "dashed") +
  facet_grid(. ~ Feature) +
  labs(y = "Correlation with model–human similarity") +
  scale_x_continuous(breaks = c(0, 5, 10, 16),
                     labels = c(0, 5, 10, "A"))
ggsave("feats.png", plot = ft, width = 3150, height = 840, unit = "px")

oc <- (vv_oc + ggtitle("VV") | trog_oc + ggtitle("TROG") | wg_oc + ggtitle("WG"))
ggsave("oc.png", plot = oc, width = 3150, height = 840, unit = "px")

oc_all <- (lwl_oc + ggtitle("LWL") +  guides(colour = guide_legend(position = "right")) |
             wat_oc + ggtitle("WAT") + guides(colour = guide_legend(position = "right"))) / 
  (voc_oc + ggtitle("VOC") + guides(colour = guide_legend(position = "right")) | 
     things_oc + ggtitle("THINGS") + guides(colour = guide_legend(position = "right")))
ggsave("oc_all.png", plot = oc_all, width = 2650, height = 1680, unit = "px")

feats_mean <- feats |> 
  group_by(Task, Feature) |> 
  summarise(Correlation = mean(Correlation))

ft_mean <- ggplot(feats_mean, aes(x = Task, y = Correlation, fill = Feature)) + 
  geom_col(position = "dodge") +
  geom_hline(yintercept = 0, lty = "dashed") +
  labs(y = "Correlation with model–human similarity") +
  scale_fill_manual(values = my_palette[5:7])

ggsave("feats_mean.png", plot = ft_mean, width = 1600, height = 950, unit = "px")

ft_full <- (ft_mean | vv_all + ggtitle("VV")) + 
  plot_layout(widths = c(3, 2)) +
  plot_annotation(tag_levels = 'a') & 
  theme(plot.tag = element_text(face = "bold"))
ggsave("feats_full.png", plot = ft_full, width = 2800, height = 950, unit = "px")

# feats_gen <- read_csv("gen_acc.csv")
# ft_gen <- ggplot(feats_gen, aes(x = Task, y = Correlation, 
#                                 fill = factor(Method, levels = c("NTP", "LL")))) +
#   geom_col(position = "dodge") + 
#   labs(y = "Correlation with model–human similarity",
#        fill = "Method") +
#   scale_fill_manual(values = c("#3e7ff0", "#8eafe8"),
#                     breaks = c("NTP", "LL"))
# ggsave("feats_gen.png", plot = ft_gen, width = 1200, height = 950, unit = "px")
