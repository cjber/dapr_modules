box::use(
    .. / scripts / defaults,
    here[here],
    readr[read_csv],
    tidyr[fill],
    dplyr[group_by, summarise],
    ggplot2[...]
)

metrics <- read_csv(here("csv_logs/default/version_0/metrics.csv"))
metrics <- metrics |>
    fill(train_loss, train_f1, epoch, val_loss, val_f1)

steps_per_epoch <- (max(metrics$step) / max(metrics$epoch, na.rm = TRUE))

loss_graph <- metrics |>
    ggplot() +
    geom_line(aes(x = step, y = train_loss, linetype = "Train"), alpha = .5) +
    geom_line(aes(x = step, y = val_loss, linetype = "Val")) +
    scale_x_continuous(
        labels = unique(metrics$epoch),
        breaks = unique(metrics$epoch * steps_per_epoch),
        expand = c(0, 0)
    ) +
    scale_color_manual(values = c("Train" = "solid", "Val" = "dashed")) +
    labs(
        x = "Epoch",
        y = "Loss"
    ) +
    theme(
        axis.text.y = element_text()
    )

f1_graph <- metrics |>
    ggplot() +
    geom_line(aes(x = step, y = train_f1, linetype = "Train"), alpha = .5) +
    geom_line(aes(x = step, y = val_f1, linetype = "Val")) +
    scale_x_continuous(
        labels = unique(metrics$epoch),
        breaks = unique(metrics$epoch * steps_per_epoch),
        expand = c(0, 0)
    ) +
    scale_color_manual(values = c("Train" = "solid", "Val" = "dashed")) +
    labs(
        x = "Epoch",
        y = "F1"
    ) +
    theme(
        axis.text.y = element_text()
    )
