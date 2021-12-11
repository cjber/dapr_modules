box::use(
    .. / scripts / defaults,
    here[here],
    readr[read_csv],
    tidyr[drop_na],
    dplyr[mutate, count, group_by, summarise, filter],
    ggplot2[...]
)

ft <- read_csv(here("data/out/full_labelled.csv")) |>
    mutate(
        diff_date = sub(" .*", "", diff_date)
    ) |>
    drop_na()


mean_graph <- ft |>
    group_by(diff_date, label) |>
    count() |>
    ggplot() +
    geom_col(
        aes(x = as.numeric(diff_date), y = n, fill = label),
        colour = "black",
        width = 1,
    ) +
    labs(
        x = "Day delta from flood warning",
        y = "Number of Tweets"
    ) +
    scale_fill_grey() +
    theme(
        axis.text.y = element_text()
    ) +
    scale_x_continuous(
        labels = as.character(unique(ft$diff_date)),
        breaks = as.numeric(unique(ft$diff_date)),
        expand = c(0, 0)
    ) +
    scale_y_continuous(
        expand = c(0, 0)
    )
