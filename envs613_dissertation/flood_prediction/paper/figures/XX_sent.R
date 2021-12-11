box::use(
    .. / scripts / defaults,
    here[here],
    readr[read_csv],
    dplyr[group_by, summarise, mutate],
    tidyr[pivot_longer],
    ggplot2[...]
)

ft <- read_csv(here("data/floods/flood_sent.csv")) |>
    mutate(diff_date = as.numeric(gsub("[^0-9.-]", "", diff_date)))

sent_graph <- ft |>
    group_by(diff_date) |>
    summarise(
        negative = mean(negative),
        neutral = mean(neutral),
        positive = mean(positive)
    ) |>
    pivot_longer(
        cols = c("negative", "neutral", "positive"), names_to = "sentiment"
    ) |>
    ggplot() +
    geom_col(
        aes(fill = sentiment, y = value, x = diff_date),
        colour = "black", width = 1
    ) +
    labs(
        x = "Day delta from flood warning",
        y = "Mean number of Tweets"
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
