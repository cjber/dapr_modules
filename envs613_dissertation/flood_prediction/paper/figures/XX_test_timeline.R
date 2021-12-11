box::use(
    .. / scripts / defaults,
    here[here],
    readr[read_csv],
    dplyr[mutate, group_by, count, filter, summarise],
    tidyr[drop_na],
    scales[date_format],
    ggplot2[...]
)


ft <- read_csv(here("data/floods/flood_tweets.csv"))

ft <- ft |>
    mutate(
        date = sub(" .*", "", ft$created_at),
        warning_date = sub(" .*", "", ft$warning_time),
        date_diff = as.Date(date) - as.Date(warning_date)
    )

ftx <- ft |>
    group_by(date) |>
    count()


timeline_graph <- ftx |> ggplot() +
    geom_col(
        aes(x = as.character(date), y = n),
        width = 1,
        fill = "black"
    ) +
    geom_vline(xintercept = ft$warning_date, alpha = .2) +
    scale_fill_grey(start = 0.4, end = 0) +
    theme(axis.text.y = element_text()) +
    scale_x_discrete(
        expand = c(0, 0),
        breaks = ftx$date[c(TRUE, rep(FALSE, 50))]
    ) +
    scale_y_continuous(expand = c(0, 0)) +
    labs(x = "Time", y = "Number of Tweets") +
    theme(legend.position = "none")
