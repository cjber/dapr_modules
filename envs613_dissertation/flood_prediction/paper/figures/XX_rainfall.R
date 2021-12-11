box::use(
    .. / scripts / defaults,
    here[here],
    readr[read_csv],
    dplyr[bind_rows, filter, group_by, summarise, mutate],
    tidyr[replace_na],
    jsonlite[fromJSON],
    ggplot2[...],
    patchwork[...]
)

ft <- readLines(here("data/floods/flood_tweets.jsonl")) |>
    lapply(fromJSON) |>
    lapply(unlist) |>
    bind_rows() |>
    filter(idx == 47010) |>
    mutate(date = as.Date(sub("T.*", "", created_at))) |>
    group_by(date, label) |>
    filter(label == "FLOOD") |>
    dplyr::count(.drop = FALSE)


rainfall <- read_csv(here("data/floods/rainfall_dates.csv")) |>
    filter(idx == 47010) |>
    mutate(
        diff_date = as.numeric(as.numeric(sub(" .*", "", diff_date)))
    )

ft <- merge(ft, rainfall, by = "date", all.y = TRUE)
ft <- ft |> mutate(
    label = replace_na(label, "FLOOD"),
    n = replace_na(n, 0)
)
ft <- ft |>
    mutate(
        diff_date = as.numeric(as.numeric(sub(" .*", "", diff_date)))
    )

ggplot() +
    geom_col(
        data = rainfall,
        aes(x = diff_date, y = value),
        colour = "black",
        width = 1
    ) +
    scale_fill_grey() +
    theme(
        axis.text.y = element_text()
    ) +
    scale_x_continuous(
        labels = as.character(rainfall$diff_date),
        breaks = as.numeric(rainfall$diff_date),
        expand = c(0, 0)
    ) +
    scale_y_continuous(
        expand = c(0, 0)
    ) +
    theme(legend.position = c(0.1, 0.9)) |
    ggplot() +
        geom_col(
            data = ft, aes(x = diff_date, y = n), colour = "black", width = 1
        ) +
        scale_fill_grey() +
        theme(
            axis.text.y = element_text()
        ) +
        scale_x_continuous(
            labels = as.character(rainfall$diff_date),
            breaks = as.numeric(rainfall$diff_date),
            expand = c(0, 0)
        ) +
        scale_y_continuous(
            expand = c(0, 0)
        ) +
        theme(legend.position = c(0.1, 0.9))
