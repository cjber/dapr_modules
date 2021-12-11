box::use(
    .. / scripts / defaults,
    here[here],
    readr[read_csv],
    dplyr[group_by, summarise, mutate, filter],
    stringr[str_split, str_replace_all, str_sub],
    tidyr[pivot_longer, drop_na],
    ggplot2[...]
)

ft <- read_csv(here("data/out/flood_places.csv")) |>
    drop_na(places) |>
    mutate(
        diff_date = as.numeric(gsub("[^0-9.-]", "", diff_date)),
        places = gsub("'", "", places) |>
            str_sub(2, -2) |>
            str_split(", "),
        num_places = lengths(places)
    )

place_graph <- ft |>
    group_by(diff_date, label) |>
    summarise(mean_places = sum(num_places)) |>
    ggplot() +
    geom_col(
        aes(x = as.numeric(diff_date), y = mean_places, fill = label),
        colour = "black",
        width = 1,
    ) +
    labs(
        x = "Day delta from flood warning",
        y = "Number of places mentioned in Tweets"
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
