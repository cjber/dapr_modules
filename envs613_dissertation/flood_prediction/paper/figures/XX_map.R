box::use(
    here[here],
    ggplot2[...],
    ggrastr[rasterise],
    ggfx[with_shadow],
    sf[st_read, st_simplify, st_difference],
    cjrmd
)

theme_set(cjrmd::cj_map_theme)

fw <- st_read(here("data/floods/flood_warnings.gpkg"), quiet = TRUE)
fa <- st_read(here("data/floods/flood_areas.gpkg"), quiet = TRUE)

ggplot() +
    rasterise(
        geom_sf(data = fa, colour = ggplot2::alpha("grey", 0.2)),
        dpi = 300
    ) +
    rasterise(
        with_shadow(
            geom_sf(data = fw, colour = "black"),
            x_offset = 0, y_offset = 0, sigma = 5
        )
    )
