if (!requireNamespace("renv")) {
    install.packages("renv", repos = "https://cloud.r-project.org/")
}

# these are dev dependencies
box::use(
    languageserver,
    nvimcom
)

## -- defaults
cjrmd::default_latex_chunk_opts()
ggplot2::theme_set(cjrmd::cj_plot_theme)
