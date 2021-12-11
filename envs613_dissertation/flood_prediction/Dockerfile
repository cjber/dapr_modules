FROM cjber/cuda

RUN pacman -S python-pip r pyenv pandoc --noconfirm \
    && pip install poetry

WORKDIR /flood_tweets
COPY pyproject.toml .python-version renv.lock .Rprofile ./
COPY paper paper
COPY renv/activate.R renv/activate.R
COPY src conf ./

RUN Rscript -e "install.packages('renv', repos = 'https://cran.r-project.org')" \
    && Rscript -e "renv::restore()"

RUN yes | pyenv install $(cat .python-version) \
    && poetry install
