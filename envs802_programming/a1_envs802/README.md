# Agent Based Model GUI

Produced as part of the requirements of Assignment 1 of the Leeds Programming for Social Sciences Module.

This README provides a summary of the core functions of the model.

It is recommended that users access this program through its web page:

[https://cjber.github.io/portfolio/agent-based-model/](https://cjber.github.io/portfolio/agent-based-model/)


* Documentation provided through [Sphinx](http://www.sphinx-doc.org/en/master/) can be found [here](https://cjber.github.io/phd/).
* [Download](https://cjber.github.io/content/abm.zip) the latest stable release.

```bash
unzip abm.zip
cd abm/
pip install --user -r requirements.txt
python gui.py
```

This repository contains three core python scripts.

## agent.py

This script contains the `Agent` class, which provides object functions to provide agents with certain parameters.

## environment.py

This script contains the `Environment` class, giving one function allowing for the environment to grow.

## gui.py

This is the main script, and when run will provide a tkinter gui, importing both previously mentioned scripts.

## Other files

The other files contains within this repository primarily relate to the automated production of Sphinx documentation. Additionally `in.txt` provides some essential input for building the model environment, and `requirement.txt` provides the external python packages used.

## Notes

Type hints have been included within functions and key variable assignments, in combination with `[my]py` linting this allowed for more comprehensive debugging. I haven't personally worked with type hinting previously, however I enjoy the benefits it provides and aim to utilise it more frequently.
