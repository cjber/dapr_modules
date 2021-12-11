from typing import List
import tkinter as tk  # gui
import imageio  # save images and gifs
import os  # os related functions
import requests  # web 'scraping'
import csv  # read and write csv files
import bs4  # further web scraping tools
import random  # provides (pseudo) random number functions
import agent as agf  # agent class framework
import environment as envf  # environment class framework
import matplotlib as mpl  # various plotting utilities
mpl.use('TkAgg')  # noqa, have to load before plt etc.
import matplotlib.pyplot as plt  # plots function
import matplotlib.animation as anim  # animated plots
import matplotlib.backends.backend_tkagg as mpltk  # plot backend for gui

# ensure that when updating my own imports they are updated here as well
import importlib
importlib.reload(agf)
importlib.reload(envf)

# taken from https://stackoverflow.com/a/34351483
# as using LinearSegmentedColormap, have to reverse each color individually


def reverse_colourmap(cmap, name='my_cmap_r'):
    """
    Reverse the colour direction of a mpl cmap

    :param cmap: Input colormap
    :type cmap: dict()
    :param name: reverse cmap out name, defaults to 'my_cmap_r'
    :type name: dict(), optional
    """
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1-t[0], t[2], t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k, reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r


def create_env(env_file: str) -> List[List[int]]:
    """
    Create a base environment using predetermined values.

    Values given here are provided in a text file and read in
    as a two dimensional array.
    Values above 200 are assigned a random value between 201 and 205,
    this allows for values above 200 to represent green algae and be eaten.

    """
    environment = []  # empty matrix

    # create 2d matrix using csv txt file
    with open(env_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            rowlist = []
            for value in row:
                int_value: int = int(value)
                if int_value > 200:
                    int_value = random.randrange(201, 205)
                rowlist.append(int_value)
            environment.append(rowlist)
    return environment


# read in table of agent positional data
r = requests.get(
    'http://www.geog.leeds.ac.uk/courses/computing/practicals/python/agent-framework/part9/data.html')  # noqa
content = r.text  # extract text content
soup = bs4.BeautifulSoup(content, 'html.parser')  # parse html for content
table = soup.find(id='yxz')  # find xyz cols from table
td_xs = soup.find_all(attrs={"class": "y"})  # assign y vals
td_ys = soup.find_all(attrs={"class": "x"})  # assign x vals

fig = plt.figure()  # start figure environment

# assign fixed min and max environment values
min_val: int = 0
max_val: int = 200
# must be type matplotlib.color.LinearSegmentedColormap
cmap = plt.cm.Blues
cmap = reverse_colourmap(cmap)
cmap.set_over(color='green')


class AgentGUI:

    # print some preliminary warnings to consider before running the model
    print("Select Agents and Iterations from dropdowns...")
    print("Before running select save as gif, or infinite iterations")

    def __init__(self, master: tk.Tk, env_file: str) -> None:
        """
        Initial state of the GUI.

        Running through tkinter, allows for interactivity.

        :param master: tkinter main frame
        :type master: tk.Tk()
        """

        # default master variables
        self.master = master
        self.master.title("Model GUI")
        self.agents: List[agf.Agent] = []
        self.environment: List[List[int]] = create_env(env_file)

        # tkinter frames
        self.frame = tk.Frame(master)
        self.frame_controls = tk.Frame(master)
        self.frame_widgets = tk.Frame(master)

        # frame positioning
        self.frame.grid(row=0, column=1, sticky="n")
        self.frame_controls.grid(row=0, column=0, sticky="nw")
        self.frame_widgets.grid(row=1, column=1, sticky="swe")

        # model plot area position and init
        self.canvas = mpltk.FigureCanvasTkAgg(fig, master=self.frame)
        self.canvas._tkcanvas.grid(row=0, column=1, rowspan=2, sticky="nswe")

        # variables for saving models
        self.save_img = 0
        self.filenames: List[str] = []

        # tkinter interactive buttons etc, very large section ending line 233
        # generally assigning functions to buttons, and button positioning
        self.var_agent = tk.IntVar()
        self.opts_agent = [10, 20, 30, 40, 50]
        self.var_agent.set(self.opts_agent[0])
        self.var_agent.trace("w", self.change)
        self.var_agent.trace("w", self.initial_vars)

        self.var_iter = tk.IntVar()
        self.opts_iter = [10, 100, 1000]
        self.var_iter.set(self.opts_iter[0])
        self.var_iter.trace("w", self.change)
        self.var_iter.trace("w", self.initial_vars)

        self.menu_agent = tk.OptionMenu(self.frame_controls, self.var_agent,
                                        *self.opts_agent)
        self.menu_agent.grid(row=10,
                             column=1,
                             padx=5,
                             pady=5)

        self.menu_iter = tk.OptionMenu(self.frame_controls, self.var_iter,
                                       *self.opts_iter)
        self.menu_iter.grid(row=11, column=1)

        self.ag_lab = tk.Label(self.frame_controls, text="Agents")
        self.ag_lab.grid(row=10, column=0)
        self.it_lab = tk.Label(self.frame_controls, text="Iterations")
        self.it_lab.grid(row=11, column=0)

        self.start_btn = tk.Button(
            self.frame_controls,
            text="Run",
            command=self.run)
        self.start_btn.grid(row=0,
                            column=0,
                            sticky="we",
                            padx=5,
                            pady=2,
                            columnspan=2)

        self.stop_btn = tk.Button(
            self.frame_controls,
            text="Stop/Pause",
            command=self.stop)
        self.stop_btn.grid(row=2,
                           column=0,
                           sticky="we",
                           padx=5,
                           pady=2,
                           columnspan=2)

        self.resume_btn = tk.Button(
            self.frame_controls,
            text="Resume",
            command=self.resume)
        self.resume_btn.grid(row=1,
                             column=0,
                             sticky="we",
                             padx=5,
                             pady=2,
                             columnspan=2)

        self.inf_btn = tk.Button(
            self.frame_widgets,
            text="Infinite",
            command=self.toggle_inf)
        self.inf_btn.grid(row=0,
                          column=0,
                          padx=5,
                          pady=5)

        self.save_btn = tk.Button(
            self.frame_widgets,
            text="Save as GIF",
            command=self.toggle_save)
        self.save_btn.grid(row=0,
                           column=6,
                           padx=5,
                           pady=5)

        self.close_button = tk.Button(self.frame_widgets, text="Close",
                                      command=master.destroy)
        self.close_button.grid(row=0,
                               column=10,
                               pady=5,
                               padx=5)

        # default iteration variables
        self.carry_on: bool = True
        self.current_gen: int = 0
        self.inf: bool = False
        self.num_agent: int = self.var_agent.get()
        self.num_iter: int = self.var_iter.get()

    def update(self, *args: int) -> None:
        """
        Takes inputs from self.agents using the Agent class.

        Agent class taken from the import agent (agf), functions associated
        with the agent class are updated iteratively and displayed on a plot.

        Update also contains code to allow for the production of a model gif.

        :param args: Values associated with dropdown selections
        :type args: tk.IntVar()
        """

        # iterate through each agent and run each class function
        for agent in self.agents:
            agent.move()
            agent.eat(self.environment)
            agent.make_baby()
            agent.death()
            agent.carnivore()

        # agent age increases every iteration
            agent.age += 1
            envf.Environment(self.environment).grow_algae(self.environment)
            fig.clear()

        # plot each agent on the input environment
        for agent in self.agents:
            plt.imshow(self.environment, vmin=min_val, vmax=max_val,
                       cmap=cmap)
            plt.scatter(agent.x, agent.y, c=agent.colour, alpha=agent.alpha,
                        s=agent.size)
            plt.text(280, 280, self.current_gen)
            plt.axis('off')
            plt.xlim(0, 300)
            plt.ylim(0, 300)
        # allow for saving of the current model step as an image
        if self.save_img == 1 and self.inf == False:
            savename = str(self.current_gen + 1)
            plt.savefig(savename + ".jpg")

    def initial_vars(self, *args: int) -> None:
        """
        Save the initial dropdown values as variables.

        :param args: Values associated with the dropdown selections
        :type args: tk.IntVar()
        """
        self.num_agent = self.var_agent.get()
        self.num_iter = self.var_iter.get()

    def gen_function(self) -> None:
        """
        Determine the number of generations using a Tk.IntVar().

        The number of generations is selected by a dropdown, additionally
        the number of iterations may be set to infinite.

        In this function, if save as a gif is selection, the images are
        grouped into an array.
        Doing this iteratively preserves the correct order.

        """
        if self.inf == False:
            self.num_iter = self.num_iter
        else:
            self.num_iter = 999999999  # cannot use inf float

        # stop the function if number of iterations are exceeded
        # add one to each current gen for each iteration
        while (self.current_gen < self.num_iter) & (self.carry_on):
            yield self.current_gen
            self.current_gen = self.current_gen + 1
            # append the new image to the list of filenames
            if self.save_img == 1:
                self.filenames.append(str(self.current_gen) + ".jpg")
        # after the gen functions ends, turn images into a gif if selected
        if self.save_img == 1:
            self.create_gif()

    def run(self, *args: int) -> None:
        """
        Initial setup of the agents and environment. Create a mpl animation.

        xy coordinates of the agents here are given as a list from a web table
        td_xs and td_yx.
        The try: except: blocks prevent errors from occurring before dropdown
        selection, and indicate to the user what is required.

        """
        try:
            # state initial variables each time run is clicked
            self.carry_on = True
            self.agents = []
            self.current_gen = 0

            # create the initial environment from the env framework import
            # providing the environment from the Environment class
            self.environment = envf.Environment(self.environment).environment

            # make the agents
            for i in range(self.num_agent):
                # xy values from web input
                x = int(td_xs[i].text)
                y = int(td_ys[i].text)
                # empty list of agents, append each time
                self.agents.append(agf.Agent(self.environment,
                                             self.agents, x, y))
                # randomly shuffle each time a new agent is added
                random.shuffle(self.agents)

                animation = anim.FuncAnimation(  # noqa (animation unused)
                    fig, self.update, frames=self.gen_function, repeat=False
                )
            self.canvas.draw()
        except:  # noqa don't know why this doesn't comply with pep
            print("Error: First choose parameters from dropdowns.")

    def resume(self) -> None:
        """
        Allow for resuming from current iteration after stopping the model

        """
        if (self.current_gen < self.var_iter.get() or self.inf == True):
            self.carry_on = True
            animation = anim.FuncAnimation(  # noqa
                fig, self.update, frames=self.gen_function, repeat=False
            )
            self.canvas.draw()
        else:
            print("Cannot run the model! Check Parameters.")

    def stop(self) -> None:
        """
        Immediately stop the model, displaying the current generation.

        """
        self.carry_on = False

    def toggle_inf(self, *args: int) -> None:
        """
        Allow a button to hold a sunken position to indicate toggling of a
        parameter.

        This merely changes the variable self.inf to 1 and allows the model
        to resume.

        :param args: Values associated with the dropdown selections
        :type args: tk.IntVar()
        """
        if self.inf_btn.config('relief')[-1] == 'sunken':
            self.inf_btn.config(relief="raised")
            print("Running", self.var_iter.get(), "iterations.")
            self.inf = False
            self.carry_on = False

        elif self.save_btn.config("relief")[-1] == 'raised':
            self.inf_btn.config(relief="sunken")
            print("Running infinite iterations.")
            self.inf = True
            self.carry_on = True
        else:
            print("Cannot run infinite iteraions and save as GIF!")

    def toggle_save(self) -> None:
        """
        As with toggle_inf() allows a tkinter button to be sunken.

        Toggles save_img to 1.

        """
        if self.save_btn.config('relief')[-1] == 'sunken':
            self.save_btn.config(relief="raised")
            self.save_img = 0
        elif self.inf_btn.config("relief")[-1] == 'raised':
            self.save_btn.config(relief="sunken")
            self.save_img = 1
        else:
            print("Cannot save as GIF and run infinite iterations!")

    def create_gif(self) -> None:
        """
        Save a GIF of the model from gen 0 to the final generation.

        Uses a list of all images stored during each iteration.

        """
        # get_write and imread allow for a very fast gif creation, despite
        # large files and number
        with imageio.get_writer("model.gif", mode="I",
                                duration=0.5) as writer:
            for filename in self.filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                os.remove(filename)
        print("Creating GIF:", os.getcwd(), "/model.gif")

    def change(self, *args: int) -> None:
        """
        Indicate when parameters and changed and what has changed.

        :param args: Values associated with the dropdown selections
        :type args: tk.IntVar()
        """
        print("Options changed.")
        print("Number of Agents:", self.var_agent.get())
        print("Number of Iterations:", self.var_iter.get())


root = tk.Tk()
gui = AgentGUI(root, env_file='in.txt')

root.resizable(False, False)

# to write docs this cannot be uncommented
#root.mainloop()
