import random
from typing import List


class Agent():
    def __init__(self, environment: List[List[int]], agents: List['Agent'],
                 x: int = None, y: int = None) -> None:
        """
        Create agents for use in a Agent Based Model.

        :param environment: Environment for agents to interact with.
        :type environment: list()
        :param agents: Agents for use in model
        :type agents: list()
        :param x: Agent x location in environment, defaults to None
        :type x: list(), optional
        :param y: Agent y location in environment, defaults to None
        :type y: list(), optional
        """
        self.environment = environment

        # initial location of each agent is a random xy location
        # in the environment or if xy is given as an integer 1-100,
        # this is instead used and multiplied by 3
        if (x is None):
            self.x: int = random.randint(0, len(self.environment))
        else:
            self.x = x*3
        if (y is None):
            self.y: int = random.randint(0, len(self.environment[0]))
        else:
            self.y = y*3

        # initial agent variables
        self.agents = agents
        self.store: int = 10
        self.dead: bool = False
        self.size: int = 1
        self.preg: int = 0
        self.alpha: float = 0.6
        self.age: int = 1
        self.colour: str = "green"

    def distance_between(self, agent) -> int:
        """
        Find euclidean distance between two xy coordinates

        :param agent: Every agent in the model
        :type agent: list()
        """
        distance = ((self.x - agent.x)**2 +
                    (self.y - agent.y)**2)**0.5
        return(distance)

    def move(self) -> None:
        """
        Move each agent randomly with a range of 1 to 5 pixels in positive
        and negative x/y directions

        """
        height: int = len(self.environment)
        width: int = len(self.environment[0])
        rand: int = random.randint(1, 5)

        # modulus ensures that an agent may not randomly move outside the
        # environment by giving the remainder of the calculation
        # if (self.xy +/- rand) > width/height essentially wraps the agent
        # around the environment
        if self.dead is not True:
            if random.random() < 0.5:
                self.x = (self.x + rand) % width
            else:
                self.x = (self.x - rand) % width
            if random.random() < 0.5:
                self.y = (self.y + rand) % height
            else:
                self.y = (self.y - rand) % height

    def eat(self, environment: List[List[int]]) -> None:
        """
        Allow each agent to eat the environment.

        Increase self store,and reduce environment value.
        Agents may eat from a random range surrounding their current position,
        as agents eat more they grow and are able to eat from a large range.
        Agents may only eat green environment.

        :param environment: The environment
        :type environment: list()
        """
        # increase range by size, not a fixed scale to prevent exponential
        # increase
        if self.size < 10:
            y_range = self.size
            x_range = self.size
        elif self.size < 100:
            y_range = round(self.size / 10)
            x_range = round(self.size / 10)
        # don't eat environment if above 100, i.e. a carnivore
        else:
            y_range = 0
            x_range = 0
        if random.random() < 0.25:
            if self.dead == False:
                # randomly eat within the x_range and y_range
                for i in range(- x_range, x_range):
                    for j in range(- y_range, y_range):
                        y_coord = (self.y + j) % len(self.environment[0])
                        x_coord = (self.x + i) % len(self.environment)
                        # only eat environment above 200, (algae)
                        if self.environment[y_coord][x_coord] > 200:
                            self.environment[y_coord][x_coord] -= 1
                            self.store += 1
        # size must be 1/10 the store or they grow too large
        self.size = round(self.store/10)

    def death(self) -> None:
        """
        Determine a state for agents to die.

        Agents die if they do not eat enough for their age.

        """
        if self.age > round(self.store*10):
            if self.dead is not True:
                self.dead = True
                self.alpha = .1
                self.colour = "black"
                print("An agent died of hunger.")

    def make_baby(self) -> None:
        """
        Allow production of more agents given certain conditions.

        Agents must be within a fixed proximity determined by the
        distance_between() function.
        Pregnancy increases with further interaction until a baby is born.

        """
        for agent in self.agents:
            distance = self.distance_between(agent)
        if random.random() < 0.5:
            if distance < 20:
                self.preg += 1

        if self.preg == 4:
            if self.dead is not True:
                print("A baby is born.")
                self.agents.append(Agent(self.environment, self.agents,
                                         self.x, self.y))
                self.preg = 0

    def carnivore(self) -> None:
        """
        At a certain stage agents will grow into carnivores

        Carnivores combine their current store with that of the agent
        they have eaten.
        Carnivores may no longer eat the environment.
        Larger carnivores eat smaller ones.

        """
        if self.size > 100:
            self.colour = "red"
            self.alpha = 1
            self.preg = 0
            for agent in self.agents:
                distance = self.distance_between(agent)
                if distance < 20:
                    # the larger carnivore eats the smaller carnivore
                    if self.store > agent.store & agent.store > 0:
                        self.store = self.store + agent.store
                        agent.dead = True
                        # remove the eaten agent
                        agent.x = -999
                        agent.y = -999
                        print("A predator eats another, gaining", agent.store,
                              "resources.")
                        agent.store = 0
