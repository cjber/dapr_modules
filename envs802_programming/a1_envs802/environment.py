import random
from typing import List


class Environment():
    def __init__(self, environment: List[List[int]]) -> None:
        """
        Allows for manipulation of the environment without agents.

        :param environment: Environment values
        :type environment: list()
        :param x: All x coordinates in the environment, defaults to None
        :type x: list(), optional
        :param y: All y coordinates in the environment, defaults to None
        :type y: list(), optional
        """
        self.environment = environment

    # takes input environment which is the updated environment each iteration
    def grow_algae(self, environment: List[List[int]]):
        """
        Randomly grow the green portion of the environment every iteration.

        Growth only occurs in close proximity to existing algae.

        :param environment: The input environment.
        :type environment: list()
        """
        x: int = random.randint(0, len(self.environment))
        y: int = random.randint(0, len(self.environment[0]))
        y_range: int = random.randint(1, 5)
        x_range: int = random.randint(1, 5)
        rand: int = random.randrange(-10, 10)

        # allow random (range 1-5) 'growth' (environment > 200)
        # if within -10, 10 rand range of current algae
        for i in range(- x_range, x_range):
            for j in range(- y_range, y_range):
                y_coord: int = (y + j) % 300
                x_coord: int = (x + i) % 300
                if self.environment[y_coord][x_coord] < 200:
                    if self.environment[(y_coord + rand)
                                        % 300][(x_coord + rand) % 300] > 200:
                        if random.random() < 0.3:
                            self.environment[y_coord][x_coord] = 201
                        else:
                            self.environment[y_coord][x_coord]
