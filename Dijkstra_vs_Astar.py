import random
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Node:
    x: int
    y: int
    cost: float
    visited: bool = False
    parent: tuple | None = None


class GridMap:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.grid = np.zeros((height, width), dtype=int)
        self.start = None
        self.goal = None

    def generate_maze_like_map(self):
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        for y in range(2, self.h - 2, 2):
            for x in range(2, self.w - 2, 2):
                self.grid[y, x] = 1
                dx, dy = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
                self.grid[y + dy, x + dx] = 1

        self.start = self._find_nearest_free((1, 1))
        self.goal = self._find_nearest_free((self.w - 2, self.h - 2))

    def _find_nearest_free(self, seed):
        sx, sy = seed
        for r in range(max(self.w, self.h)):
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    x, y = sx+dx, sy+dy
                    if 0 <= x < self.w and 0 <= y < self.h:
                        if self.grid[y, x] == 0:
                            return (x, y)
        raise RuntimeError("No free cell")


def heuristic(x, y, goal):
    gx, gy = goal
    return abs(x-gx) + abs(y-gy)


class DijkstraSolver:
    def __init__(self, grid_map):
        self.nodes = {}
        self.pq = []
        self.finished = False
        self.goal_node = None

        sx, sy = grid_map.start

        for y in range(grid_map.h):
            for x in range(grid_map.w):
                if grid_map.grid[y, x] == 1:
                    continue
                cost = 0.0 if (x, y) == (sx, sy) else math.inf
                self.nodes[(x, y)] = Node(x, y, cost)
                heapq.heappush(self.pq, (cost, (x, y)))

    def step(self, grid_map):
        if not self.pq or self.finished:
            return

        cost, (x, y) = heapq.heappop(self.pq)
        cur = self.nodes[(x, y)]

        if cost > cur.cost or cur.visited:
            return

        cur.visited = True

        if (x, y) == grid_map.goal:
            self.goal_node = cur
            self.finished = True
            return

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if (nx, ny) not in self.nodes:
                continue

            nb = self.nodes[(nx, ny)]
            if nb.visited:
                continue

            new_cost = cur.cost + 1
            if new_cost < nb.cost:
                nb.cost = new_cost
                nb.parent = (x, y)
                heapq.heappush(self.pq, (nb.cost, (nx, ny)))


class AStarSolver:
    def __init__(self, grid_map):
        self.nodes = {}
        self.pq = []
        self.finished = False
        self.goal_node = None

        sx, sy = grid_map.start

        for y in range(grid_map.h):
            for x in range(grid_map.w):
                if grid_map.grid[y, x] == 1:
                    continue
                cost = 0.0 if (x, y) == (sx, sy) else math.inf
                self.nodes[(x, y)] = Node(x, y, cost)

        h0 = heuristic(sx, sy, grid_map.goal)
        heapq.heappush(self.pq, (h0, (sx, sy)))

    def step(self, grid_map):
        if not self.pq or self.finished:
            return

        _, (x, y) = heapq.heappop(self.pq)
        cur = self.nodes[(x, y)]

        if cur.visited:
            return

        cur.visited = True

        if (x, y) == grid_map.goal:
            self.goal_node = cur
            self.finished = True
            return

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if (nx, ny) not in self.nodes:
                continue

            nb = self.nodes[(nx, ny)]
            if nb.visited:
                continue

            g_new = cur.cost + 1
            if g_new < nb.cost:
                nb.cost = g_new
                nb.parent = (x, y)
                h_new = heuristic(nx, ny, grid_map.goal)
                f = g_new + h_new
                heapq.heappush(self.pq, (f, (nx, ny)))


def draw(ax, solver, grid_map, title):
    state = np.zeros_like(grid_map.grid)
    state[grid_map.grid == 1] = 1

    for n in solver.nodes.values():
        if n.visited:
            state[n.y, n.x] = 2

    if solver.finished:
        node = solver.goal_node
        while node:
            state[node.y, node.x] = 4
            node = solver.nodes[node.parent] if node.parent else None

    cmap = plt.cm.colors.ListedColormap(
        ["white", "black", "#9fd3ff", "red", "yellow"]
    )

    ax.imshow(state, cmap=cmap, vmin=0, vmax=4)
    sx, sy = grid_map.start
    gx, gy = grid_map.goal
    ax.scatter(sx, sy, c="green")
    ax.scatter(gx, gy, c="red")
    ax.set_title(title)
    ax.axis("off")


def main():
    grid_map = GridMap(20, 20)
    grid_map.generate_maze_like_map()

    dijkstra = DijkstraSolver(grid_map)
    astar = AStarSolver(grid_map)

    plt.figure(figsize=(12, 6))

    while not (dijkstra.finished and astar.finished):
        dijkstra.step(grid_map)
        astar.step(grid_map)

        plt.clf()
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

        draw(ax1, dijkstra, grid_map, "Dijkstra")
        draw(ax2, astar, grid_map, "A*")

        plt.pause(0.01)

    plt.show()


if __name__ == "__main__":
    main()
