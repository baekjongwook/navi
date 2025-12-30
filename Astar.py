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
    cost: float          # g-cost
    visited: bool = False
    parent: tuple | None = None

class GridMap:
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self.grid = np.zeros((self.h, self.w), dtype=int)
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
        max_r = max(self.w, self.h)

        for r in range(max_r):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    x, y = sx + dx, sy + dy
                    if self._in_bounds(x, y) and self.grid[y, x] == 0:
                        return (x, y)
        raise RuntimeError("No free cell found")

    def _in_bounds(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h

def heuristic(x, y, goal):
    gx, gy = goal
    return abs(x - gx) + abs(y - gy)   # Manhattan distance

class Astar:  
    def __init__(self, grid_map: GridMap):
        self.nodes = {}
        self.pq = []

        sx, sy = grid_map.start

        for y in range(grid_map.h):
            for x in range(grid_map.w):

                if grid_map.grid[y, x] == 1:
                    continue

                if (x, y) == (sx, sy):
                    cost = 0.0
                else:
                    cost = math.inf

                self.nodes[(x, y)] = Node(x, y, cost)

        h0 = heuristic(sx, sy, grid_map.goal)
        heapq.heappush(self.pq, (h0, (sx, sy)))

def draw(state, grid_map):
    cmap = plt.cm.colors.ListedColormap(
        ["white", "black", "#9fd3ff", "yellow", "red"]
    )

    plt.clf()
    plt.imshow(state, cmap=cmap, vmin=0, vmax=4)
    plt.title("<A* Algorithm>", fontsize=20)

    sx, sy = grid_map.start
    gx, gy = grid_map.goal
    plt.scatter(sx, sy, c="green", s=80)
    plt.scatter(gx, gy, c="red", s=80)

    plt.axis("equal")
    plt.axis("off")
    plt.pause(0.001)

def reconstruct_path(goal_node: Node, dnodes: Astar):
    path = []
    node = goal_node

    while node is not None:
        path.append((node.x, node.y))
        if node.parent is None:
            break
        node = dnodes.nodes[node.parent]

    path.reverse()
    return path

def main():
    grid_map = GridMap(width=20, height=20)
    grid_map.generate_maze_like_map()

    state = np.zeros_like(grid_map.grid, dtype=int)
    state[grid_map.grid == 1] = 1

    plt.figure(figsize=(8, 6))

    dnodes = Astar(grid_map)
    goal_node = None

    while dnodes.pq:
        f, (x, y) = heapq.heappop(dnodes.pq)
        current = dnodes.nodes[(x, y)]

        if current.visited:
            continue

        state[y, x] = 3
        draw(state, grid_map)

        current.visited = True
        state[y, x] = 2

        if (x, y) == grid_map.goal:
            goal_node = current
            break

        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in dnodes.nodes:
                continue

            neighbor = dnodes.nodes[(nx, ny)]
            if neighbor.visited:
                continue

            g_new = current.cost + 1
            if g_new < neighbor.cost:
                neighbor.cost = g_new
                neighbor.parent = (x, y)

                h_new = heuristic(nx, ny, grid_map.goal)
                f_new = g_new + h_new
                heapq.heappush(dnodes.pq, (f_new, (nx, ny)))

    if goal_node is not None:
        path = reconstruct_path(goal_node, dnodes)
        for x, y in path:
            state[y, x] = 4
        draw(state, grid_map)

    plt.show()


if __name__ == "__main__":
    main()
