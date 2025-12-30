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


class DijkstraNodes:
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

                heapq.heappush(self.pq, (cost, (x, y)))


def draw(state, grid_map):
    # 0=free, 1=obstacle, 2=visited, 3=current, 4=path
    cmap = plt.cm.colors.ListedColormap(
        ["white", "black", "#9fd3ff", "yellow", "red"]
    )

    plt.clf()
    plt.imshow(state, cmap=cmap, vmin=0, vmax=4)
    plt.title("<Dijkstra Algorithm>", fontsize=20)

    sx, sy = grid_map.start
    gx, gy = grid_map.goal
    plt.scatter(sx, sy, c="green", s=80)
    plt.scatter(gx, gy, c="red", s=80)

    plt.axis("equal")
    plt.axis("off")
    plt.pause(0.001)

def reconstruct_path(goal_node: Node, dnodes: DijkstraNodes):
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

    dnodes = DijkstraNodes(grid_map) # 1.맵 전체 픽셀을 노드화(장애물 제외) 스타트 노드는 0, 그 외의 노드는 인피니티 그리고 전체 노드를 pq에 집어넣음.
    goal_node = None

    while dnodes.pq: #Priority Queue가 비워질 때 까지 반복
        cost, (x, y) = heapq.heappop(dnodes.pq) #우선, pq안에서 cost가 가장 작은 노드를 빼냄.
        current = dnodes.nodes[(x, y)] #현재 노드를 저장.

        if cost > current.cost: #만약 현재 노드의 cost가 과거 탐색한 cost 최솟값보다 크다면 업데이트 안하고 무시.
            continue
        if current.visited: #이전에 탐색한 노드도 무시
            continue

        state[y, x] = 3
        draw(state, grid_map)

        current.visited = True
        state[y, x] = 2

        if (x, y) == grid_map.goal: #골 포인트에 도달하면 반복문 종료
            goal_node = current
            break

        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]: #이웃한 노드 탐색 상하좌우 4방향 탐색
            nx, ny = x + dx, y + dy
            if (nx, ny) not in dnodes.nodes: #이웃 노드가 장애물이라면 무시.
                continue

            neighbor = dnodes.nodes[(nx, ny)] #이웃 노드를 neighbor 노드변수에 저장.
            if neighbor.visited: #이미 방문한 이웃 노드라면 무시
                continue

            new_cost = current.cost + 1 #모든 노드간의 엣지는 1로 규정하며, 이웃 노드를 탐색하기 위해서 1의 cost 증가
            if new_cost < neighbor.cost: #만약 이웃 노드의 cost가 최소값을 가진다면, 업데이트 후 경로 저장 후, pq에 넣음.
                neighbor.cost = new_cost
                neighbor.parent = (x, y)
                heapq.heappush(dnodes.pq, (neighbor.cost, (nx, ny)))

    # draw final path
    if goal_node is not None:
        path = reconstruct_path(goal_node, dnodes)
        for x, y in path:
            state[y, x] = 4
        draw(state, grid_map)

    plt.show()


if __name__ == "__main__":
    main()
