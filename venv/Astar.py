import numpy as np

# class Node():
#     def __init__(self, value, anc, state):
#         self.value = value
#         self.ancestor = anc
#         self.state = state


initialMat = np.array([[7, 8, 6],
                      [5, 4, 3],
                      [2, 0, 1]])

targetMat = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 0]])


def heuristics_one(initial, target):
    m, n = initial.shape # shape
    cnt = 0
    for y in range(n): # y-axis
        for x in range(m): # x-axis
            if target[y][x] == 0:
                continue
            if initial[y][x] != target[y][x]:
                cnt += 1

    return cnt


def heuristics_two(initial, target):
    m, n = initial.shape
    cnt = 0
    for y in range(m):
        for x in range(n):
            if initial[y][x] == 0:
                continue
            if initial[y][x] != target[x][y]:
                posY, posX = np.where(target == initial[y][x])
                cnt += abs(x - int(posX)) + abs(y - int(posY))
    return cnt


def main():
    print(f'Heuristika 1: {heuristics_one(initialMat, targetMat)}')
    print(f'Heuristika 2: {heuristics_two(initialMat, targetMat)}')


if __name__ == '__main__':
    main()