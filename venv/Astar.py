import numpy as np
import heapq as hq
import copy
import time as t

heuristics = 1 # heuristics
totalNodes = 0
analyzedNodes = 0

class Node():
    def __init__(self, value, distance, anc, state, move):
        self.value = value
        self.distance = distance
        self.combo = distance + value # combination of distance and heuristic value
        self.ancestor = anc
        self.move = move
        self.state = state

    def __lt__(self, other): # heapq requirement
        if self.combo == other.combo:
            return self.value < other.value
        else:
            return self.combo < other.combo


initialMat = np.array([[3, 2, 5, 0], [7, 6, 1, 4]])
targetMat = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])


def recognizeHeur(initial, target): # a function to choose the right heuristic function
    if heuristics == 1:
        return heuristics_one(initial, target)
    elif heuristics == 2:
        return heuristics_two(initial, target)
    return None


def heuristics_one(initial, target):
    m, n = initial.shape # shape
    cnt = 0

    for y in range(m): # y-axis
        for x in range(n): # x-axis
            if target[y][x] == 0:
                continue

            if initial[y][x] != target[y][x]:
                cnt += 1

    return cnt


def heuristics_two(initial, target):
    m, n = initial.shape # shape
    cnt = 0

    for y in range(m): # y-axis
        for x in range(n): # x-axis
            if initial[y][x] == 0:
                continue

            if initial[y][x] != target[y][x]:
                posY, posX = np.where(target == initial[y][x]) # find y and x positions of the item in target matrix
                cnt += abs(x - int(posX)) + abs(y - int(posY)) # calculate the difference of the positions

    return cnt


def up(state, x, y): # swap up
    matrix = copy.deepcopy(state)
    matrix[y][x], matrix[y-1][x] = matrix[y-1][x], matrix[y][x]
    return matrix


def down(state, x, y): # swap down
    matrix = copy.deepcopy(state)
    matrix[y][x], matrix[y+1][x] = matrix[y+1][x], matrix[y][x]
    return matrix


def left(state, x, y): # swap left
    matrix = copy.deepcopy(state)
    matrix[y][x], matrix[y][x-1] = matrix[y][x-1], matrix[y][x]
    return matrix


def right(state, x, y): # swap right
    matrix = copy.deepcopy(state)
    matrix[y][x], matrix[y][x+1] = matrix[y][x+1], matrix[y][x]
    return matrix


def aStar(initial, target):
    global totalNodes, analyzedNodes

    minHeap = [] # states
    created = {} # generated states
    initNode = Node(recognizeHeur(initial, target), 0, None, initial, None)
    # hq.heappush(minHeap, initNode)
    m, n = initial.shape

    while initNode.value != 0:
        posY, posX = np.where(initNode.state == 0)

        if posY - 1 >= 0: # up
            modState = up(initNode.state, int(posX), int(posY)) # create new state
            totalNodes += 1
            if created.get(str(modState)) is None:
                created[str(modState)] = True
                node = Node(recognizeHeur(modState, target), initNode.distance + 1, initNode, modState, 'UP')
                hq.heappush(minHeap, node)
                analyzedNodes += 1

        if posY + 1 < m: # down
            modState = down(initNode.state, int(posX), int(posY)) # create new state
            totalNodes += 1
            if created.get(str(modState)) is None:
                created[str(modState)] = True
                node = Node(recognizeHeur(modState, target), initNode.distance + 1, initNode, modState, 'DOWN')
                hq.heappush(minHeap, node)
                analyzedNodes += 1

        if posX - 1 >= 0: # left
            modState = left(initNode.state, int(posX), int(posY)) # create new state
            totalNodes += 1
            if created.get(str(modState)) is None:
                created[str(modState)] = True
                node = Node(recognizeHeur(modState, target), initNode.distance + 1, initNode, modState, 'LEFT')
                hq.heappush(minHeap, node)
                analyzedNodes += 1

        if posX + 1 < n: # right
            modState = right(initNode.state, int(posX), int(posY)) # create new state
            totalNodes += 1
            if created.get(str(modState)) is None:
                created[str(modState)] = True
                node = Node(recognizeHeur(modState, target), initNode.distance + 1, initNode, modState, 'RIGHT')
                hq.heappush(minHeap, node)
                analyzedNodes += 1

        initNode = hq.heappop(minHeap)
    return initNode


def printTrace(node: Node): # print the trace of the algorithm
    if node.ancestor is None:
        return
    printTrace(node.ancestor)
    print('\n', node.move, sep='') # print moves
    print(node.state) # print state matrices


def main():
    t_start = t.time()
    output = aStar(initialMat, targetMat)
    t_end = t.time()

    printTrace(output)
    print(f'\nTime needed to generate {totalNodes} nodes and analyze {analyzedNodes} out of them: {round(t_end - t_start, 5)}s with final depth: {output.distance}')

if __name__ == '__main__':
    main()