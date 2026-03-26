import heapq

def greedy_bfs(graph, start, goal, heuristic):
    visited=set()
    queue=[(heuristic[start], start)]

    while queue:
        cost, node=heapq.heappop(queue)

        if node==goal:
            return True

        if node in visited:
            continue

        visited.add(node)
        print(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                heapq.heappush(queue, (heuristic[neighbor], neighbor))

    return False