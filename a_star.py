import heapq

def a_star(graph, start, goal, heuristic):
    visited=set()
    queue=[(heuristic[start], 0,start)]

    while queue:
        priority, cost, node=heapq.heappop(queue)

        if node == goal:
            return cost

        if node in visited :
            continue

        visited.add(node)
        print(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                new_cost=cost+1
                new_priority=new_cost+heuristic[neighbor]
                heapq.heappush(queue, (new_priority, new_cost, neighbor))

    return None