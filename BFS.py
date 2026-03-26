def BFS(graph, start):
    visited=set()
    queue=[start]

    while queue:
        node=queue.pop(0)
        if node not in visited:
            visited.add(node)
            print(node)
            
            for neighbour in graph[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
    return visited