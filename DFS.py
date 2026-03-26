#AI quiz practice
def dfs(graph, node, visited=None):
    if visited is None:
        visited =set()
    visited.add(node)
    print(node)

    for neighbour in graph[node]:
        if neighbour not in visited:
            dfs(graph, neighbour, visited)

    return visited