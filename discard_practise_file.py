def dfs(graph, node, visited=None):
    if visited is None:
        visited=set()
    visited.add(node)
    print(node)

    for neighbour in graph[node]:
        if neighbour not in visited:
            dfs(graph, neighbour, visited)
    return visited

def dfs_iterative(graph, start):
    visited=set()
    stack=[start]

    while stack:
        node=stack.pop()
        
        if node not in visited:
            visited.add(node)
            print(node)

            for neighbour in graph[node]:
                if neighbour not in visited:
                    stack.append(neighbour)
    return visited
