from numpy import linalg

class MyGeniousKDTree:
    
    class Node:
        def __init__(self, point, left=None, right=None):
            self.point = point
            self.left = left
            self.right = right


    @staticmethod
    def build_kd_tree(points, depth=0):
        if not points:
            return None
        k = len(points[0]) - 1
        axis = depth % k
        points.sort(key=lambda x: x[axis])
        median = len(points) // 2
        return MyGeniousKDTree.Node(
            points[median],
            MyGeniousKDTree.build_kd_tree(points[:median], depth + 1),
            MyGeniousKDTree.build_kd_tree(points[median + 1:], depth + 1)
        )

    @staticmethod
    def distance(point1, point2):
        return linalg.norm(point1[:-1] - point2)


    @staticmethod
    def nearest_neighbors(root, query_point, k):
        best_points = []

        def search(node):
            if node is not None:
                dist = MyGeniousKDTree.distance(node.point, query_point)
                if len(best_points) < k:
                    best_points.append((node.point, dist))
                    best_points.sort(key=lambda x: x[1])
                elif dist < best_points[-1][1]:
                    best_points.pop()
                    best_points.append((node.point, dist))
                    best_points.sort(key=lambda x: x[1])

                axis = len(node.point) % len(query_point)
                if query_point[axis] < node.point[axis]:
                    search(node.left)
                    if (node.point[axis] - query_point[axis]) ** 2 < best_points[-1][1]:
                        search(node.right)
                else:
                    search(node.right)
                    if (query_point[axis] - node.point[axis]) ** 2 < best_points[-1][1]:
                        search(node.left)

        search(root)

        return [point for point, _ in best_points]