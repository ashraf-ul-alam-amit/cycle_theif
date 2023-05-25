import heapq

class KDNode:
    def __init__(self, point, axis, left=None, right=None):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, points):
        def build_kdtree(points, depth=0):
            if not points:
                return None

            axis = depth % len(points[0])
            sorted_points = sorted(points, key=lambda point: point[axis])
            mid = len(points) // 2

            return KDNode(
                sorted_points[mid],
                axis,
                build_kdtree(sorted_points[:mid], depth + 1),
                build_kdtree(sorted_points[mid + 1:], depth + 1),
            )

        self.root = build_kdtree(points)

    def get_nearest(self, target_point):
        def search_kdtree(node, target_point, nearest_point, nearest_distance, depth=0):
            if not node:
                return nearest_point, nearest_distance

            distance = sum((node.point[i] - target_point[i]) ** 2 for i in range(len(target_point)))
            if distance < nearest_distance:
                nearest_point, nearest_distance = node.point, distance

            axis = depth % len(target_point)
            if target_point[axis] < node.point[axis]:
                nearest_point, nearest_distance = search_kdtree(node.left, target_point, nearest_point, nearest_distance, depth + 1)
                if node.right and (node.point[axis] - target_point[axis]) ** 2 < nearest_distance:
                    nearest_point, nearest_distance = search_kdtree(node.right, target_point, nearest_point, nearest_distance, depth + 1)
            else:
                nearest_point, nearest_distance = search_kdtree(node.right, target_point, nearest_point, nearest_distance, depth + 1)
                if node.left and (node.point[axis] - target_point[axis]) ** 2 < nearest_distance:
                    nearest_point, nearest_distance = search_kdtree(node.left, target_point, nearest_point, nearest_distance, depth + 1)

            return nearest_point, nearest_distance

        return search_kdtree(self.root, target_point, None, float('inf'))[0]


def closest_red_black_pairs(red_points, black_points):
    # Separate the points and IDs into two separate lists
    red_ids, red_points = zip(*red_points)
    black_ids, black_points = zip(*black_points)

    # Construct a KD-Tree from the red points
    tree = KDTree(list(red_points))

    # Find the nearest red point to each black point
    nearest_red_ids = {}
    for i, black_point in enumerate(black_points):
        nearest_red_point = tree.get_nearest(black_point)
        nearest_red_index = red_points.index(nearest_red_point)
        nearest_red_id = red_ids[nearest_red_index]
        nearest_red_ids[black_ids[i]] = nearest_red_id

    return nearest_red_ids


# red_points = [(1, (1, 2)), (2, (3, 4)), (3, (5, 6))]
# black_points = [(1, (2, 3)), (2, (4, 5)), (3, (7, 8))]

# nearest_red_ids = closest_red_black_pairs(red_points, black_points)

# print(nearest_red_ids)
