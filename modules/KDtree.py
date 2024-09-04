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


def closest_cycle_person_pairs(cycle_points, person_points):
    # Separate the points and IDs into two separate lists
    cycle_ids, cycle_points = zip(*cycle_points)
    person_ids, person_points = zip(*person_points)

    # Construct a KD-Tree from the cycle points
    tree = KDTree(list(cycle_points))

    # Find the nearest cycle point to each person point
    nearest_cycle_ids = {}
    for i, person_point in enumerate(person_points):
        nearest_cycle_point = tree.get_nearest(person_point)
        nearest_cycle_index = cycle_points.index(nearest_cycle_point)
        nearest_cycle_id = cycle_ids[nearest_cycle_index]
        nearest_cycle_ids[person_ids[i]] = nearest_cycle_id

    return nearest_cycle_ids


# red_points = [(1, (1, 2)), (2, (3, 4)), (3, (5, 6))]
# black_points = [(1, (2, 3)), (2, (4, 5)), (3, (7, 8))]

# nearest_red_ids = closest_cycle_person_pairs(red_points, black_points)

# print(nearest_red_ids)
