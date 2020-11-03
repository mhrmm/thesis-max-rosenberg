"""
class Taxonomy:

    def is_instance(self, node):
        raise NotImplementedError('Cannot call this method on abstract class.')

    def is_category(self, node):
        raise NotImplementedError('Cannot call this method on abstract class.')

    def num_instances(self):
        raise NotImplementedError('Cannot call this method on abstract class.')

    def get_root(self):
        raise NotImplementedError('Cannot call this method on abstract class.')

    def get_categories(self):
        return NotImplementedError('Cannot call this method on abstract class.')

    def get_instances(self):
        return NotImplementedError('Cannot call this method on abstract class.')

    def get_ancestor_categories(self, node):
        raise NotImplementedError('Cannot call this method on abstract class.')

    def get_descendant_instances(self, node):
        raise NotImplementedError('Cannot call this method on abstract class.')
"""


def flatness(taxonomy, node):
    """
    The ratio of children to total descendents of a category.
    A 'flatter' node will have a higher flatness.

    TODO: consider an alternative based on average branching factor in the
    graph rooted at that node

    """
    num_total_hyps = len(taxonomy.get_descendants(node)) - 1
    if num_total_hyps <= 0:
        return 0
    return len(taxonomy.get_children(node)) / num_total_hyps


def repetitions(self, node):
    """
    Revise this so that:
    - it counts the number of instances that can be reached via
    more than 1 child. (for fruit example, "entity" == 2) <-- orange and peach
    - or: the average number of children through which we can access an instance
    (for fruit example "entity" = (2+2+1+1+1+1) / 6

    """
    if node == self.get_root_node():
        return self.get_descendents(self.get_root_node()).count(node) + 1
    else:
        return self.get_descendents(self.get_root_node()).count(node)


def wu_palmer_similarity(taxonomy, node1, node2):
    """
    Similarity metric from Wu and Palmer (1994).

    Given two nodes node 1 and node 2,
    the Wu Palmer similarity of the two nodes is the depth of the lowest
    common ancestor of the two nodes divided by the sum of the depths of
    the two nodes. This ratio is then multiplied by two.

    """
    # Get dicts of hypernym:distance from node to hypernym (AKA index of list)
    node1_ancestor_distances = dict()
    node1_ancestors = sorted(taxonomy.get_ancestor_categories(node1))
    for h in node1_ancestors:
        node1_ancestor_distances[h] = node1_ancestors.index(h)

    node2_ancestor_distances = dict()
    node2_ancestors = sorted(taxonomy.get_ancestor_categories(node2))
    for h in node2_ancestors:
        node2_ancestor_distances[h] = node2_ancestors.index(h)

    # Find the common hypernyms between the two nodes
    common = set(node1_ancestors) & set(node2_ancestors)

    # get sums of distances of common hypernyms, then get the word with minimum sum
    candidates = dict()
    for c in common:
        candidates[c] = node1_ancestor_distances[c] + node2_ancestor_distances[c]

    lowest_common_ancestor = min(candidates, key=candidates.get)

    node1_lca_distance = node1_ancestor_distances[lowest_common_ancestor]
    node2_lca_distance = node2_ancestor_distances[lowest_common_ancestor]
    node3_distance = len(taxonomy.get_ancestor_categories(lowest_common_ancestor)) - 1
    numerator = 2 * node3_distance
    denominator = node1_lca_distance + node2_lca_distance + (2 * node3_distance)
    print(node1_lca_distance, node2_lca_distance, node3_distance)

    if denominator == 0:
        return 0
    return numerator / denominator


def rosenberg_descendent_similarity(self, node):
    total = 0
    num_tests = 10
    descendents = self.get_descendents(node)
    for d in descendents:
        d_sim_total = 0
        for _ in range(num_tests):
            x = self.random_descendents(node, 1)[0]
            d_sim_total += self.wu_palmer_similarity(d, x)
        d_sim_avg = d_sim_total / num_tests
        total += d_sim_avg
    if len(descendents) == 0:
        return 0
    avg = total / len(descendents)
    return avg
