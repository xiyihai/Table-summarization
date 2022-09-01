import warnings,numpy as np
from munkres import Munkres


"""
Summary of this module here.

The module is used to select k representative entities from a set of entities
"""

class Entity:
    def __init__(self, name, domains, nb):
        self.name = name
        self.domains = domains
        self.nb = nb


class Summary:
    """
     Summary of this class here.

     Summary class takes the entity notability, entity domain information, the summary size as input
     and outputs the k size summary consisting of k entities (including name, notability score, domains).

     usage example:
     summary = Summary()
     summary.outputSummary()
    """

    def __init__(self):
        self.sim_metrix = {}


    def outputSummary(self, nb_file, domains_file,  k):
        """
        The main function of Summary class
        :param nb_file: the file with the notability score for each entity
        :param domains_file: the file with the domain information for each entity
        :param k: the summary size
        :return: a k-size summary with Entity class (name, domains, nb)
        """

        with open(domains_file, 'r', encoding='utf-8') as f:
            domains = eval(f.read())

        entity_domains = domains['entity_domains']
        self.sim_metrix = domains['metrix']

        with open(nb_file, 'r', encoding='utf-8') as f:
            entity_notability = eval(f.read())


        entities = []
        for e,d in entity_domains.items():
            if len(d) != 0 and e in entity_notability:
                entities.append(Entity(e,d,entity_notability[e]))

        domain_centrality = self.domainCentrality(entities)

        centroids = self.initK_center(entities, domain_centrality, k)

        flag = 1
        k = 0
        while flag:
            clusters = self.classify(entities, centroids, self.nb_distance)
            centroids, flag = self.get_new_center(clusters, self.nb_distance)
            k += 1

        return centroids


    # find the optimized initial centroids
    def initK_center(self, entities, domain_centrality, k):
        centroids = []
        candidate_e = set()
        sorted_entities = sorted(entities, key=lambda k: k.nb, reverse=True)

        for e in entities:
            if e.nb >= 0.2:
                candidate_e.add(e)

        while len(centroids) < k:
            if len(candidate_e) > 0:
                max_nb = 0
                max_e = -1
                for e in candidate_e:
                    if e.nb > max_nb:
                        max_nb = e.nb
                        max_e = e

                subset = set()
                for e in candidate_e:
                    if self.dom_distance_km(max_e, e) <= 0.3:
                        subset.add(e)

                center_e = -1
                center_nb = 0
                for e in subset:
                    nb = 0
                    for d in e.domains:
                        nb = nb + domain_centrality[d]
                    if nb > center_nb:
                        center_e = e
                        center_nb = nb
                centroids.append(center_e)

                candidate_e = candidate_e - subset
            else:
                can_e = []
                for e in entities:
                    if e not in centroids and len(e.domains) > 0:
                        flag = 1
                        for c in centroids:
                            if self.dom_distance_km(c, e) <= 0.3:
                                flag = 0
                        if flag == 1:
                            can_e.append(e)
                cans = sorted(can_e, key=lambda k: k.nb, reverse=True)
                if len(cans) > 0:
                    centroids.append(cans[0])
                else:
                    for e in sorted_entities:
                        if e not in centroids:
                            centroids.append(e)
        return centroids


    def classify(self, entities, centroids, distance_func):
        clusters = {}
        for center in centroids:
            clusters[center] = []

        for e in entities:
            min = 100
            closest_center = -1
            for center in centroids:
                if e == center:
                    closest_center = center
                    min = 0
                    break
                else:
                    dist = distance_func(center, e)
                    if dist < min:
                        min = dist
                        closest_center = center
            clusters[closest_center].append(e)

        return clusters


    def get_new_center(self, clusters, distance_func):
        centroids = []
        flag = 0

        for (center, clu_entities) in clusters.items():

            min_dis_cluster = 0
            new_center = center

            for e in clu_entities:
                min_dis_cluster += max(0, distance_func(center, e))

            for new_c in clu_entities:
                dist = 0
                for non_center in clu_entities:
                    dist += max(0, distance_func(new_c, non_center))
                if dist < min_dis_cluster:
                    min_dis_cluster = dist
                    new_center = new_c
                    flag = 1

            centroids.append(new_center)
        return centroids, flag


    def nb_distance(self, center, e1):
        loss = 0
        for d1 in e1.domains:
            max_sim = 0
            for d2 in center.domains:
                sim = float(self.sim_metrix[d1][d2])
                if sim < 0.7:
                    sim = 0
                if sim > max_sim:
                    max_sim = sim
            loss += max(0, e1.nb - center.nb * max_sim)

        if loss == 0:
            loss = -(1 - self.dom_distance_km(center, e1))
        return loss


    def dom_distance_km(self, e1, e2):
        dom1 = e1.domains
        dom2 = e2.domains

        len1 = len(dom1)
        len2 = len(dom2)
        if len1 == 0 or len2 == 0:
            return 0
        else:
            N = max(len(dom1), len(dom2))
            matrix = np.zeros((N, N))
            for i, d1 in enumerate(dom1):
                for j, d2 in enumerate(dom2):
                    matrix[i][j] = float(self.sim_metrix[d1][d2])

            cost_matrix = []
            for row in matrix:
                cost_row = []
                for col in row:
                    cost_row += [1 - col]
                cost_matrix += [cost_row]

            m = Munkres()
            indexes = m.compute(cost_matrix)
            total = 0
            for row, column in indexes:
                value = matrix[row][column]
                total += value
            return 1 - total / N


    # calculate the domain centrality for each domain
    def domainCentrality(self, entities):
        edges = []
        nodes = []
        domains = []

        for e in entities:
            for d in e.domains:
                edges.append([e, d])
                if d not in domains:
                    domains.append(d)

        for d1 in domains:
            for d2 in domains:
                if d1 != d2 and float(self.sim_metrix[d1][d2]) >= 0.4:
                    edges.append([d1, d2])

        for edge in edges:
            if edge[0] not in nodes:
                nodes.append(edge[0])
            if edge[1] not in nodes:
                nodes.append(edge[1])

        N = len(nodes)
        i = 0
        node_to_num = {}
        num_to_node = {}
        for node in nodes:
            node_to_num[node] = i
            num_to_node[i] = node
            i += 1

        for edge in edges:
            edge[0] = node_to_num[edge[0]]
            edge[1] = node_to_num[edge[1]]

        S = np.zeros([N, N])
        for edge in edges:
            S[edge[1], edge[0]] = 1

        for j in range(N):
            sum_of_col = sum(S[:, j])
            for i in range(N):
                if sum_of_col == 0:
                    S[i, j] = 0
                else:
                    S[i, j] /= sum_of_col

        nb_vector = np.ones(N) / N
        ft = 0.15
        nb_list = []
        for node in nodes:
            if hasattr(node, 'nb'):
                nb_list.append(node.nb)
            else:
                nb_list.append(0)

        sum_nb = sum([value for value in nb_list])

        nb_vector = np.array([float(s) / sum_nb for s in nb_list])

        A = (1 - ft) * S

        P_n = np.ones(N) / N
        P_n1 = np.zeros(N)

        e = 100000

        while e > 0.00000001:
            P_n1 = np.dot(A, P_n) + ft * nb_vector
            e = P_n1 - P_n
            e = max(map(abs, e))
            P_n = P_n1

        domain_centrality = {}
        for i, node in enumerate(nodes):
            domain_centrality[node] = P_n.tolist()[i]

        return domain_centrality


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    path = './file/'
    domains_file = path + 'Company_domains.json'
    nb_file = path + 'Company_nb.json'

    summary = Summary()
    print("input the size of summary: [k >= 1], e.g., 10")
    k = int(input())

    results= summary.outputSummary(nb_file, domains_file, k)
    print("output the summary with", k, "representative entities:")
    for entity in results:
        print("entity:",entity.name,"|| notability:", entity.nb, "|| domains:", entity.domains)