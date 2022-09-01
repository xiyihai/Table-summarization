import  math,json
import numpy as np
import warnings,random
from munkres import Munkres

"""
Summary of this module here.

The module is used to calculate the notability score of each entity.
"""

class Entity:
    def __init__(self, name, domains, nb):
        self.name = name
        self.domains = domains
        self.nb = nb


class Notability:
    def __init__(self):
        self.sim_metrix = {}

    """
        Summary of this class here.

        Notability class takes the entity information (including linkages, domains, page views from functions.py) as input
        and outputs the notability score for each entity.

        usage example:
        nb = Notability()
        nb.NB_eval()
    """

    def NB_eval(self, links_file, domains_file, pv_file, domain_flag = 0,  pv_flag = 0, ft = 0.15):
        """
        The main function of Notability.
        :param links_file: the file containing entity linkage information (output of functions.EntityLinks)
        :param domains_file: the file containing entity domains information (retrieve by SPARQL)
        :param pageview_file: the file containing entity page view information (output of functions.EntityViews)
        :param domain_flag: whether to consider the domain similarity in notability evaluation formulate (7) (refer to paper)
        :param pv_flag: whether to consider the popularity in notability evaluation formulate (8) (refer to paper)
        :param ft: ft is damping parameter
        :return: the notability score for each entity.
        """

        with open(domains_file, 'r', encoding='utf-8') as f:
            domains = eval(f.read())

        entity_domains = domains['entity_domains']
        self.sim_metrix = domains['metrix']


        with open(links_file, 'r', encoding='utf-8') as f:
            links = eval(f.read())

        edges = []
        for entity in links:
            for inlink in entity['inlinks']:
                edge = [inlink, entity['name']]
                edges.append(edge)

        nodes = []
        for edge in edges:
            if edge[0] not in nodes:
                nodes.append(edge[0])
            if edge[1] not in nodes:
                nodes.append(edge[1])


        N = len(nodes)
        # map name to number
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


        # build the matrix S
        # The value in row i and column j represents the probability that a user goes from page j to page i
        S = np.zeros([N, N])
        for edge in edges:
            S[edge[1], edge[0]] = 1


        # whether to consider domain similarity
        if domain_flag == 0:
            for j in range(N):
                sum_of_col = sum(S[:, j])
                for i in range(N):
                    if sum_of_col == 0:
                        S[i, j] = 0
                    else:
                        S[i, j] /= sum_of_col
        else:
            for j in range(N):
                sim_sum = {}
                for i in range(N):
                    if S[i, j] != 0:
                        ei_dom = entity_domains[num_to_node[i]]
                        ej_dom = entity_domains[num_to_node[j]]

                        sim_ij = self.dom_distance_km(Entity(i, ei_dom, 0), Entity(j, ej_dom, 0))
                        sim_sum[i] = sim_ij
                    else:
                        sim_sum[i] = 0

                sum_of_col = sum(sim_sum.values())
                for k in range(N):
                    if sum_of_col == 0:
                        S[k, j] = 0
                    else:
                        S[k, j] = sim_sum[k] / sum_of_col


        # whether to consider popularity
        view_vector = np.ones(N) / N
        if pv_flag != 0:
            with open(pv_file, 'r', encoding='utf-8') as f2:
                entity_view = eval(f2.read())

            node_all_views = {}
            node_size = {}

            # consider the entity partition results based on the formulate (8) in the paper
            partitions = self.partition(entity_domains)

            for cluster in partitions:
                sum_views = 0
                for node in cluster:
                    if node in entity_view:
                        sum_views += entity_view[node]

                for node in cluster:
                    node_all_views[node] = sum_views
                    node_size[node] = len(cluster)

            for node in nodes:
                if node not in node_all_views:
                    node_all_views[node] = 0
                    node_size[node] = 1

            view_entity_list = []
            for node in nodes:
                if node in entity_view:
                    sum_views = node_all_views[node]
                    size = node_size[node]
                    if sum_views != 0:
                        value =  (entity_view[node] / sum_views) * (size/len(nodes))
                    else:
                        value = 0
                    view_entity_list.append(value)
                else:
                    view_entity_list.append(0)

            view_vector = np.array(view_entity_list)


        if (1 - ft) == 1:
            A = 0.85 * S + 0.15 * np.ones([N, N]) / N
        else:
            A = (1 - ft) * S


        P_n = np.ones(N) / N
        P_n1 = np.zeros(N)

        e = 100000
        print('loop...')

        while e > 0.00000001:
            P_n1 = np.dot(A, P_n) + ft * view_vector
            e = P_n1 - P_n
            e = max(map(abs, e))
            P_n = P_n1

        entity_nb = {}
        for i, node in enumerate(nodes):
            entity_nb[node] = P_n.tolist()[i]


        # Normalize the value
        items = sorted(entity_nb.items(), key=lambda k: k[1], reverse=True)
        max_nb = items[0][1]
        k = 0
        for k in range(10):
            if max_nb * (math.pow(10, k)) > 1: break

        entity_tmp = {}
        for (entity, nb) in entity_nb.items():
            nb = round(nb * (math.pow(10, k)), 7)
            entity_tmp[entity] = nb


        items = sorted(entity_tmp.items(), key=lambda k: k[1], reverse=True)
        max_nb = items[5][1]
        min_nb = items[len(entity_tmp.keys())-1][1]
        entity_nb_norm = {}
        for (entity, nb) in entity_tmp.items():
            nb_norm = min(1.0, round((nb - min_nb) / (max_nb - min_nb), 7))
            entity_nb_norm[entity] = nb_norm


        entity_nb_score = {}
        result = sorted(entity_nb_norm.items(), key=lambda k: k[1], reverse=True)
        for r in result:
            entity_nb_score[r[0]] = r[1]

        return entity_nb_score


    def dom_distance_km(self, e1, e2,):
        dom1 = e1.domains
        dom2 = e2.domains

        len1 = len(dom1)
        len2 = len(dom2)

        if len1 == 0 or len2 == 0:
            return 0
        else:
            N = max(len(dom1), len(dom2))
            matrix = np.zeros((N,N))
            for i,d1 in enumerate(dom1):
                for j,d2 in enumerate(dom2):
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
            return 1 - total/N



    def partition(self, entity_domains, k=10):

        entities = []
        for e,d in entity_domains.items():
            if len(d) != 0:
                entities.append(Entity(e,d,0))

        centroids = set()
        while len(centroids) < k:
            ran = random.randint(0, len(entities) - 1)
            centroids.add(entities[ran])

        flag = 1
        k = 0
        clusters = {}
        while flag:
            clusters = self.classify(entities, centroids, self.dom_distance_km)
            centroids, flag = self.get_new_center(clusters, self.dom_distance_km)
            k+=1

        cluster_names = []
        for entities in clusters.values():
            names = []
            for e in entities:
                names.append(e.name)
            cluster_names.append(names)
        return cluster_names


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
                min_dis_cluster += max(0,distance_func(center, e))

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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    path = './file/'
    links_file = path + 'Company_links.json'
    domains_file = path + 'Company_domains.json'
    pageview_file = path + 'Company_pv.json'

    nb = Notability()
    results= nb.NB_eval(links_file, domains_file, pageview_file, domain_flag = 1,  pv_flag = 1, ft = 0.15)
    print("output the notability score for each entity: [0 <= nb <= 1] ")
    for e,nb in results.items():
        print(e,nb)
