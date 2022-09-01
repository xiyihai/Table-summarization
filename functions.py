import string
import urllib.parse, urllib.request, urllib
from urllib.parse import quote
import json
import warnings
import copy,csv

"""
Summary of this module here.

The functions module is used to provide the functions of obtaining the information
needed in NotabilityEva.py including mapping entities to Wikipages (class: EntityMap), 
obtaining entity linkages between Wikipages (class EntityLinks), 
and retrieving the page views of entities (class: EntityViews).
"""


class EntityMap:
    """
        Summary of this class here.

        EntityMap class maps the entities in the table to wikipages
        and returns a dict with key = (name in wikipage), value = (name in table).

        usage example:
        em = EntityMap()
        em.mapped_entities()
    """


    def mapped_entities(self, table_path, k):
        """
        the main function in EntityMap
        :param table_path: the path of the source table
        :param k: read the entities in the k-th column (the primary entity column) in the table
        :return: a dict with key = (name in wikipage), value = (name in table).
        """

        with open(table_path, 'r', encoding='utf-8-sig') as f:
            csv_reader = csv.reader(f)
            entities = []
            for i, tuple in enumerate(csv_reader):
                try:
                    entities.append(tuple[k])
                except:
                    print(i, tuple)

            entity2mention = {}
            mention2entity = self.NED(entities)
            for (e, mapped_e) in mention2entity.items():
                entity2mention[mapped_e] = e
            return entity2mention


    def NED(self, table_entities):
        text = ''
        for e in table_entities:
            text = text + e + ','
        mention2entity = self.CallWikifier(text)
        mention2entity_filter = copy.deepcopy(mention2entity)
        for mention in mention2entity:
            if mention not in table_entities:
                del mention2entity_filter[mention]
        return mention2entity_filter


    def CallWikifier(self, text, lang="en", threshold=0.8):
        # Prepare the URL.
        data = urllib.parse.urlencode([
            ("text", text), ("lang", lang),
            ("userKey", "xntgwvolgpzwvipqgbjhsdsuamzhri"),
            ("pageRankSqThreshold", "%g" % threshold), ("applyPageRankSqThreshold", "false"),
            ("nTopDfValuesToIgnore", "0"), ("nWordsToIgnoreFromList", "-1"),
            ("wikiDataClasses", "false"), ("wikiDataClassIds", "false"),
            ("support", "true"), ("ranges", "false"),
            ("includeCosines", "false"), ("maxMentionEntropy", "-1"),
            ("maxTargetsPerMention", "20")
        ])
        url = "http://www.wikifier.org/annotate-article"
        # Call the Wikifier and read the response.
        req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
        with urllib.request.urlopen(req, timeout=60) as f:
            response = f.read()
            response = json.loads(response.decode("utf8"))
        # Output the annotations.
        if 'words' not in response:
            print(response)
        else:
            words = response['words']
            spaces = response['spaces']
            mention2entity = {}
            for annotation in response["annotations"]:
                max = 0
                mention_final = {}
                for mention in annotation['support']:
                    if mention['pMentionGivenSurface'] > max:
                        mention_final = mention
                        max = mention['pMentionGivenSurface']
                begin = mention_final['wFrom']
                end = mention_final['wTo']
                mention_str = ""
                for i in range(begin, end + 1):
                    if i == begin:
                        mention_str = words[i]
                    else:
                        mention_str = mention_str + spaces[i] + words[i]
                mention2entity[mention_str] = annotation['title']
            return mention2entity


class EntityLinks:
    """
    Summary of this class here.

    EntityLinks class obtains the Wikipage links for each pair of entities
    and writes these linkages into the local file.

    usage example:
    el = EntityLinks()
    el.outputLinks()
    """

    def outputLinks(self, entities, links_file, entity2mention):
        """
        the main function in EntityLinks
        :param entities: a list of the mapped entities in the table
        :param links_file:  the path of local file
        :param entity2mention: the dict for mapping entity (the name in the wikipage) to mention (the name in the table)
        :return: None
        """

        links = self.getLinks(entities)
        for entity in links:
            entity['name'] = entity2mention[entity['name']]

        with open(links_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(links))


    def getLinks(self, entities):
        entities_link = []
        for index, e in enumerate(entities):
            realEntities = []
            all_entities = []
            page_entity = self.getLinks_child(e)
            if type(page_entity) != int:
                if page_entity['name'] in page_entity['outlinks']:
                    page_entity['outlinks'].remove(page_entity['name'])
                for entity in page_entity['outlinks']:
                    all_entities.append(entity)
                    if entity in entities:
                        realEntities.append(entity)
            print(e)
            entities_link.append({'name': e, 'outlink_number': len(realEntities), 'outlinks': realEntities})

        entityInLinks = []
        for target in entities_link:
            entity = {}
            entity['name'] = target['name']
            inlink_number = 0
            inlinks = []
            for source in entities_link:
                if target['name'] in source['outlinks']:
                    inlink_number = inlink_number + 1
                    inlinks.append(source['name'])
            entity['inlink_number'] = inlink_number
            entity['inlinks'] = inlinks
            entityInLinks.append(entity)

        return entityInLinks


    def getLinks_child(self, entity):
        try:
            page_entity = self.NeighGraph('en', entity, 0, 1)
            return page_entity
        except:
            print("cannot find child of ", entity)
            return 0


    def NeighGraph(self, lang, title, nPredLevels, nSuccLevels):
        # Prepare the URL.
        data = urllib.parse.urlencode([("lang", lang), ("title", title),
                                       ("nPredLevels", nPredLevels), ("nSuccLevels", nSuccLevels)])
        url = "http://www.wikifier.org/get-neigh-graph?" + data
        # Call the Wikifier and read the response.
        with urllib.request.urlopen(url, timeout=60) as f:
            response = f.read()
            response = json.loads(response.decode("utf8"))

        if 'error' in response:
            print(response['error'])
            return 0
        else:
            nVertices = response["nVertices"]
            titles = response["titles"]
            outlinks = []
            for v in response["successors"][0]:
                outlinks.append(titles[v])
            return {'name': titles[0], 'outlinks': outlinks}


class EntityViews:
    """
       Summary of this class here.

       EntityViews class retrieves the page view for each entity within the given time span.
       and writes the information into the local file.

       usage example:
       ev = EntityViews()
       ev.outputViews()
       """

    def outputViews(self, entities, time, pv_file, entity2mention):
        """
        the main function in EntityViews
        :param entities: a list of the mapped entities in the table
        :param time:  the time span, e.g., time = [ '20200101', '20200630']
        :param pv_file:  the path of local file
        :param entity2mention: the dict for mapping entity (the name in the wikipage) to mention (the name in the table)
        :return: None
        """

        view_entity_dict = {}
        for entity in entities:
            view_entity_dict[entity2mention[entity]] = self.getView_child(entity.replace(' ', '_'), time)

        with open(pv_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(view_entity_dict))


    def getView_child(self, entity, timespan):
        try:
            views = self.getviews(entity, timespan)
            print(entity, views)
            return views
        except:
            print("cannot obtain the page view of ", entity, timespan)
            return 0


    def getviews(self, entity, timespan, time_unit='monthly'):
        sum_views = 0
        url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/" \
              + entity + "/" + time_unit + "/" + timespan[0] + "/" + timespan[1]

        url = quote(url, safe=string.printable)
        with urllib.request.urlopen(url, timeout=60) as f:
            response = f.read()
            response = json.loads(response.decode("utf8"))

        if 'detail' in response:
            print(response['detail'])
            return 0
        else:
            for item in response['items']:
                sum_views += item['views']
            return sum_views



if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    path = './file/'
    table_path = path + 'Company.csv'

    entitymap = EntityMap()
    entity2mention = entitymap.mapped_entities(table_path, k=1)
    entities = entity2mention.keys()


    links_file = path + 'Company_links.json'
    links = EntityLinks()
    links.outputLinks(entities, links_file, entity2mention)


    year = '2020'
    time = [ year+'0101', year+'0630']
    pv_file = path + 'Company_pv.json'
    views = EntityViews()
    views.outputViews(entities, time, pv_file, entity2mention)

