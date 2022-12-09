import time
from collections import defaultdict
from typing import List, Tuple
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import urllib
from pathlib import Path
from tqdm import tqdm

# sparql = SPARQLWrapper("http://localhost:3001/sparql")
sparql = SPARQLWrapper("http://127.0.0.1:3001/sparql")
sparql.setReturnFormat(JSON)

path = str(Path(__file__).parent.absolute())

# with open(path + '/../ontology/fb_roles', 'r') as f:
#     contents = f.readlines()
#
# roles = set()
# for line in contents:
#     fields = line.split()
#     roles.add(fields[1])

def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass

def wrap_ent(ent):
    if ent.startswith("m.") or ent.startswith("g."):
        ent_new = ':' + ent
    elif len(ent.split("-")) == 3 and all([is_number(e) for e in ent.split("-")]):  # 1921-09-13
        ent_new = '"' + ent + '"' + '^^<http://www.w3.org/2001/XMLSchema#date>'
    elif len(ent.split("-")) == 2 and all([is_number(e) for e in ent.split("-")]) and len(ent.split("-")[0]) == 4:  # 1921-09
        ent_new = '"' + ent + '"' + '^^<http://www.w3.org/2001/XMLSchema#gYearMonth>'
    elif ent.isdigit and all([is_number(e) for e in ent.split("-")]) and len(ent) == 4:  # 1990
        ent_new = '"' + ent + '"' + '^^<http://www.w3.org/2001/XMLSchema#gYear>'
    elif is_number(ent):  # 3.99
        ent_new = '"' + ent + '"'
    elif len(ent) > 50:
        ent_new = None
    else:  # Elizabeth II
        ent_new = ent.replace('"', "'")  # replace " to '
        ent_new = '"' + ent_new + '"' + '@en'

    return ent_new

def execute_query(query: str) -> List[str]:
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        assert len(result) == 1  # only select one variable
        for var in result:
            rtn.append(result[var]['value'].replace('http://rdf.freebase.com/ns/', '').replace("-08:00", ''))

    return rtn


def execute_unary(type: str) -> List[str]:
    query = ("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://rdf.freebase.com/ns/> 
    SELECT (?x0 AS ?value) WHERE {
    SELECT DISTINCT ?x0  WHERE {
    """
             '?x0 :type.object.type :' + type + '. '
                                                """
    }
    }
    """)
    # # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        rtn.append(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return rtn


def execute_binary(relation: str) -> List[Tuple[str, str]]:
    query = ("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://rdf.freebase.com/ns/> 
    SELECT DISTINCT ?x0 ?x1 WHERE {
    """
             '?x0 :' + relation + ' ?x1. '
                                  """
    }
    """)
    # # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        rtn.append((result['x0']['value'], result['x1']['value']))

    return rtn


def get_types(entity: str) -> List[str]:
    query = ("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://rdf.freebase.com/ns/> 
    SELECT (?x0 AS ?value) WHERE {
    SELECT DISTINCT ?x0  WHERE {
    """
             ':' + entity + ' :type.object.type ?x0 . '
                            """
    }
    }
    """)
    # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        rtn.append(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return rtn


def get_notable_type(entity: str):
    query = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        
        """
             ':' + entity + ' :common.topic.notable_types ?y . '
                            """
        ?y :type.object.name ?x0
        FILTER (lang(?x0) = 'en')
    }
    }
    """)

    # print(query)
    sparql.setQuery(query)
    results = sparql.query().convert()
    rtn = []
    for result in results['results']['bindings']:
        rtn.append(result['value']['value'])

    if len(rtn) == 0:
        rtn = ['entity']

    return rtn


def get_friendly_name(entity: str) -> str:
    query = ("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://rdf.freebase.com/ns/> 
    SELECT (?x0 AS ?value) WHERE {
    SELECT DISTINCT ?x0  WHERE {
    """
             ':' + entity + ' :type.object.name ?x0 . '
                            """
    }
    }
    """)
    # # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        if result['value']['xml:lang'] == 'en':
            rtn.append(result['value']['value'])

    if len(rtn) == 0:
        query = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
                 ':' + entity + ' :common.topic.alias ?x0 . '
                                """
        }
        }
        """)
        # # print(query)
        sparql.setQuery(query)
        try:
            results = sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        for result in results['results']['bindings']:
            if result['value']['xml:lang'] == 'en':
                rtn.append(result['value']['value'])

    if len(rtn) == 0:
        return 'null'

    return rtn[0]


def get_degree(entity: str):
    degree = 0

    query1 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT count(?x0) as ?value WHERE {
            """
              '?x1 ?x0 ' + ' :' + entity + '. '
                                           """
     FILTER regex(?x0, "http://rdf.freebase.com/ns/")
     }
     """)
    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        degree += int(result['value']['value'])

    query2 = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT count(?x0) as ?value WHERE {
        """
              ':' + entity + ' ?x0 ?x1 . '
                             """
    FILTER regex(?x0, "http://rdf.freebase.com/ns/")
    }
    """)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        degree += int(result['value']['value'])

    return degree


def get_in_attributes(value: str):
    in_attributes = set()

    query1 = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/> 
                SELECT (?x0 AS ?value) WHERE {
                SELECT DISTINCT ?x0  WHERE {
                """
              '?x1 ?x0 ' + value + '. '
                                   """
    FILTER regex(?x0, "http://rdf.freebase.com/ns/")
    }
    }
    """)
    # print(query1)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        in_attributes.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return in_attributes


def get_in_relations(entity: str):
    in_relations = set()

    query1 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT (?x0 AS ?value) WHERE {
            SELECT DISTINCT ?x0  WHERE {
            """
              '?x1 ?x0 ' + ' :' + entity + '. '
                                           """
     FILTER regex(?x0, "http://rdf.freebase.com/ns/")
     }
     }
     """)
    # print(query1)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        in_relations.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return in_relations


def get_in_entities(entity: str, relation: str):
    neighbors = set()

    query1 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT (?x1 AS ?value) WHERE {
            SELECT DISTINCT ?x1  WHERE {
            """
              '?x1' + ' :' + relation + ' :' + entity + '. '
                                                        """
                 FILTER regex(?x1, "http://rdf.freebase.com/ns/")
                 }
                 }
                 """)
    # print(query1)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        neighbors.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return neighbors


def get_in_entities_for_literal(value: str, relation: str):
    neighbors = set()

    query1 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT (?x1 AS ?value) WHERE {
            SELECT DISTINCT ?x1  WHERE {
            """
              '?x1' + ' :' + relation + ' ' + value + '. '
                                                      """
                FILTER regex(?x1, "http://rdf.freebase.com/ns/")
                }
                }
                """)
    # print(query1)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        neighbors.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return neighbors


def get_out_relations(entity: str):
    """
    Get the out relations of one source entity.
    :param entity: the head entity, which need to be wrapped before used in spqrql.
    :return: all relations stored in a set().
    """
    wrapped_ent = wrap_ent(entity)
    out_relations = set()
    if not wrapped_ent:
        print("%s can't as a src entity" % (entity))
        return out_relations

    query2 = ("""
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?r0 AS ?value) WHERE {
            SELECT DISTINCT ?r0  WHERE {
                """ '' + wrapped_ent + ' ?r0 ?t1 . ' """
                FILTER regex(?x0, "http://rdf.freebase.com/ns/")
            }
        }
    """)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        rel = result['value']['value'].replace('http://rdf.freebase.com/ns/', '')
        # some costumed filtered relations
        if "common.topic.description" not in rel:
            out_relations.add(rel)
    return out_relations


def get_out_entities(entity: str, relation: str):
    wrapped_ent = wrap_ent(entity)
    neighbors = set()
    if not wrapped_ent:
        print("%s can't as src entity" % (entity))
        return neighbors

    query2 = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x1 AS ?value) WHERE {
        SELECT DISTINCT ?x1  WHERE {
        """
              '' + wrapped_ent + ' :' + relation + ' ?x1 . '
                                               """
                     }
                     }
                     """)
    # print(query2)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        neighbors.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return neighbors


def get_entities_cmp(value, relation: str, cmp: str):
    neighbors = set()

    query2 = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x1 AS ?value) WHERE {
        SELECT DISTINCT ?x1  WHERE {
        """
              '?x1' + ' :' + relation + ' ?sk0 . '
                                        """
              FILTER regex(?x1, "http://rdf.freebase.com/ns/")
              """
                                        f'FILTER (?sk0 {cmp} {value})'
                                        """
                                       }
                                       }
                                       """)
    # print(query2)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        neighbors.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return neighbors


def get_adjacent_relations(entity: str):
    in_relations = set()
    out_relations = set()

    query1 = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
              '?x1 ?x0 ' + ' :' + entity + '. '
                                           """
     FILTER regex(?x0, "http://rdf.freebase.com/ns/")
     }
     }
     """)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        in_relations.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    query2 = ("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://rdf.freebase.com/ns/> 
    SELECT (?x0 AS ?value) WHERE {
    SELECT DISTINCT ?x0  WHERE {
    """
              ':' + entity + ' ?x0 ?x1 . '
                             """
    FILTER regex(?x0, "http://rdf.freebase.com/ns/")
    }
    }
    """)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        out_relations.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return in_relations, out_relations


def get_2hop_relations_from_2entities(entity0: str, entity1: str):  # m.027lnzs  m.0zd6  3200017000000
    query = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/>
            SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
            """
             '?x1 ?x0 ' + ' :' + entity0 + ' .\n' + '?x1 ?y ' + ' :' + entity1 + ' .'
                                                                                 """
                                                       FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                                                       FILTER regex(?y, "http://rdf.freebase.com/ns/")
                                                       }
                                                       """)
    # print(query)
    pass


def get_2hop_relations(entity: str):
    in_relations = set()
    out_relations = set()
    paths = []

    query1 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/>
            SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
            """
              '?x1 ?x0 ' + ' :' + entity + '. '
                                           """
                ?x2 ?y ?x1 .
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  """)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        r1 = result['r1']['value'].replace('http://rdf.freebase.com/ns/', '')
        r0 = result['r0']['value'].replace('http://rdf.freebase.com/ns/', '')
        in_relations.add(r0)
        in_relations.add(r1)

        if r0 in roles and r1 in roles:
            paths.append((r0, r1))

    query2 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
            """
              '?x1 ?x0 ' + ' :' + entity + '. '
                                           """
                ?x1 ?y ?x2 .
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  """)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        r1 = result['r1']['value'].replace('http://rdf.freebase.com/ns/', '')
        r0 = result['r0']['value'].replace('http://rdf.freebase.com/ns/', '')
        out_relations.add(r1)
        in_relations.add(r0)

        if r0 in roles and r1 in roles:
            paths.append((r0, r1 + '#R'))

    query3 = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>
                SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
                """
              ':' + entity + ' ?x0 ?x1 . '
                             """
                ?x2 ?y ?x1 .
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  """)

    sparql.setQuery(query3)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query3)
        exit(0)
    for result in results['results']['bindings']:
        r1 = result['r1']['value'].replace('http://rdf.freebase.com/ns/', '')
        r0 = result['r0']['value'].replace('http://rdf.freebase.com/ns/', '')
        in_relations.add(r1)
        out_relations.add(r0)

        if r0 in roles and r1 in roles:
            paths.append((r0 + '#R', r1))

    query4 = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>
                SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
                """
              ':' + entity + ' ?x0 ?x1 . '
                             """
                ?x1 ?y ?x2 .
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  """)

    sparql.setQuery(query4)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query4)
        exit(0)
    for result in results['results']['bindings']:
        r1 = result['r1']['value'].replace('http://rdf.freebase.com/ns/', '')
        r0 = result['r0']['value'].replace('http://rdf.freebase.com/ns/', '')
        out_relations.add(r1)
        out_relations.add(r0)

        if r0 in roles and r1 in roles:
            paths.append((r0 + '#R', r1 + '#R'))

    return in_relations, out_relations, paths


def get_label(entity: str) -> str:
    query = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?label) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
             ':' + entity + ' rdfs:label ?x0 . '
                            """
                            FILTER (langMatches( lang(?x0), "EN" ) )
                             }
                             }
                             """)
    # # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        label = result['label']['value']
        rtn.append(label)
    if len(rtn) != 0:
        return rtn[0]
    else:
        return None


def get_complete_ans(ans, type):
    if type == "uri_ans":
        ans_new = ':' + ans
    elif type == "date_ans":
        ans_new = '"' + ans + '"' + '^^<http://www.w3.org/2001/XMLSchema#date>'
    elif type == "gyear_ans":
        ans_new = '"' + ans + '"' + '^^<http://www.w3.org/2001/XMLSchema#gYear>'
    elif type == "gyear_month_ans":
        ans_new = '"' + ans + '"' + '^^<http://www.w3.org/2001/XMLSchema#gYearMonth>'
    elif type == "en_literal_ans":
        ans_new = '"' + ans + '"' + '@en'
    elif type == "digit_ans":
        ans_new = '"' + ans + '"'
    return ans_new


def search_paths(topic_ent, ans, hop, type):
    wrapper_ans = get_complete_ans(ans, type)
    if hop == 1:
        query = ("""
                    PREFIX : <http://rdf.freebase.com/ns/> 
                    SELECT distinct (?x0 as ?r0) WHERE {
                        """ ':' + topic_ent + ' ?x0 ' + wrapper_ans + ' .' """
                    }
                """)
    elif hop == 2:
        query = ("""
                    PREFIX : <http://rdf.freebase.com/ns/>
                    SELECT distinct ?x0 as ?r0 ?x1 as ?r1 WHERE {
                        """ ':' + topic_ent + ' ?x0 ' + ' ?t0 ' + '.\n' + '?t0 ?x1 ' + wrapper_ans + ' .' """
                    }
                """)
    elif hop == 3:
        query = ("""
                    PREFIX : <http://rdf.freebase.com/ns/>
                    SELECT distinct ?r0 ?r1 ?r2
                    WHERE {
                        # FILTER (?t0 != """ '' + wrapper_ans + '' """)
                        """ '?t1 ?r2 ' + wrapper_ans + ' .' """
                        {
                            SELECT distinct ?r0 ?r1 ?t1
                            WHERE{
                                ?t0 ?r1 ?t1 .
                                {
                                    SELECT distinct ?r0 ?t0
                                    WHERE{
                                        """ ':' + topic_ent + ' ?r0 ?t0 .' """
                                    }
                                }
                            }
                        }
                    }
                """)
    elif hop == 4:
        query = ("""
                    PREFIX : <http://rdf.freebase.com/ns/>
                    SELECT distinct ?r0 ?r1 ?r2 ?r3
                    WHERE {
                        # FILTER (?t1 != """ '' + wrapper_ans + '' """)
                        """ '?t2 ?r3 ' + wrapper_ans + ' .' """
                        {
                            SELECT distinct ?r0 ?r1 ?r2 ?t2
                            WHERE{
                                ?t1 ?r2 ?t2 .
                                {
                                    SELECT distinct ?r0 ?r1 ?t1
                                    WHERE{
                                        ?t0 ?r1 ?t1 .
                                        {
                                            SELECT distinct ?r0 ?t0
                                            WHERE{
                                                """ ':' + topic_ent + ' ?r0 ?t0 .' """
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                """)
    paths = []
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    for result in results['results']['bindings']:
        # path = [topic_ent]
        path = []
        for value in result.values():
            rel = value["value"].replace("http://rdf.freebase.com/ns/", "")
            path.append(rel)
        assert len(path) == hop, (result, path, hop)
        paths.append(path)
    return paths


def instaniate_and_execute(path):
    te = path[0]
    if len(path) == 2:
        query = ("""
                    PREFIX : <http://rdf.freebase.com/ns/> 
                    SELECT (?x1 AS ?value) WHERE {
                        SELECT DISTINCT ?x1  WHERE {
                            """ ':' + te + ' :' + path[1] + ' ?x1 . ' """
                        }
                    }
                """)
    elif len(path) == 3:
        query = ("""
                    PREFIX : <http://rdf.freebase.com/ns/> 
                    SELECT (?x1 AS ?value) WHERE {
                        SELECT DISTINCT ?x1 WHERE {
                            """ '?t1' + ' :' + path[2] + ' ?x1 .' """
                            {
                                SELECT DISTINCT ?t1 WHERE {
                                    """ ':' + te + ' :' + path[1] + ' ?t1 .' """
                                }
                            }
                        }
                    }
                """)
    elif len(path) == 4:
        query = ("""
                    PREFIX : <http://rdf.freebase.com/ns/> 
                    SELECT (?x3 AS ?value) WHERE {
                        SELECT DISTINCT ?x3  WHERE {
                            """ '?x2' + ' :' + path[3] + ' ?x3 .' """
                            {
                                SELECT DISTINCT ?x2 WHERE {
                                    """ '?x1' + ' :' + path[2] + ' ?x2 .' """
                                    {
                                        SELECT DISTINCT ?x1 WHERE {
                                        """ ':' + te + ' :' + path[1] + ' ?x1 .' """
                                        }
                                    }
                                }
                            }
                        }
                    }
                """)
    elif len(path) == 5:
        query = ("""
                    PREFIX : <http://rdf.freebase.com/ns/> 
                    SELECT (?x4 AS ?value) WHERE {
                        """ '?x3' + ' :' + path[4] + ' ?x4 .' """
                        {
                            SELECT DISTINCT ?x3  WHERE {
                                """ '?x2' + ' :' + path[3] + ' ?x3 .' """
                                {
                                    SELECT DISTINCT ?x2 WHERE {
                                        """ '?x1' + ' :' + path[2] + ' ?x2 .' """
                                        {
                                            SELECT DISTINCT ?x1 WHERE {
                                            """ ':' + te + ' :' + path[1] + ' ?x1 .' """
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                """)
    answers = set()
    # if "rdf-schema#domain" in query:
    #     return answers
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    for result in results['results']['bindings']:
        ans = result['value']['value'].replace("http://rdf.freebase.com/ns/", "")
        ans = ans.replace("^^<http://www.w3.org/2001/XMLSchema#date>", "")
        ans = ans.replace("^^<http://www.w3.org/2001/XMLSchema#gYear>", "")
        ans = ans.replace("<http://www.w3.org/2001/XMLSchema#gYearMonth>", "")
        ans = ans.replace("@en", "")
        ans = ans.replace("T05:12:00", "")
        ans = ans.strip('"')
        answers.add(ans)

    return list(answers)


def get_relations_within_2hop(topic_ent):
    wrapped_ent = wrap_ent(topic_ent)
    relations = set()
    if not wrapped_ent:
        print("%s can't as src entity" % (topic_ent))
        return relations

    query = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>
                SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
                    """ '' + wrapped_ent + ' ?x0 ' + ' ?t0 ' + '.\n' + '?t0 ?y ?t1' + ' .'"""
                    FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                    FILTER regex(?y, "http://rdf.freebase.com/ns/")
                }
            """)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    for result in results['results']['bindings']:
        path = []
        for value in result.values():
            rel = value["value"].replace("http://rdf.freebase.com/ns/", "")
            if "common.topic.description" not in rel:
                path.append(rel)
        relations.update(path)
    return list(relations)


def get_subgraph_within_2hop(src, src_type):
    # time0 = time.time()
    wrapper_srcs = get_complete_ans(src, src_type)
    query = ("""
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX : <http://rdf.freebase.com/ns/>
                    SELECT distinct ?x0 as ?r0 ?t0 as ?e0 ?x1 as ?r1 ?t1 as ?e1 WHERE {
                        """ '' + wrapper_srcs + ' ?x0 ' + ' ?t0 ' + '.\n' + '?t0 ?x1 ?t1' + ' .' """
                        FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                        FILTER regex(?x1, "http://rdf.freebase.com/ns/")
                    }
                """)
    triples = []
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    # print("One query consume %.2f s" % (time.time()-time0))
    # time1 = time.time()
    cvt_nodes = set()
    for result in results['results']['bindings']:
        values = []
        for v in result.values():
            values.append(v)

        if len(values) != 4:
            continue

        if "freebase.type_hints.mediator" in values[2]:
            # This is a cvt node, we should expand one more hop
            cvt_nodes.add(values[1])

        triple = [src]
        for value in values[:2]:
            ele = value["value"].replace("http://rdf.freebase.com/ns/", "")
            triple.append(ele)
        assert len(triple) == 3
        triples.append(triple)
        two_hop_triple = [triple[2]]
        for value in values[2:]:
            ele = value["value"].replace("http://rdf.freebase.com/ns/", "")
            two_hop_triple.append(ele)
        assert len(two_hop_triple) == 3
        triples.append(two_hop_triple)
    # print("Parse one query result consume %.2f s" % (time.time() - time1))
    return triples, cvt_nodes


def get_neibouring_relations(src, max_hop):
    if max_hop == 1:
        query = ("""
                    PREFIX : <http://rdf.freebase.com/ns/>
                    SELECT distinct ?r0 
                    WHERE {
                        FILTER (!isLiteral(?t0) OR lang(?t0) = '' OR langMatches(lang(?t0), 'en'))
                        FILTER regex(?r0, "http://rdf.freebase.com/ns/")
                        """ ':' + src + ' ?r0 ' + ' ?t0 ' + '.\n' """
                    }
                """)
    elif max_hop == 2:
        query = ("""
                    PREFIX : <http://rdf.freebase.com/ns/>
                    SELECT distinct ?r0 ?r1
                    WHERE {
                        """ 'FILTER (?t1 != ' + ':' + src + ')\n' """
                        FILTER (!isLiteral(?t1) OR lang(?t1) = '' OR langMatches(lang(?t1), 'en'))
                        FILTER regex(?r1, "http://rdf.freebase.com/ns/")
                        """ ':' + src + '?r0 ?t0 .\n' +
                            '?t0 ?r1 ?t1 .' """
                    }
                """)
    elif max_hop == 3:
        query = ("""
                    PREFIX : <http://rdf.freebase.com/ns/>
                    SELECT distinct ?r0 ?r1 ?r2
                    WHERE{
                        FILTER (?t2 != ?t0)
                        FILTER (!isLiteral(?t2) OR lang(?t2) = '' OR langMatches(lang(?t2), 'en'))
                        FILTER regex(?r2, "http://rdf.freebase.com/ns/")
                        ?t1 ?r2 ?t2 .
                        {
                            SELECT distinct ?t1
                            WHERE{
                                FILTER (?t1 != """ ' :' + src + """)
                                FILTER (!isLiteral(?t1) OR lang(?t1) = '' OR langMatches(lang(?t1), 'en'))
                                """ '?t0 ?r1 ?t1 .' """
                                {
                                    SELECT distinct ?t0
                                    WHERE{
                                        FILTER (!isLiteral(?t0) OR lang(?t0) = '' OR langMatches(lang(?t0), 'en'))
                                        """ ':' + src + ' ?r0 ?t0 .' """
                                    }
                                }
                            }
                        }
                    }
                """)
    paths = []
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    for result in results['results']['bindings']:
        # path = [topic_ent]
        path = []
        for value in result.values():
            rel = value["value"].replace("http://rdf.freebase.com/ns/", "")
            path.append(rel)
        paths.append(path)
    return paths


def creat_query_and_execute(te, path):
    if len(path) == 3:
        query = ("""
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX : <http://rdf.freebase.com/ns/> 
                    SELECT DISTINCT ?t1
                    WHERE {
                        FILTER (!isLiteral(?t1) OR lang(?t1) = '' OR langMatches(lang(?t1), 'en'))
                        """ ':' + te + ' :' + path[1] + ' ?t1 . ' """
                    }
                """)
    elif len(path) == 5:
        query = ("""
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX : <http://rdf.freebase.com/ns/>
                    SELECT DISTINCT ?t1 ?t2  WHERE {
                        FILTER (!isLiteral(?t2) OR lang(?t2) = '' OR langMatches(lang(?t2), 'en'))
                        """ ':' + te + ' :' + path[1] + ' ?t1 .\n' +
                            '?t1' + ' :' + path[3] + ' ?t2 . ' """
                    }
                """)
    elif len(path) == 7:
        query = ("""
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX : <http://rdf.freebase.com/ns/> 
                    SELECT DISTINCT ?t1 ?t2 ?t3 WHERE {
                        FILTER (!isLiteral(?t3) OR lang(?t3) = '' OR langMatches(lang(?t3), 'en'))
                        """ ':' + te + ' :' + path[1] + ' ?t1 .\n' +
                            '?t1' + ' :' + path[3] + ' ?t2 .\n' +
                            '?t2' + ' :' + path[5] + ' ?t3 . ' """
                    }
                """)
    elif len(path) == 9:
        query = ("""
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX : <http://rdf.freebase.com/ns/> 
                    SELECT DISTINCT ?t1 ?t2 ?t3 ?t4 WHERE {
                        FILTER (!isLiteral(?t4) OR lang(?t4) = '' OR langMatches(lang(?t4), 'en'))
                        """ ':' + te + ' :' + path[1] + ' ?t1 .\n' +
                             '?t1' + ' :' + path[3] + ' ?t2 .\n' +
                             '?t2' + ' :' + path[5] + ' ?t3 .\n' +
                             '?t3' + ' :' + path[7] + ' ?t4 . ' """
                    }
                """)
    # real_ent = [set()]*int((len(path)-1)/2)
    real_ent = defaultdict(set)
    # triples = [set()]*int((len(path)-1)/2)
    triples = defaultdict(set)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    for result in results['results']['bindings']:
        values = []
        for value in result.values():
            res = value["value"].replace("http://rdf.freebase.com/ns/", "")
            values.append(res)
        last_v = te
        for idx, v in enumerate(values):
            real_ent[idx].add(v)
            triples[idx].add((last_v, path[idx*2+1], v))
            last_v = v

    return real_ent, triples