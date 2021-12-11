from SPARQLWrapper import SPARQLWrapper, CSV
from io import BytesIO
import pandas as pd

csv = pd.DataFrame()
for i in range(0, 100000, 10000):
    query = """
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX res: <http://dbpedia.org/resource/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
    PREFIX dbp: <http://dbpedia.org/property/>

    SELECT DISTINCT ?uri ?label ?abs
    WHERE {
            { ?uri dbo:country res:England } UNION
            { ?uri dbo:country res:United_Kingdom } UNION
            { ?uri dbo:country res:Scotland } UNION
            { ?uri dbo:country res:Wales } UNION
            { ?uri dbo:location res:England } UNION
            { ?uri dbo:location res:United_Kingdom } UNION
            { ?uri dbo:location res:Scotland } UNION
            { ?uri dbo:location res:Wales } .

            { ?uri rdf:type dbo:Place } UNION
            { ?uri rdf:type dbo:Organisation } .
              ?uri rdfs:label ?label . FILTER (lang(?label) = 'en')
              ?uri dbo:abstract ?abs . FILTER (lang(?abs) = 'en')
    }
    LIMIT 10000 OFFSET
    """ + str(i)
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(CSV)
    results = sparql.query().convert()
    df = pd.read_csv(BytesIO(results), dtype=str)
    csv = csv.append(df)

csv = csv.drop_duplicates()

csv.to_csv("~/data/dbpedia/db_descriptions.csv")
