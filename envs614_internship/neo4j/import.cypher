MATCH (n) DETACH DELETE n;

CREATE CONSTRAINT ON (pc:Postcode) ASSERT pc.name IS UNIQUE;
CREATE CONSTRAINT ON (b:Borough) ASSERT b.name IS UNIQUE;
CREATE CONSTRAINT ON (c:County) ASSERT c.name IS UNIQUE;
CREATE CONSTRAINT ON (cy:Country) ASSERT cy.name IS UNIQUE;

USING PERIODIC COMMIT 10000
LOAD CSV WITH HEADERS FROM "file:///wiki.csv" AS row
WITH row WHERE NOT row.label IS null
MERGE (l:Location {name: row.label, type: row.TYPE, X: row.GEOMETRY_X, Y: row.GEOMETRY_Y, abstract: row.abs})
RETURN count(l);

USING PERIODIC COMMIT 10000
LOAD CSV WITH HEADERS FROM "file:///wiki.csv" AS row
WITH row WHERE NOT row.POSTCODE_DISTRICT IS null
MERGE (pc:Postcode {name: row.POSTCODE_DISTRICT})
RETURN count(pc);

USING PERIODIC COMMIT 10000
LOAD CSV WITH HEADERS FROM "file:///wiki.csv" AS row
WITH row WHERE NOT row.DISTRICT_BOROUGH IS null
MERGE (b:Borough {name: row.DISTRICT_BOROUGH})
RETURN count(b);

USING PERIODIC COMMIT 10000
LOAD CSV WITH HEADERS FROM "file:///wiki.csv" AS row
WITH row WHERE NOT row.COUNTY_UNITARY IS null
MERGE (c:County {name: row.COUNTY_UNITARY})
RETURN count(c);

USING PERIODIC COMMIT 10000
LOAD CSV WITH HEADERS FROM "file:///wiki.csv" AS row
WITH row WHERE NOT row.REGION IS null
MERGE (r:Region {name: row.REGION})
RETURN count(r);

USING PERIODIC COMMIT 10000
LOAD CSV WITH HEADERS FROM "file:///wiki.csv" AS row
WITH row WHERE NOT row.COUNTRY IS null
MERGE (cy:Country {name: row.COUNTRY})
RETURN count(cy);

USING PERIODIC COMMIT 10000
LOAD CSV WITH HEADERS FROM "file:///wiki.csv" AS row
WITH row.label as lname, row.POSTCODE_DISTRICT AS pcname
MATCH (l:Location {name: lname})
MATCH (pc:Postcode {name: pcname})
MERGE (pc)-[rel:CONTAINS]->(l)
RETURN count(rel);

USING PERIODIC COMMIT 10000
LOAD CSV WITH HEADERS FROM "file:///wiki.csv" AS row
WITH row.label as lname, row.DISTRICT_BOROUGH AS bname
MATCH (l:Location {name: lname})
MATCH (b:Borough {name: bname})
MERGE (b)-[rel:CONTAINS]->(l)
RETURN count(rel);

USING PERIODIC COMMIT 10000
LOAD CSV WITH HEADERS FROM "file:///wiki.csv" AS row
WITH row.label as lname, row.COUNTY_UNITARY AS cname
MATCH (l:Location {name: lname})
MATCH (c:County {name: cname})
MERGE (c)-[rel:CONTAINS]->(l)
RETURN count(rel);

USING PERIODIC COMMIT 10000
LOAD CSV WITH HEADERS FROM "file:///wiki.csv" AS row
WITH row.label as lname, row.REGION AS rname
MATCH (l:Location {name: lname})
MATCH (r:Region {name: rname})
MERGE (r)-[rel:CONTAINS]->(l)
RETURN count(rel);

USING PERIODIC COMMIT 10000
LOAD CSV WITH HEADERS FROM "file:///wiki.csv" AS row
WITH row.label as lname, row.COUNTRY AS cyname
MATCH (l:Location {name: lname})
MATCH (cy:Country {name: cyname})
MERGE (cy)-[rel:CONTAINS]->(l)
RETURN count(rel);
