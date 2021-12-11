import pandas as pd

poi = pd.read_csv("~/data/os_data/poi_3522288/poi-extract-2019_12.csv",
                  sep='|', low_memory=False)
drop_cols_poi = ['pos_accuracy', 'uprn', 'topo_toid',
                 'topo_toid_version', 'usrn', 'usrn_mi',
                 'distance', 'telephone_number', 'url',
                 'supply_date', 'provenance', 'pointx_class',
                 'verified_address', 'qualifier_type', 'qualifier_data',
                 'brand']
poi = poi.drop(drop_cols_poi, axis=1)

places = pd.read_csv("~/data/os_data/open-names_3522290/combined.csv",
                     low_memory=False, index_col=0)
places['TYPE'].unique()
drop_cols_places = ['NAMES_URI', 'NAME1_LANG', 'NAME2_LANG',
                    'MOST_DETAIL_VIEW_RES', 'LEAST_DETAIL_VIEW_RES',
                    'POPULATED_PLACE', 'POPULATED_PLACE_URI']
places = places.drop(drop_cols_places, axis=1)

dbpedia = pd.read_csv("~/data/dbpedia/db_descriptions.csv", index_col=0)
poi['postcode'] = poi['postcode'].str[:2]
poi.rename(columns={'feature_easting': 'GEOMETRY_X',
                    'feature_northing': 'GEOMETRY_Y',
                    'abmin_boundary': 'DISTRICT_BOROUGH',
                    'geographic_county': 'COUNTY_UNITARY',
                    'postcode': 'POSTCODE_DISTRICT'
                    }, inplace=True)
poi['TYPE'] = 'poi'

dbpedia_poi = dbpedia.merge(poi, left_on='label', right_on='name')
dbpedia_poi.drop_duplicates(keep=False, inplace=True, subset='abs')

dbpedia_place = dbpedia.merge(places, left_on='label', right_on='NAME1')
dbpedia_place.drop_duplicates(keep=False, inplace=True, subset='abs')

# same as dbpedia column
db_check = places[
    ~places
    .iloc[:, -2]
    .str.contains('Unnamed: ', na=True)
    .dropna()
]
same_as_dbpedia = dbpedia.merge(db_check,
                                left_on='uri',
                                right_on='SAME_AS_DBPEDIA')
dbpedia = same_as_dbpedia
dbpedia = dbpedia.append(dbpedia_poi)
dbpedia = dbpedia.append(dbpedia_place)
dbpedia = dbpedia.drop_duplicates()

keep = ['label', 'TYPE',
        'POSTCODE_DISTRICT',
        'DISTRICT_BOROUGH',
        'COUNTY_UNITARY',
        'REGION', 'COUNTRY',
        'GEOMETRY_X', 'GEOMETRY_Y', 'abs']

dbpedia = dbpedia[keep]
dbpedia = dbpedia.drop_duplicates(subset=['abs'])
dbpedia.to_csv('~/data/dbpedia/db_combined.csv')
dbpedia.to_csv('~/apps/neo4j-community-4.0.3/import/wiki.csv')
