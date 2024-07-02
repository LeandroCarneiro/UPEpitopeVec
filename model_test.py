from helpers.DatasetReader import GetAllAminoacids
from helpers.epitope_encoder import embedding_epitopes


aminoacids = GetAllAminoacids()
print(embedding_epitopes(
    ['RGPGRAFVTIGKIGNM', 'RGPGRAFVTIGKIGNMRGPGRAFVTIGKIGNM', 'RGPGRAFVTIGKIGNMIGKIGNM'], aminoacids, 30))