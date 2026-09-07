from scripts.build_bio_source_domain_inventory import source_domain


def test_source_domain_extracts_idr_accession_component():
    domain, family = source_domain(
        "/root/idr0099-jain-beetlelightsheet/20201001-ftp/sample.tif"
    )
    assert domain == "idr:idr0099-jain-beetlelightsheet"
    assert family == "idr"


def test_source_domain_extracts_external_collection_component():
    domain, family = source_domain(
        "/root/000-LM-dataset-preprocessed/0-large-model-dataset/57-CPJUMP1/source/image.tif"
    )
    assert domain == "external_collection:57-cpjump1"
    assert family == "external_collection"
