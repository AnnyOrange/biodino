from scripts.build_expert_feature_bank import role_reliability


def _metadata(*, organism="", acquisition_family="unresolved", sample_type=""):
    return {
        "organism": organism,
        "acquisition_family": acquisition_family,
        "sample_type": sample_type,
    }


def test_tissue_routing_prefers_human_tissue_but_keeps_cross_species_vote():
    human = role_reliability("tissue", _metadata(organism="human", sample_type="tissue"))
    mouse = role_reliability("tissue", _metadata(organism="mouse", sample_type="tissue"))
    off_domain = role_reliability("tissue", _metadata(sample_type="cell"))
    assert human == 1.0
    assert mouse == 0.75
    assert 0 < off_domain < mouse


def test_organism_cell_routing_uses_species_and_acquisition_metadata():
    fluorescence = role_reliability(
        "organism_cell",
        _metadata(organism="human", acquisition_family="fluorescence_microscopy"),
    )
    known_species = role_reliability("organism_cell", _metadata(organism="beetle"))
    unresolved = role_reliability("organism_cell", _metadata())
    assert fluorescence == 1.0
    assert known_species == 0.85
    assert unresolved == 0.35
