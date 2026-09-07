from scripts.enrich_idr_source_domains import (
    acquisition_family,
    parse_study_fields,
    parse_submodule_repositories,
)


def test_parses_idr_submodule_repository_by_accession():
    text = """
    [submodule "idr0099-jain-beetlelightsheet"]
        path = idr0099-jain-beetlelightsheet
        url = https://github.com/IDR/idr0099-jain-beetlelightsheet
    """
    assert parse_submodule_repositories(text) == {
        "idr0099": "idr0099-jain-beetlelightsheet"
    }


def test_parses_repeated_study_fields_and_acquisition_family():
    fields = parse_study_fields(
        "Study Organism\tTribolium castaneum\n"
        "Experiment Imaging Method\tlight sheet fluorescence microscopy\tSPIM\n"
    )
    assert fields["Study Organism"] == ["Tribolium castaneum"]
    assert fields["Experiment Imaging Method"] == [
        "light sheet fluorescence microscopy",
        "SPIM",
    ]
    assert (
        acquisition_family(" | ".join(fields["Experiment Imaging Method"]), "", "")
        == "light_sheet_fluorescence"
    )
