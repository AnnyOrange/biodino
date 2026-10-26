"""Task-specific frozen224 inference estimates, distinct from decoder training."""
def requirements(spec,dataset):
    if spec['min_free_mib']<16000:return {}
    # Observed frozen-readout peaks: GigaPath6396, Hoptimus5968,
    # GenBio6106, OpenPath5942, H+5862MiB. Keep >=25percent headroom.
    # Six-channel native H+ inference retains an extra conservative margin.
    peak=10000 if dataset=='rxrx1-cross'and spec['kind']=='biodino'else 8000
    return dict(min_free_mib=peak+2000,expected_peak_mib=peak)
