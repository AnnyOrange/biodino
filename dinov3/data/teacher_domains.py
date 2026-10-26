"""Teacher applicability from training-source metadata, never benchmark labels."""
import json


def teacher_domain(sample):
    metadata = sample.get('meta.json')
    if metadata is None:
        raise ValueError('Domain-routed distillation requires sample meta.json')
    if isinstance(metadata, (str, bytes, bytearray)):
        metadata = json.loads(metadata)
    # r9 mixes microscopy and H&E: its shard name is not a domain label.
    modality = metadata.get('imaging_modality', '')
    if modality == 'brightfield_pathology':
        if metadata.get('tar_member_type') != 'pathology_rgb_patch':
            raise ValueError('Pathology metadata lacks the RGB pathology contract')
        return 'pathology'
    return 'other'
