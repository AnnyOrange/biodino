"""Physical 3D correspondence scoring and calibrated CLEM data admission.

No z-projection, slice-retrieval surrogate, or ground-truth registration fitting
is used. FM->EM descriptor matches fit a single volume-level 3D affine map.
"""
import csv
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter

from .core import apply_affine, register_descriptors, registration_metrics, voxel_coordinates


@dataclass(frozen=True)
class VolumeGeometry:
    shape_zyx: tuple
    spacing_xyz_um: tuple
    origin_xyz_um: tuple = (0., 0., 0.)

    @property
    def diagonal_um(self):
        return float(np.linalg.norm(np.asarray(self.shape_zyx)[::-1] * self.spacing_xyz_um))

    def coordinates(self, indices_zyx):
        return voxel_coordinates(indices_zyx, self.spacing_xyz_um, self.origin_xyz_um)


def tiff_geometry(path):
    """Read TIFF ImageJ calibration, explicitly reject uncalibrated EM data."""
    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        metadata = handle.imagej_metadata or {}
        unit = metadata.get("unit", "")
        if unit not in {"micron", "um", "micrometer", "µm", r"\u00B5m", r"\\u00B5m"} or series.axes != "ZYX":
            raise ValueError(f"Explicit XYZ micrometer calibration required: {path}")
        page = handle.pages[0]
        def resolution(axis):
            numerator, denominator = page.tags[axis + "Resolution"].value
            return denominator / numerator
        spacing = (resolution("X"), resolution("Y"), float(metadata["spacing"]))
        if min(spacing) <= 0:
            raise ValueError("Invalid TIFF voxel calibration")
        return VolumeGeometry(tuple(series.shape), spacing)


def czi_metadata(path):
    """Use czifile and XML, never infer voxel spacing or mitochondrial channel."""
    from czifile import CziFile
    with CziFile(path) as handle:
        xml = ET.fromstring(handle.metadata())
        axes, shape = handle.axes, handle.shape
        spacing = {}
        for element in xml.findall(".//Scaling/Items/Distance"):
            spacing[element.attrib["Id"]] = float(element.findtext("Value")) * 1e6
        if not {"X", "Y", "Z"}.issubset(spacing) or min(spacing.values()) <= 0:
            raise ValueError("CZI lacks positive XYZ spacing in meters")
        geometry = VolumeGeometry(tuple(shape[axes.index(axis)] for axis in "ZYX"),
                                  tuple(spacing[axis] for axis in "XYZ"))
        channels = []
        for channel in xml.findall(".//Information/Image/Dimensions/Channels/Channel"):
            channels.append({"id": channel.attrib.get("Id"), "name": channel.attrib.get("Name"),
                             "fluorophore": channel.findtext("Fluor"), "dye": channel.findtext("DyeName")})
        return {"axes": axes, "shape": list(shape), "geometry": geometry, "channels": channels}


def bigwarp_landmarks(path):
    """BigWarp physical XYZ coordinates (moving FM, target EM), active rows only.

    BioStudies identifies these files as BigWarp landmarks. BigWarp exports
    world coordinates; for these entries calibrated image units are micrometers.
    Never multiply these world coordinates by voxel spacings again.
    """
    with Path(path).open(newline="") as handle:
        rows = list(csv.reader(handle))
    active = [row for row in rows if len(row) == 8 and row[1].lower() == "true"]
    ids = [row[0] for row in active]
    if not active or len(ids) != len(set(ids)):
        raise ValueError("Empty/duplicate BigWarp landmark IDs")
    source = np.asarray([[float(v) for v in row[2:5]] for row in active])
    target = np.asarray([[float(v) for v in row[5:8]] for row in active])
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite BigWarp coordinates")
    return ids, source, target


def evaluate_volume_correspondence(source_features, target_features, source_indices_zyx,
    target_indices_zyx, source_geometry, target_geometry, source_landmarks_xyz_um,
    target_landmarks_xyz_um, ratio=.9, threshold_um=1., seed=0,
    matching_backend="scipy", device="cpu"):
    """Full 3D affine from calibrated appearance correspondences, scoring heldout GT."""
    result = register_descriptors(source_features, target_features,
        source_geometry.coordinates(source_indices_zyx), target_geometry.coordinates(target_indices_zyx),
        ratio=ratio, threshold=threshold_um, seed=seed, matching_backend=matching_backend, device=device)
    warped = apply_affine(source_landmarks_xyz_um, result.matrix)
    scores = registration_metrics(source_landmarks_xyz_um, target_landmarks_xyz_um, warped,
                                  target_geometry.diagonal_um, coordinate_unit="micrometers")
    return {"matrix": result.matrix.tolist(), "success": result.success, "reason": result.reason,
            "matches": result.matches, "inliers": result.inliers, "metrics": scores}


def extract_volume_descriptors(backbone, geometry, read_slice, layers, device,
                               max_side=256, z_samples=8):
    """DINO 2D appearance descriptors lifted into genuine sampled 3D coordinates.

    Every sampled plane retains its physical Z. No projection/retrieval score is
    used: one 3D robust affine is fit across all sampled planes. This is a declared
    DINO-2D-descriptor/3D-correspondence method, not a native 3D neural backbone.
    """
    from .runner import extract_dinov3_descriptors
    features, indices = [], []
    zs = np.unique(np.linspace(0, geometry.shape_zyx[0] - 1,
                   min(z_samples, geometry.shape_zyx[0])).round().astype(int))
    for z in zs:
        plane = np.asarray(read_slice(int(z)), dtype=np.float32)
        low, high = np.percentile(plane, [1, 99])
        normalized = np.clip((plane - low) / max(high - low, 1.), 0, 1)
        image = Image.fromarray((normalized * 255).round().astype(np.uint8)).convert("RGB")
        descriptors, xy, _ = extract_dinov3_descriptors(backbone, image, layers, device, max_side)
        # XY centers are in edge-origin native image coordinates. Subtract .5
        # here because VolumeGeometry adds .5 for ordinal voxel-center indices.
        coordinates = np.column_stack([np.full(len(xy), z), xy[:, 1] - .5, xy[:, 0] - .5])
        features.append(descriptors)
        indices.append(coordinates)
    return np.concatenate(features), np.concatenate(indices)


def evaluate_clem_registration(backbone, entry, layers, device, max_side=256,
                               z_samples=8, ratio=.9, threshold_um=1., seed=0,
                               descriptor_cache=None):
    """Actual FM-channel and EM-volume DINO 3D correspondence evaluation.

    Fail closed on unmatched physical annotation frames, not on poor registration
    accuracy. In particular current EMPIAR-11666 needs its source frame resolved.
    """
    from czifile import CziFile
    entry = Path(entry)
    em_path = next((entry / "EM").glob("*.tif"))
    fm_path = next((entry / "FM").glob("*.czi"))
    em_geometry, fm_info = tiff_geometry(em_path), czi_metadata(fm_path)
    fm_geometry = fm_info["geometry"]
    _, source_landmarks, target_landmarks = bigwarp_landmarks(entry / "metadata/landmarks.csv")
    for landmarks, geometry in [(source_landmarks, fm_geometry), (target_landmarks, em_geometry)]:
        bound = np.asarray(geometry.shape_zyx)[::-1] * geometry.spacing_xyz_um
        if np.any(landmarks < 0) or np.any(landmarks > bound):
            raise ValueError(f"BigWarp landmarks exceed associated physical volume bounds: {entry.name}")
    mito = [i for i, channel in enumerate(fm_info["channels"])
            if "mito" in ((channel["name"] or "") + (channel["fluorophore"] or "")).lower()]
    if len(mito) != 1:
        raise ValueError("Exactly one explicit mitochondrial FM channel required")
    cache = descriptor_cache if descriptor_cache is not None else {}
    layer_key = layers if isinstance(layers, int) else tuple(layers)
    key = (id(backbone), str(entry), layer_key, max_side, z_samples, mito[0])
    if key not in cache:
        with CziFile(fm_path) as handle:
            array = handle.asarray()
        index = tuple(mito[0] if axis == "C" else slice(None) if axis in "ZYX" else 0 for axis in fm_info["axes"])
        fm = array[index]
        source_features, source_indices = extract_volume_descriptors(backbone, fm_geometry,
            lambda z: fm[z], layers, device, max_side, z_samples)
        with tifffile.TiffFile(em_path) as handle:
            target_features, target_indices = extract_volume_descriptors(backbone, em_geometry,
                lambda z: handle.pages[z].asarray(), layers, device, max_side, z_samples)
        cache[key] = source_features, target_features, source_indices, target_indices
    sf, tf, si, ti = cache[key]
    result = evaluate_volume_correspondence(sf, tf, si, ti, fm_geometry, em_geometry,
        source_landmarks, target_landmarks, ratio=ratio, threshold_um=threshold_um, seed=seed,
        matching_backend="torch", device=device)
    result.update({"entry": entry.name, "mitochondrial_channel": mito[0], "layers": layers,
        "max_side": max_side, "z_samples": z_samples, "ratio": ratio,
        "threshold_um": threshold_um, "seed": seed, "landmarks_used_for_fitting": False})
    return result


def clem_preflight(root, require_fm_metadata=True):
    """Validate all three real volume headers/annotation world-coordinate bounds."""
    records = []
    for entry in sorted(Path(root).glob("EMPIAR-*")):
        ems, fms = sorted((entry / "EM").glob("*.tif")), sorted((entry / "FM").glob("*.czi"))
        if len(ems) != 1 or len(fms) != 1:
            raise ValueError(f"Expected one processed EM and FM per entry: {entry}")
        em = tiff_geometry(ems[0])
        ids, source, target = bigwarp_landmarks(entry / "metadata/landmarks.csv")
        record = {"entry": entry.name, "em_path": str(ems[0]), "fm_path": str(fms[0]),
                  "em_shape_zyx": list(em.shape_zyx), "em_spacing_xyz_um": list(em.spacing_xyz_um),
                  "active_landmarks": len(ids), "source_world_bounds_xyz_um": [source.min(0).tolist(), source.max(0).tolist()],
                  "target_world_bounds_xyz_um": [target.min(0).tolist(), target.max(0).tolist()]}
        if require_fm_metadata:
            fm = czi_metadata(fms[0])
            record.update({"fm_axes": fm["axes"], "fm_shape": fm["shape"],
                "fm_shape_zyx": list(fm["geometry"].shape_zyx),
                "fm_spacing_xyz_um": list(fm["geometry"].spacing_xyz_um), "channels": fm["channels"]})
            source_bounds = np.asarray(fm["geometry"].shape_zyx)[::-1] * fm["geometry"].spacing_xyz_um
            record["source_landmarks_in_physical_bounds"] = bool(np.all(source >= 0) and np.all(source <= source_bounds))
        target_bounds = np.asarray(em.shape_zyx)[::-1] * em.spacing_xyz_um
        record["target_landmarks_in_physical_bounds"] = bool(np.all(target >= 0) and np.all(target <= target_bounds))
        records.append(record)
    if len(records) != 3:
        raise ValueError("Expected three independent CLEM volume pairs")
    return {"volume_pairs": 3, "records": records, "success": True,
            "task_ready": False, "limitation": "EMPIAR-11666 annotation frame mismatch; empirical 1TB selection and official mask/reference scoring remain required"}


def clem_cpu_smoke(entry, max_side=96, z_samples=16):
    """Real cross-modal volume appearance correspondence smoke, NOT hs6 scores.

    Calibrated 3D neighborhood appearance descriptors on actual FM/EM volumes
    drive OpenCV's genuine 3D robust affine estimator. Landmark GT is scoring
    ONLY. This deliberately small CPU baseline validates I/O and coordinate flow.
    """
    from czifile import CziFile
    entry = Path(entry)
    em_path = next((entry / "EM").glob("*.tif"))
    fm_path = next((entry / "FM").glob("*.czi"))
    em_geometry, fm_info = tiff_geometry(em_path), czi_metadata(fm_path)
    mito = [i for i, channel in enumerate(fm_info["channels"])
            if "mito" in ((channel["name"] or "") + (channel["fluorophore"] or "")).lower()]
    if len(mito) != 1:
        raise ValueError("Exactly one authoritative mitochondrial FM channel required")
    with CziFile(fm_path) as handle:
        image = handle.asarray()
    index = tuple(mito[0] if axis == "C" else slice(None) if axis in "ZYX" else 0 for axis in fm_info["axes"])
    fm = image[index]
    if tuple(fm.shape) != fm_info["geometry"].shape_zyx:
        raise ValueError("CZI channel extraction axes do not match calibrated ZYX geometry")
    def sampled_descriptors(geometry, read_slice):
        zs = np.unique(np.linspace(0, geometry.shape_zyx[0] - 1, min(z_samples, geometry.shape_zyx[0])).round().astype(int))
        scale = max_side / max(geometry.shape_zyx[1:])
        width, height = [max(8, round(d * scale)) for d in geometry.shape_zyx[:0:-1]]
        slices = []
        for z in zs:
            plane = np.asarray(read_slice(int(z)), dtype=np.float32)
            low, high = np.percentile(plane, [1, 99])
            plane = np.clip((plane - low) / max(high - low, 1.), 0, 1)
            slices.append(np.array(Image.fromarray(plane).resize((width, height), Image.Resampling.BILINEAR)))
        volume = gaussian_filter(np.stack(slices), .5)
        centers = np.array(np.meshgrid(np.arange(1, len(zs) - 1), np.arange(2, height - 2, 4),
                                      np.arange(2, width - 2, 4), indexing="ij")).reshape(3, -1).T
        descriptors = np.column_stack([volume[centers[:, 0] + dz, centers[:, 1] + dy, centers[:, 2] + dx]
                                     for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        # Deterministic high-contrast sampling excludes most uniform background.
        contrast = descriptors.std(1)
        keep = np.flatnonzero(contrast > .01)
        keep = keep[np.argsort(-contrast[keep], kind="stable")[:1500]]
        centers, descriptors = centers[keep].astype(float), descriptors[keep]
        centers[:, 0] = zs[centers[:, 0].astype(int)]
        centers[:, 1] = (centers[:, 1] + .5) * geometry.shape_zyx[1] / height - .5
        centers[:, 2] = (centers[:, 2] + .5) * geometry.shape_zyx[2] / width - .5
        return descriptors, centers
    fm_features, fm_indices = sampled_descriptors(fm_info["geometry"], lambda z: fm[z])
    with tifffile.TiffFile(em_path) as handle:
        em_features, em_indices = sampled_descriptors(em_geometry, lambda z: handle.pages[z].asarray())
    _, source, target = bigwarp_landmarks(entry / "metadata/landmarks.csv")
    result = evaluate_volume_correspondence(fm_features, em_features, fm_indices, em_indices,
        fm_info["geometry"], em_geometry, source, target, ratio=.9, threshold_um=1.)
    result.update({"entry": entry.name, "model": "CPU-3D-neighborhood-appearance-NOT-HS6",
        "source_descriptor_count": len(fm_features), "target_descriptor_count": len(em_features),
        "mitochondrial_channel": mito[0], "coordinate_unit": "micrometers", "z_samples": z_samples,
        "max_side": max_side, "landmarks_used_for_fitting": False})
    return result
