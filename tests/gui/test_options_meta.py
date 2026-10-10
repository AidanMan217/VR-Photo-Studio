"""The option metadata must cover every job dataclass field (no Qt needed)."""

from __future__ import annotations

import dataclasses

import pytest

from vrphoto.gui import options_meta as om
from vrphoto.pipeline.jobs import JOB_TYPES, CommonOptions


def _fields(cls: type) -> list[dataclasses.Field]:
    return [f for f in dataclasses.fields(cls) if f.name not in om.STRUCTURAL_FIELDS]


def _default(f: dataclasses.Field):
    if f.default is not dataclasses.MISSING:
        return f.default
    return f.default_factory()  # type: ignore[misc]


@pytest.mark.parametrize("kind", sorted(JOB_TYPES))
def test_every_job_field_has_metadata(kind):
    meta = om.JOB_META.get(kind)
    assert meta is not None, f"No JOB_META entry for job kind {kind!r}"
    names = {f.name for f in _fields(JOB_TYPES[kind])}
    missing = names - set(meta)
    assert not missing, f"Add GUI metadata (gui/options_meta.py) for {kind} field(s): {sorted(missing)}"
    extra = set(meta) - names
    assert not extra, f"Metadata for fields that do not exist in {kind}: {sorted(extra)}"


def test_every_common_field_has_metadata():
    names = {f.name for f in dataclasses.fields(CommonOptions)}
    assert names - set(om.COMMON_META) == set(), "Add GUI metadata for new CommonOptions fields"
    assert set(om.COMMON_META) - names == set()


def test_no_metadata_for_unknown_job_kinds():
    assert set(om.JOB_META) <= set(JOB_TYPES)


def test_every_job_type_has_a_mode_card():
    assert {m.kind for m in om.MODES} == set(JOB_TYPES)
    assert len(om.MODES) == len(om.MODE_BY_KIND)
    for m in om.MODES:
        assert m.title and m.tagline and m.description
        assert m.accepts and m.accepts <= {"image", "video"}


def _all_meta():
    for kind, meta in om.JOB_META.items():
        cls = JOB_TYPES[kind]
        for f in _fields(cls):
            yield f"{kind}.{f.name}", meta[f.name], _default(f)
    for f in dataclasses.fields(CommonOptions):
        yield f"common.{f.name}", om.COMMON_META[f.name], _default(f)


@pytest.mark.parametrize(
    "name,meta,default", list(_all_meta()), ids=lambda x: x if isinstance(x, str) else ""
)
def test_metadata_is_consistent(name, meta, default):
    assert meta.kind in om.KINDS, name
    assert meta.label.strip(), name
    assert len(meta.tooltip) >= 20, f"{name}: tooltips should explain the option in plain words"
    if meta.kind == "combo":
        assert meta.choices or meta.choices_from, name
        if meta.choices and not meta.choices_from:
            assert default in meta.choice_values(), f"{name}: default {default!r} is not a choice"
            assert all(label for _, label in meta.choices)
    if meta.kind in ("slider", "spin"):
        assert meta.minimum is not None and meta.maximum is not None, name
        assert meta.minimum < meta.maximum
        assert meta.step > 0
        if default is not None:
            d = default * meta.scale
            assert meta.minimum - 1e-9 <= d <= meta.maximum + 1e-9, f"{name}: default {d} outside range"
            if meta.kind == "slider":
                pos = (d - meta.minimum) / meta.step
                assert abs(pos - round(pos)) < 1e-6, f"{name}: default {d} is not on the slider grid"
        else:
            assert meta.none_label, f"{name}: optional numeric fields need a none_label"
    if meta.enabled_if:
        other, allowed = meta.enabled_if
        assert allowed
    if meta.only_modes is not None:
        assert meta.only_modes <= set(JOB_TYPES)


def test_enabled_if_refers_to_sibling_fields():
    for kind, meta in om.JOB_META.items():
        for name, m in meta.items():
            if m.enabled_if:
                assert m.enabled_if[0] in meta, f"{kind}.{name} depends on unknown field"


@pytest.mark.parametrize("kind", sorted(om.PRESETS))
def test_presets_are_valid(kind):
    assert kind in JOB_TYPES
    presets = om.PRESETS[kind]
    assert set(presets) <= set(om.PRESET_NAMES)
    for pname, values in presets.items():
        for key, value in values.items():
            meta = om.field_meta(kind, key)
            if meta.kind == "combo" and meta.choices and not meta.choices_from:
                assert value in meta.choice_values(), f"{kind}/{pname}: {key}={value!r} is not a choice"
            if meta.kind in ("slider", "spin") and value is not None:
                d = value * meta.scale
                assert meta.minimum <= d <= meta.maximum, f"{kind}/{pname}: {key}={value} outside range"


def test_every_mode_has_presets():
    for m in om.MODES:
        assert m.kind in om.PRESETS
        assert set(om.PRESETS[m.kind]) == set(om.PRESET_NAMES)


def test_required_addons():
    assert om.required_addon_candidates("stereo", {}) == ()
    assert om.required_addon_candidates("capture", {}) == ("capture",)
    assert om.required_addon_candidates("object", {"method": "billboard"}) == ()
    assert om.required_addon_candidates("object", {"method": "trellis"}) == ("trellis",)
    assert om.required_addon_candidates("vr180", {"fill": "outpaint"}) == ("sdxl360",)
    assert om.required_addon_candidates("unknown", {}) == ()


def test_addon_choice_ids_match_manifests():
    """Add-on ids referenced by the GUI exist as manifests (when the manager is available)."""
    manager = pytest.importorskip("vrphoto.addons.manager")
    known = {m.id for m in manager.list_addons(include_experimental=True)}
    refs = {aid for m in om.MODES for aid in m.needs_addon}
    for meta in om.JOB_META.values():
        for fm in meta.values():
            for cands in fm.addon_choices.values():
                refs.add(cands[0])
    assert refs <= known, f"Unknown add-on ids in options_meta: {sorted(refs - known)}"
