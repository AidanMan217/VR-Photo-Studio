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
    for other, allowed in meta.enabled_if_all:
        assert isinstance(other, str) and isinstance(allowed, tuple) and allowed, name
    assert meta.conditions() == ((meta.enabled_if,) if meta.enabled_if else ()) + meta.enabled_if_all
    if meta.only_modes is not None:
        assert meta.only_modes <= set(JOB_TYPES)
    if meta.addon_optional:
        assert meta.addon_choices, f"{name}: addon_optional without addon_choices"
    if meta.file_filter:
        assert meta.kind == "file" and meta.file_filter in ("images",), name


def test_enabled_if_refers_to_sibling_fields():
    for kind, meta in om.JOB_META.items():
        for name, m in meta.items():
            for other, allowed in m.conditions():
                assert other in meta, f"{kind}.{name} depends on unknown field {other!r}"
                assert other != name, f"{kind}.{name} depends on itself"
                sib = meta[other]
                if sib.kind == "combo" and sib.choices and not sib.choices_from:
                    unknown = set(allowed) - set(sib.choice_values())
                    assert not unknown, f"{kind}.{name}: {other} has no choice(s) {sorted(unknown)}"


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
    # SHARP 360 needs only SHARP: the built-in back needs nothing
    req = om.required_addon_candidates
    assert req("object", {"method": "sharp360"}) == ("sharp",)
    assert req("object", {"method": "sharp360", "back_source": "shell"}) == ("sharp",)
    assert req("object", {"method": "billboard", "back_source": "auto"}) == ()
    # the flattened wrapper lists the optional back source too
    assert req("object", {"method": "sharp360", "back_source": "trellis"}) == ("sharp", "trellis")
    assert req("object", {"method": "billboard", "back_source": "trellis"}) == ()


def test_required_addon_groups():
    """docs/design/SHARP360.md §1.7 / §5.5: (any-of ids, optional) groups."""
    groups = om.required_addon_groups
    assert groups("stereo", {}) == ()
    assert groups("unknown", {"method": "trellis"}) == ()
    assert groups("capture", {}) == ((("capture",), False),)
    assert groups("object", {"method": "trellis"}) == ((("trellis",), False),)
    assert groups("vr180", {"fill": "outpaint"}) == ((("sdxl360",), False),)
    sharp360 = {"method": "sharp360", "back_source": "trellis"}
    assert groups("object", sharp360) == ((("sharp",), False), (("trellis",), True))
    for source in ("auto", "shell", "images"):  # nothing to install: the job picks / needs none
        assert groups("object", {"method": "sharp360", "back_source": source}) == ((("sharp",), False),)
    # a remembered TRELLIS back on another method asks for nothing more (greyed-out option)
    assert groups("object", {"method": "billboard", "back_source": "trellis"}) == ()
    assert groups("object", {"method": "trellis", "back_source": "trellis"}) == ((("trellis",), False),)
    # broken remembered values name no add-on
    assert groups("object", {"method": ["x"], "back_source": {"a": 1}}) == ()


def test_required_addon_groups_follow_the_full_job_values():
    """The GUI passes every option of the form: the defaults of an ObjectJob need nothing."""
    from vrphoto.pipeline.jobs import ObjectJob

    values = {f.name: getattr(ObjectJob(), f.name) for f in _fields(ObjectJob)}
    assert om.required_addon_groups("object", values) == ()
    values.update(method="sharp360", back_source="trellis")
    assert om.required_addon_groups("object", values) == ((("sharp",), False), (("trellis",), True))
    values.update(method="sharp")
    assert om.required_addon_groups("object", values) == ((("sharp",), False),)


def test_a_group_needed_both_ways_is_required(monkeypatch):
    obj = dict(om.JOB_META["object"])
    back = obj["back_source"]
    monkeypatch.setitem(
        om.JOB_META,
        "object",
        {**obj, "back_source": dataclasses.replace(back, addon_choices={"trellis": ("sharp",)})},
    )
    assert om.required_addon_groups("object", {"method": "sharp360", "back_source": "trellis"}) == (
        (("sharp",), False),
    )


# --------------------------------------------------------------------------- field_applies
def test_field_applies_follows_enabled_if_and_enabled_if_all():
    fa = om.field_applies
    assert fa("object", "method", {})  # no condition
    assert fa("object", "seed", {"method": "trellis"}) and not fa("object", "seed", {"method": "billboard"})
    on = {"method": "sharp360", "back_source": "images"}
    assert fa("object", "back_images", on)
    assert not fa("object", "back_images", {**on, "back_source": "trellis"})  # enabled_if_all
    assert not fa("object", "back_images", {**on, "method": "sharp"})  # enabled_if
    for source, tries_on in (("auto", True), ("trellis", True), ("shell", False), ("images", False)):
        vals = {"method": "sharp360", "back_source": source}
        for name in ("back_tries", "back_steps"):
            assert fa("object", name, vals) is tries_on, (name, source)
    # unknown kinds / fields and the Common options never grey anything out
    assert fa("nope", "x", {}) and fa("object", "not_a_field", {"method": "billboard"})
    assert fa("common", "depth_model", {})


def test_field_applies_missing_siblings_do_not_restrict():
    """The panel leaves out invalid entries: their value greys nothing out."""
    assert om.field_applies("object", "back_images", {})
    assert om.field_applies("object", "back_images", {"method": "sharp360"})
    assert not om.field_applies("object", "back_images", {"back_source": "shell"})


def test_field_applies_is_transitive_and_safe_with_cycles():
    """A field under a greyed-out switch is greyed out even if it forgot the switch."""
    meta = {
        "mode": om.FieldMeta("Mode", "x" * 20, "combo", choices=(("a", "A"), ("b", "B"))),
        "sub": om.FieldMeta(
            "Sub", "x" * 20, "combo", choices=(("p", "P"), ("q", "Q")), enabled_if=("mode", ("a",))
        ),
        "leaf": om.FieldMeta("Leaf", "x" * 20, "check", enabled_if=("sub", ("p",))),
        "loop1": om.FieldMeta("L1", "x" * 20, "check", enabled_if=("loop2", (True,))),
        "loop2": om.FieldMeta("L2", "x" * 20, "check", enabled_if=("loop1", (True,))),
    }
    vals = {"mode": "a", "sub": "p", "leaf": True, "loop1": True, "loop2": True}
    assert om.field_applies(None, "leaf", vals, meta=meta)
    assert not om.field_applies(None, "leaf", {**vals, "mode": "b"}, meta=meta)  # via "sub"
    assert not om.field_applies(None, "leaf", {**vals, "sub": "q"}, meta=meta)
    assert om.field_applies(None, "loop1", vals, meta=meta)
    assert not om.field_applies(None, "loop1", {**vals, "loop1": False}, meta=meta)  # loop2 off


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


# --------------------------------------------------------------------------- review fixes
def test_seed_is_not_offered_for_sharp():
    """SHARP is deterministic and ignores the seed: 'Variation' must be greyed out for it.
    SHARP 360 uses it (TRELLIS's result and the seam pattern), design §0.3."""
    meta = om.JOB_META["object"]["seed"]
    other, allowed = meta.enabled_if
    assert other == "method"
    assert "sharp" not in allowed
    assert set(allowed) == {"trellis", "trellis2", "hunyuan", "sharp360"}
    assert not meta.enabled_if_all
    assert "SHARP 360" in meta.tooltip and "Plain SHARP" in meta.tooltip


# --------------------------------------------------------------------------- SHARP 360
SHARP360_FIELDS = ("back_source", "back_quality", "seam_deg", "size_from", "save_extras")
#: M2 fields: (name, extra all-of condition on back_source or None)
SHARP360_M2_FIELDS = (
    ("back_images", ("images",)),
    ("back_relief", None),
    ("color_match", None),
    ("back_tries", ("auto", "trellis")),
    ("back_steps", ("auto", "trellis")),
)


def test_sharp360_is_an_experimental_method_that_needs_sharp():
    meta = om.JOB_META["object"]["method"]
    assert "sharp360" in meta.choice_values()
    assert "sharp360" in meta.experimental_choices
    assert meta.addon_choices["sharp360"] == ("sharp",)
    label = dict(meta.choices)["sharp360"]
    assert label.startswith("SHARP 360") and "experimental" in label
    assert "SHARP 360" in meta.tooltip and "walk all the way round" in meta.tooltip


def test_sharp360_choices_match_the_runner():
    """The GUI offers exactly what the job accepts (pipeline/run.py constants)."""
    from vrphoto.pipeline import run as R
    from vrphoto.pipeline.jobs import ObjectJob

    obj = om.JOB_META["object"]
    assert tuple(obj["back_source"].choice_values()) == R.BACK_SOURCES
    assert tuple(obj["back_quality"].choice_values()) == R.BACK_QUALITIES
    assert tuple(obj["size_from"].choice_values()) == R.SIZE_FROM
    assert "sharp360" in R.OBJECT_METHODS
    job = ObjectJob()
    assert (job.back_source, job.back_quality, job.size_from) == ("auto", "balanced", "auto")
    assert job.seam_deg == 60.0 and job.save_extras is False


def test_sharp360_fields_metadata():
    obj = om.JOB_META["object"]
    for name in SHARP360_FIELDS:
        assert obj[name].enabled_if == ("method", ("sharp360",)), name
        assert "SHARP 360" in obj[name].tooltip, name
    assert obj["back_source"].label == "Back made by"
    assert dict(obj["back_source"].choices) == {
        "auto": "Automatic (best installed)",
        "shell": "Built-in shape (no download)",
        "trellis": "TRELLIS (add-on)",
        "images": "Your own back photo",
    }
    assert not obj["back_source"].advanced and not obj["back_quality"].advanced
    assert dict(obj["back_quality"].choices)["best"].startswith("Best")
    seam = obj["seam_deg"]
    assert (seam.kind, seam.minimum, seam.maximum, seam.step, seam.unit) == ("slider", 35, 80, 5, "°")
    assert seam.advanced and seam.label == "Keep the photo up to"
    assert obj["size_from"].advanced
    assert dict(obj["size_from"].choices) == {
        "auto": "Automatic",
        "height": "Real-world height",
        "photo": "Measured by SHARP",
    }
    assert obj["save_extras"].kind == "check" and obj["save_extras"].advanced
    # TRELLIS is optional: the job warns and falls back to the built-in back without it
    assert obj["back_source"].addon_choices == {"trellis": ("trellis",)}
    assert obj["back_source"].addon_optional
    assert not obj["method"].addon_optional  # SHARP itself is required


def test_sharp360_m2_fields_metadata():
    """Back photo, relief, colour match and the AI knobs (design §1.2)."""
    from vrphoto.pipeline import run as R
    from vrphoto.pipeline.jobs import ObjectJob

    obj = om.JOB_META["object"]
    for name, sources in SHARP360_M2_FIELDS:
        m = obj[name]
        assert m.enabled_if == ("method", ("sharp360",)), name
        assert m.enabled_if_all == ((("back_source", sources),) if sources else ()), name
        assert "SHARP 360" in m.tooltip and len(m.tooltip) > 150, name
    back = obj["back_images"]
    assert (back.kind, back.label, back.file_filter, back.advanced) == ("file", "Back photo", "images", False)
    assert ObjectJob().back_images is None
    for word in ("right", "left", "back_right", "back_left", "front_right", "front_left"):
        assert word in back.tooltip, word
    assert "walked to your right" in back.tooltip and "AI" in back.tooltip and "behind" in back.tooltip
    relief = obj["back_relief"]
    assert (relief.kind, relief.label, relief.advanced) == ("combo", "Back detail", True)
    assert tuple(relief.choice_values()) == R.BACK_RELIEFS
    cm = obj["color_match"]
    assert (cm.kind, cm.label, cm.advanced) == ("check", "Match colours to the photo", True)
    assert ObjectJob().color_match is True
    for name, label, hi in (("back_tries", "AI attempts", 4), ("back_steps", "AI steps", 100)):
        m = obj[name]
        assert (m.kind, m.label, m.minimum, m.maximum, m.zero_label, m.advanced) == (
            "spin",
            label,
            0,
            hi,
            "Automatic",
            True,
        ), name
        assert getattr(ObjectJob(), name) == 0
        assert "GTX 1080" in m.tooltip, name
    # the plain-language source tooltip names every choice and the expected time
    tip = obj["back_source"].tooltip
    for word in ("Automatic", "Built-in shape", "TRELLIS", "Your own back photo", "GTX 1080", "minutes"):
        assert word in tip, word


def test_thickness_and_back_apply_to_the_sharp360_built_in_back():
    obj = om.JOB_META["object"]
    for name in ("thickness", "back"):
        assert set(obj[name].enabled_if[1]) == {"billboard", "sharp360"}, name
        assert "SHARP 360" in obj[name].tooltip
    assert "back of their head" in obj["back"].tooltip  # mirror puts a face on the back


def test_object_presets_set_the_back_quality():
    presets = om.PRESETS["object"]
    assert [presets[n]["back_quality"] for n in om.PRESET_NAMES] == ["fast", "balanced", "best"]


def test_hunyuan_texture_checkbox_only_for_hunyuan():
    from vrphoto.pipeline.jobs import ObjectJob

    meta = om.JOB_META["object"]["hunyuan_texture"]
    assert meta.kind == "check"
    assert meta.enabled_if == ("method", ("hunyuan",))
    assert meta.advanced  # experimental method: not a greyed-out row in the basic form
    assert ObjectJob().hunyuan_texture is False  # opt-in: slower, more VRAM, extra download


def test_flat_living_photo_is_not_forced_to_hevc():
    """'Automatic' gives flat videos H.264; no preset may force a Quest HEVC preset onto the
    default flat (phone / PC) output."""
    from vrphoto.pipeline.jobs import ParallaxJob

    assert ParallaxJob().output == "flat" and ParallaxJob().encoder_preset == "auto"
    for name, values in om.PRESETS["parallax"].items():
        assert "hevc" not in str(values.get("encoder_preset", "auto")), name
    tip = om.JOB_META["parallax"]["encoder_preset"].tooltip
    assert "H.264" in tip and "flat" in tip.lower()
    assert "1920" in om.JOB_META["parallax"]["width"].tooltip


def test_object_comfortable_preset_is_thinner_than_default():
    from vrphoto.pipeline.jobs import ObjectJob

    presets = om.PRESETS["object"]
    assert presets["Comfortable"]["thickness"] < ObjectJob().thickness < presets["Strong 3D"]["thickness"]


def test_scene_detail_is_described_as_a_total():
    """The 'Detail' budget covers the whole scene, not each layer."""
    tip = om.JOB_META["scene"]["target_faces"].tooltip
    assert "in total" in tip and "all layers" in tip


def test_capture_subject_height_defaults_to_automatic():
    from vrphoto.pipeline.jobs import CaptureJob

    meta = om.JOB_META["capture"]["subject_height_m"]
    assert CaptureJob().subject_height_m == 0.0
    assert meta.kind == "spin" and meta.minimum == 0 and meta.zero_label == "Automatic"
    assert meta.unit == "m" and meta.advanced
