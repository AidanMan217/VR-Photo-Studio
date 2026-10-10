"""Generic option panels: built from the job dataclasses for every job type; the JobSpec
they produce round-trips through job_to_dict / job_from_dict."""

from __future__ import annotations

import dataclasses
import json

import pytest

from vrphoto.core.errors import InputError
from vrphoto.gui import options_meta as om
from vrphoto.gui.panels import options as opt
from vrphoto.pipeline.jobs import JOB_TYPES, CommonOptions, job_from_dict, job_to_dict

pytestmark = pytest.mark.gui


def static_choices(name: str):
    return opt._FALLBACK_CHOICES[name]


@pytest.fixture
def panel_factory(qtbot):
    def make(kind: str, **kw):
        p = opt.JobOptionsPanel(kind, choice_provider=static_choices, **kw)
        qtbot.addWidget(p)
        return p

    return make


def test_field_types_resolve_optionals():
    from vrphoto.pipeline.jobs import PanoJob, ParallaxJob, StereoJob

    t = opt.field_types(StereoJob)
    assert t["convergence_m"] == opt.FieldType(float, True)
    assert t["strength"] == opt.FieldType(float, False)
    assert t["swap_eyes"] == opt.FieldType(bool, False)
    assert opt.field_types(ParallaxJob)["keyframes"] == opt.FieldType(list, True)
    assert opt.field_types(PanoJob)["input_is_360"] == opt.FieldType(bool, True)
    assert opt.field_types(CommonOptions)["output_dir"] == opt.FieldType(str, True)


@pytest.mark.parametrize("kind", sorted(JOB_TYPES))
def test_default_panel_builds_default_spec(panel_factory, kind):
    p = panel_factory(kind)
    spec = p.build_spec(["C:/photos/x.jpg"])
    assert type(spec) is JOB_TYPES[kind]
    expected = JOB_TYPES[kind](inputs=["C:/photos/x.jpg"])
    assert spec == expected, "a fresh panel must reproduce the dataclass defaults exactly"
    again = job_from_dict(json.loads(json.dumps(job_to_dict(spec))))
    assert again == spec


@pytest.mark.parametrize("kind", sorted(JOB_TYPES))
def test_every_field_has_an_editor(panel_factory, kind):
    p = panel_factory(kind)
    names = {f.name for f in dataclasses.fields(JOB_TYPES[kind])} - set(om.STRUCTURAL_FIELDS)
    assert set(p.form.editors) == names
    shown_common = {n for n, m in om.COMMON_META.items() if m.only_modes is None or kind in m.only_modes}
    assert set(p.common.editors) == shown_common
    # hidden common options still end up in the spec
    assert set(p.common_values()) == {f.name for f in dataclasses.fields(CommonOptions)}


@pytest.mark.parametrize("kind", sorted(om.PRESETS))
def test_presets_round_trip(panel_factory, kind):
    p = panel_factory(kind)
    for name in om.PRESETS[kind]:
        p.apply_preset(name)
        assert p.current_preset() == name
        spec = p.build_spec(["a.jpg"])
        d = job_to_dict(spec)
        for key, value in om.PRESETS[kind][name].items():
            got = d["common"][key.split(".", 1)[1]] if key.startswith("common.") else d[key]
            assert got == pytest.approx(value), f"{kind}/{name}: {key}"
        assert job_from_dict(d) == spec
    p.apply_preset("Default")
    assert p.build_spec(["a.jpg"]) == JOB_TYPES[kind](inputs=["a.jpg"])


def test_user_edit_switches_preset_to_custom(panel_factory):
    p = panel_factory("stereo")
    p.apply_preset("Strong 3D")
    p.form.editors["swap_eyes"].check.setChecked(True)
    assert p.current_preset() == "Custom"


def test_non_default_values_round_trip(panel_factory):
    p = panel_factory("stereo")
    p.form.set_values(
        {
            "layout": "tb",
            "strength": 1.35,
            "convergence_m": 2.5,
            "ipd_m": 0.0655,
            "swap_eyes": True,
            "image_format": "png",
        }
    )
    p.common.set_values({"hfov_deg": 72.0, "near_m": 0.8, "output_dir": "D:/out", "name": "beach"})
    spec = p.build_spec(["a.jpg"])
    assert spec.layout == "tb"
    assert spec.strength == pytest.approx(1.35)
    assert spec.convergence_m == pytest.approx(2.5)
    assert spec.ipd_m == pytest.approx(0.0655)
    assert spec.swap_eyes is True
    assert spec.common.hfov_deg == pytest.approx(72.0)
    assert spec.common.output_dir == "D:/out"
    assert spec.common.name == "beach"
    assert job_from_dict(job_to_dict(spec)) == spec
    # back to Auto / empty
    p.form.set_values({"convergence_m": None})
    p.common.set_values({"hfov_deg": None, "output_dir": None, "name": None})
    spec = p.build_spec(["a.jpg"])
    assert spec.convergence_m is None and spec.common.hfov_deg is None
    assert spec.common.output_dir is None and spec.common.name is None


def test_options_dict_load_round_trip(panel_factory):
    a = panel_factory("parallax")
    a.form.set_values(
        {
            "path": "custom",
            "keyframes": [{"t": 0, "x": -0.05}, {"t": 0.5, "x": 0.05}],
            "fps": 60.0,
            "width": 1920,
            "output": "vr180",
        }
    )
    d = a.options_dict()
    json.dumps(d)  # JSON-able
    b = panel_factory("parallax")
    b.load_options(d)
    assert b.options_dict() == d
    assert b.current_preset() == "Custom"
    spec = b.build_spec(["x.jpg"])
    assert spec.keyframes == [{"t": 0, "x": -0.05}, {"t": 0.5, "x": 0.05}]
    assert spec.width == 1920 and spec.fps == 60.0


def test_custom_path_needs_keyframes(panel_factory):
    p = panel_factory("parallax")
    p.form.set_values({"path": "custom"})
    with pytest.raises(InputError, match="key frames"):
        p.build_spec(["x.jpg"])


def test_invalid_json_raises_input_error(panel_factory):
    p = panel_factory("parallax")
    p.form.set_values({"path": "custom"})
    ed = p.form.editors["keyframes"]
    ed.edit.setPlainText("[{oops")
    assert ed.edit.property("invalid") is True
    with pytest.raises(InputError, match="JSON"):
        p.build_spec(["x.jpg"])
    ed.edit.setPlainText('{"t": 1}')
    with pytest.raises(InputError, match="list"):
        ed.value()


def test_near_must_be_less_than_far(panel_factory):
    p = panel_factory("scene")
    p.common.set_values({"near_m": 20.0, "far_m": 10.0})
    with pytest.raises(InputError, match="Nearest"):
        p.build_spec(["x.jpg"])


def test_enabled_if_greys_out_dependent_fields(panel_factory):
    p = panel_factory("object")
    assert p.form.editors["thickness"].isEnabled()
    assert not p.form.editors["seed"].isEnabled()
    p.form.set_values({"method": "trellis"})
    assert not p.form.editors["thickness"].isEnabled()
    assert p.form.editors["seed"].isEnabled()


def test_seed_and_texture_follow_the_method(panel_factory):
    """SHARP ignores the seed; 'Paint texture' only applies to Hunyuan3D."""
    p = panel_factory("object", show_experimental=True)
    for method, seed_on, texture_on in (
        ("billboard", False, False),
        ("sharp", False, False),
        ("trellis", True, False),
        ("trellis2", True, False),
        ("hunyuan", True, True),
    ):
        p.form.set_values({"method": method})
        assert p.form.editors["seed"].isEnabled() is seed_on, method
        assert p.form.editors["hunyuan_texture"].isEnabled() is texture_on, method
    p.form.set_values({"method": "hunyuan", "hunyuan_texture": True})
    assert p.build_spec(["x.jpg"]).hunyuan_texture is True


def test_capture_subject_height_shows_automatic(panel_factory):
    p = panel_factory("capture")
    ed = p.form.editors["subject_height_m"]
    assert ed.spin.text() == "Automatic" and ed.value() == 0.0
    p.form.set_values({"subject_height_m": 1.75})
    assert p.build_spec(["a.jpg", "b.jpg"]).subject_height_m == pytest.approx(1.75)


def test_experimental_choices_hidden_unless_enabled(panel_factory):
    p = panel_factory("object")
    vals = p.form.editors["method"].values()
    assert "trellis" in vals and "hunyuan" not in vals and "trellis2" not in vals
    p.refresh_choices(True)
    assert "hunyuan" in p.form.editors["method"].values()
    q = panel_factory("object", show_experimental=True)
    assert "trellis2" in q.form.editors["method"].values()


def test_hidden_remembered_values_are_kept(panel_factory):
    p = panel_factory("object")
    p.form.set_values({"method": "hunyuan"})  # experimental, hidden choice
    assert p.form.editors["method"].value() == "hunyuan"
    assert p.build_spec(["x.jpg"]).method == "hunyuan"


def test_common_defaults_from_settings(panel_factory):
    p = panel_factory(
        "stereo",
        common_defaults={"depth_model": "moge2", "quest_model": "quest2", "segment_model": "grabcut"},
    )
    spec = p.build_spec(["x.jpg"])
    assert spec.common.depth_model == "moge2"
    assert spec.common.quest_model == "quest2"
    assert spec.common.segment_model == "grabcut"  # hidden for stereo but still applied
    p.set_common_defaults({"depth_model": "da2-small"}, {"depth_model": "da2-small"})
    assert p.build_spec(["x.jpg"]).common.depth_model == "da2-small"


def test_slider_editor_keeps_exact_values(qtbot):
    meta = om.JOB_META["stereo"]["strength"]
    ed = opt.SliderEditor("strength", meta, opt.FieldType(float, False))
    qtbot.addWidget(ed)
    ed.set_value(1.03)  # off the slider grid
    assert ed.value() == pytest.approx(1.03)
    ed.slider.setValue(ed.slider.maximum())
    assert ed.value() == pytest.approx(meta.maximum)
    ipd = opt.SliderEditor("ipd_m", om.JOB_META["stereo"]["ipd_m"], opt.FieldType(float, False))
    qtbot.addWidget(ipd)
    ipd.set_value(0.063)
    assert ipd.spin.value() == pytest.approx(63.0)
    assert ipd.value() == 0.063


def test_spin_editor_optional_uses_auto(qtbot):
    meta = om.COMMON_META["hfov_deg"]
    ed = opt.SpinEditor("hfov_deg", meta, opt.FieldType(float, True))
    qtbot.addWidget(ed)
    ed.set_value(None)
    assert ed.value() is None
    assert ed.spin.text() == "Auto"
    ed.set_value(80)
    assert ed.value() == 80.0
    ed.spin.setValue(ed.spin.minimum())
    assert ed.value() is None


def test_int_editors_return_ints(qtbot):
    meta = om.JOB_META["scene"]["target_faces"]
    ed = opt.SliderEditor("target_faces", meta, opt.FieldType(int, False))
    qtbot.addWidget(ed)
    ed.set_value(300_000)
    assert ed.value() == 300_000 and isinstance(ed.value(), int)
    w = opt.SpinEditor("width", om.JOB_META["parallax"]["width"], opt.FieldType(int, True))
    qtbot.addWidget(w)
    w.set_value(None)
    assert w.value() is None
    w.set_value(1280)
    assert w.value() == 1280 and isinstance(w.value(), int)


def test_combo_editor_distinguishes_none_and_bools(qtbot):
    meta = om.JOB_META["pano"]["input_is_360"]
    ed = opt.ComboEditor("input_is_360", meta, opt.FieldType(bool, True))
    qtbot.addWidget(ed)
    for v in (None, True, False):
        ed.set_value(v)
        assert ed.value() is v
    fps = opt.ComboEditor("fps", om.JOB_META["parallax"]["fps"], opt.FieldType(float, False))
    qtbot.addWidget(fps)
    fps.set_value(30)  # int matches 30.0
    assert fps.combo.count() == 3
    fps.set_value(25.0)  # unknown value is kept
    assert fps.value() == 25.0 and fps.combo.count() == 4


def test_text_and_path_editors(qtbot):
    ed = opt.make_editor("name", om.COMMON_META["name"], opt.FieldType(str, True))
    qtbot.addWidget(ed)
    assert ed.value() is None
    ed.edit.setText("  beach ")
    assert ed.value() == "beach"
    pe = opt.make_editor("output_dir", om.COMMON_META["output_dir"], opt.FieldType(str, True))
    qtbot.addWidget(pe)
    assert isinstance(pe, opt.PathEditor)
    pe.set_value("C:/x")
    assert pe.value() == "C:/x"


def test_dynamic_choices_fail_soft(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def boom(name, *a, **kw):
        if "depth.base" in name or name.endswith("depth.base"):
            raise ImportError("not built yet")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", boom)
    choices = opt.dynamic_choices("depth_models")
    assert ("auto", opt._FALLBACK_CHOICES["depth_models"][0][1]) == choices[0]
    assert opt.dynamic_choices("nonexistent") == []


def test_live_dynamic_choices_contain_auto():
    for name in ("depth_models", "inpaint_models", "segment_models", "devices"):
        vals = [v for v, _ in opt.dynamic_choices(name)]
        assert "auto" in vals, name


def test_section_toggle(qtbot):
    s = opt.Section("Advanced")
    qtbot.addWidget(s)
    s.show()
    assert not s.expanded and not s.body.isVisible()
    s.button.click()
    assert s.expanded and s.body.isVisible()


def test_device_choices_never_block(monkeypatch):
    """dynamic_choices('devices') must not enumerate GPUs on the UI thread."""
    from vrphoto.models import gpu

    calls = []
    monkeypatch.setattr(
        gpu,
        "device_choices",
        lambda: calls.append(1) or [("auto", "Automatic"), ("cpu", "CPU only"), ("0", "RTX 4070 (12 GB)")],
    )
    monkeypatch.setattr(opt, "_device_choices", None)
    assert [v for v, _ in opt.dynamic_choices("devices")] == ["auto", "cpu"]
    assert calls == []
    assert [v for v, _ in opt.detect_device_choices()] == ["auto", "cpu", "0"]
    assert calls == [1]
    assert [v for v, _ in opt.dynamic_choices("devices")] == ["auto", "cpu", "0"]


def test_invalid_greyed_out_field_does_not_block_jobs(panel_factory):
    """Broken key frames left over from 'Custom' must not stop a 'Circle' job."""
    p = panel_factory("parallax")
    p.form.set_values({"path": "custom"})
    p.form.editors["keyframes"].edit.setPlainText("[{broken")
    with pytest.raises(InputError, match="JSON"):
        p.build_spec(["x.jpg"])
    p.form.set_values({"path": "circle"})
    assert not p.form.editors["keyframes"].isEnabled()
    spec = p.build_spec(["x.jpg"])
    assert spec.path == "circle" and spec.keyframes is None


def test_lenient_values_skip_invalid_entries(panel_factory):
    p = panel_factory("parallax")
    p.common.set_values({"depth_model": "moge2"})
    p.form.set_values({"path": "custom"})
    p.form.editors["keyframes"].edit.setPlainText("[{broken")
    with pytest.raises(InputError):
        p.options_dict()
    d = p.options_dict(strict=False)
    assert "keyframes" not in d and d["common"]["depth_model"] == "moge2"
