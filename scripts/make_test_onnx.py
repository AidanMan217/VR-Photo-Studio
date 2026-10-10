"""Build tiny stand-in ONNX models with the SAME input/output names and shapes as the
real ones, so the onnxruntime code path can be tested without downloading weights.

Usage:  python scripts/make_test_onnx.py OUT_DIR   (tests call ``build_all(out_dir)``)
Then set e.g. ``VRPHOTO_MODEL_DA2_SMALL=<OUT_DIR>/da2-small.onnx``.

Stubs (all opset 17, deterministic):
* da2-small:   pixel_values [1,3,H,W] (dynamic) -> predicted_depth [1,H,W] = mean over RGB
               (so brighter = "nearer"; lets tests check orientation & resizing).
* lama:        image [1,3,512,512], mask [1,1,512,512] -> output [1,3,512,512] in 0..255
               = 255*(image*(1-mask) + 0.5*mask)   (fills holes with mid-grey).
* birefnet-lite: input_image [1,3,1024,1024] -> output_image [1,1,1024,1024] logits
               = 8*(mean over RGB) (normalised input, so bright subject -> high logit).
* moge2-vitl:  image [1,3,H,W], num_tokens (int64 scalar) -> points [1,H,W,3] (a plane at
               z=2m seen through a 60 deg camera), normal [1,H,W,3], mask [1,H,W],
               metric_scale [1].
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _save(model, path: Path) -> Path:
    import onnx

    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def make_da2(path: Path) -> Path:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, "height", "width"])
    y = helper.make_tensor_value_info("predicted_depth", TensorProto.FLOAT, [1, "height", "width"])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    node = helper.make_node("ReduceMean", ["pixel_values", "axes"], ["m"], keepdims=0)
    relu = helper.make_node("Relu", ["m"], ["predicted_depth"])
    g = helper.make_graph([node, relu], "da2_stub", [x], [y], initializer=[axes])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 8
    return _save(m, path)


def make_lama(path: Path) -> Path:
    from onnx import TensorProto, helper

    img = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 512, 512])
    mask = helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, 512, 512])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 512, 512])
    one = helper.make_tensor("one", TensorProto.FLOAT, [], [1.0])
    half = helper.make_tensor("half", TensorProto.FLOAT, [], [0.5])
    s255 = helper.make_tensor("s255", TensorProto.FLOAT, [], [255.0])
    nodes = [
        helper.make_node("Sub", ["one", "mask"], ["inv"]),
        helper.make_node("Mul", ["image", "inv"], ["keep"]),
        helper.make_node("Mul", ["mask", "half"], ["fill"]),
        helper.make_node("Add", ["keep", "fill"], ["mix"]),
        helper.make_node("Mul", ["mix", "s255"], ["output"]),
    ]
    g = helper.make_graph(nodes, "lama_stub", [img, mask], [out], initializer=[one, half, s255])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    return _save(m, path)


def make_birefnet(path: Path) -> Path:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("input_image", TensorProto.FLOAT, [1, 3, 1024, 1024])
    y = helper.make_tensor_value_info("output_image", TensorProto.FLOAT, [1, 1, 1024, 1024])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    k = helper.make_tensor("k", TensorProto.FLOAT, [], [8.0])
    nodes = [
        helper.make_node("ReduceMean", ["input_image", "axes"], ["m"], keepdims=1),
        helper.make_node("Mul", ["m", "k"], ["output_image"]),
    ]
    g = helper.make_graph(nodes, "birefnet_stub", [x], [y], initializer=[axes, k])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 8
    return _save(m, path)


def make_moge(path: Path, h: int = 256, w: int = 256) -> Path:
    """Static-size MoGe-2 stand-in: outputs fixed geometry of a fronto-parallel plane."""
    from onnx import TensorProto, helper, numpy_helper

    f = (w / 2) / np.tan(np.radians(30))
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    # MoGe uses normalised image coords; points are in camera space (x right, y down, z fwd)
    z = np.full((h, w), 2.0, np.float32)
    pts = np.stack([(u - w / 2) / f * z, (v - h / 2) / f * z, z], -1)[None].astype(np.float32)
    nrm = np.zeros_like(pts)
    nrm[..., 2] = -1.0
    msk = np.ones((1, h, w), np.float32)
    scale = np.array([1.0], np.float32)

    img = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, h, w])
    tok = helper.make_tensor_value_info("num_tokens", TensorProto.INT64, [])
    outs = [
        helper.make_tensor_value_info("points", TensorProto.FLOAT, [1, h, w, 3]),
        helper.make_tensor_value_info("normal", TensorProto.FLOAT, [1, h, w, 3]),
        helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, h, w]),
        helper.make_tensor_value_info("metric_scale", TensorProto.FLOAT, [1]),
    ]
    inits = [
        numpy_helper.from_array(pts, "c_points"),
        numpy_helper.from_array(nrm, "c_normal"),
        numpy_helper.from_array(msk, "c_mask"),
        numpy_helper.from_array(scale, "c_scale"),
        numpy_helper.from_array(np.zeros((1, 3, h, w), np.float32), "zero_img"),
    ]
    # consume the inputs so they are not pruned: points = c_points + 0*sum(image)
    nodes = [
        helper.make_node("Mul", ["image", "zero_img"], ["z0"]),
        helper.make_node("ReduceSum", ["z0"], ["zs"], keepdims=0),
        helper.make_node("Cast", ["num_tokens"], ["tokf"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["tokf", "zs"], ["zz"]),
        helper.make_node("Add", ["c_points", "zz"], ["points"]),
        helper.make_node("Identity", ["c_normal"], ["normal"]),
        helper.make_node("Identity", ["c_mask"], ["mask"]),
        helper.make_node("Identity", ["c_scale"], ["metric_scale"]),
    ]
    g = helper.make_graph(nodes, "moge_stub", [img, tok], outs, initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    return _save(m, path)


def build_all(out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    return {
        "da2-small": make_da2(out / "da2-small.onnx"),
        "lama": make_lama(out / "lama.onnx"),
        "birefnet-lite": make_birefnet(out / "birefnet-lite.onnx"),
        "moge2-vitl": make_moge(out / "moge2-vitl.onnx"),
    }


def env_var_for(model_id: str) -> str:
    return "VRPHOTO_MODEL_" + model_id.upper().replace("-", "_").replace(".", "_")


if __name__ == "__main__":
    paths = build_all(sys.argv[1] if len(sys.argv) > 1 else "stub_models")
    for k, p in paths.items():
        print(f"{env_var_for(k)}={p}")
