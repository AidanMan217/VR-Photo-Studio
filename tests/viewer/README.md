# WebXR viewer tests

Tests for the browser viewer in `src/vrphoto/webxr/viewer/` (gallery `index.html`, viewer
`view.html`, `js/`, vendored libraries in `vendor/`). Nothing in this folder ships with the app.

| What | How it runs | Needs |
| --- | --- | --- |
| `test_viewer_static.py` | pytest | Python, Node.js (JS parts skip without it) |
| `test_viewer_tools.py` | pytest | Python (vendor script, fixture builder, test server) |
| `specs/*.spec.mjs` | Playwright | Node.js, `npm ci`, a Chromium |
| `test_viewer_browser.py` | pytest wrapper for Playwright | `VRPHOTO_BROWSER_TESTS=1` |

## Static checks (fast, no browser)

```bash
python -m pytest -q tests/viewer
```

* `check_imports.mjs` parses every `.html` / `.js` file under the viewer (import maps, static
  imports via V8's own module parser, literal `import()` strings via a small lexer) and verifies
  that every specifier resolves to an existing local file, that pages reference only existing
  scripts / stylesheets, and that nothing reachable at runtime points at a CDN. Run it by hand
  with `node tests/viewer/check_imports.mjs` (add `--json` for the module graph).
* Every `.js` file is syntax-checked with `node --input-type=module --check < file`.
  (Plain `node --check file.js` silently accepts broken ES-module files, so do not use it.)
* `js_manifest_check.mjs` runs the viewer's own manifest validator in Node so pytest can compare
  it with `vrphoto.webxr.scene.validate_manifest`.
* `python scripts/vendor_web.py --check` (also run by the tests) verifies the vendored files
  against `vendor/VENDOR.lock.json` without network access. `vendor/.gitattributes` (written by
  the script) keeps git from converting line endings; a checkout that converted them anyway
  (Windows `core.autocrlf=true`) still passes, because the check also accepts CRLF copies of
  the recorded files.

## Browser tests (Playwright)

```bash
cd tests/viewer
npm ci                                        # @playwright/test + iwer (pinned in package.json)
npx playwright install --with-deps chromium   # CI / any machine that can download browsers
npx playwright test                           # or: npm test
```

The config starts `serve_fixtures.py` (a stand-in for the app's `ViewerServer` with the same
routes: `/`, `/view.html`, `/api/scenes`, `/scenes/<id>/...` with Range support) on
`http://127.0.0.1:8765`, which builds a fresh fixture library with `make_fixtures.py`: a two-layer
photo3d GLB, SBS stereo image, SBS VP8 video (when ffmpeg has libvpx), VR180 SBS, 360 mono with a
depth map, an object as GLB and as `.splat`, and a small splat room (inline writers, no other
modules needed), plus integration scenes made with the app's own geometry code exactly as the
pipeline does (`fx-real-photo3d`: `layered_meshes` + `write_glb`; `fx-real-object`: `billboard`
GLB with a `write_splats` .splat; `fx-real-splat-ply`: `write_splats` PLY). Python with numpy and
Pillow is required (`pip install -e .` in CI; without an editable install the repository's `src/`
is put on `sys.path`). Environment knobs: `PYTHON` (interpreter for the server),
`VIEWER_TEST_PORT`, `CHROMIUM_PATH`, and `VIEWER_TEST_SERVER=app` to run the very same specs
against the app's real `vrphoto.webxr.server.ViewerServer` (HTTP only) instead of the stand-in.
`VIEWER_TEST_LIBRARY=<folder>` serves an existing library instead of the fixtures, e.g. scenes
exported by the app (`%LOCALAPPDATA%\VRPhotoStudio\library` on Windows); combine it with
`npx playwright test -g "every fixture scene"` (the other specs expect the `fx-*` fixtures).

What the specs check:

* `gallery.spec.mjs` - the gallery lists every scene (title, thumbnail, mode badge, link), the
  empty state explains how to export, server errors show a friendly message with "Try again",
  the insecure-origin banner, phone-width layout.
* `viewer.spec.mjs` - every fixture scene loads without console errors and renders non-black,
  non-flat pixels in the desktop preview; stereo eye layers; VR180 orientation; WASD; video
  controls; friendly errors for a missing scene, damaged/invalid `scene.json`, a missing asset,
  a missing `?scene=`, and an insecure origin; a scene switch in the middle of loading a splat
  scene leaves no renderer behind.
* `xr.spec.mjs` - with [IWER](https://github.com/meta-quest/immersive-web-emulation-runtime)
  emulating a Quest 3 (injected by the test only, never shipped): Enter VR renders frames for
  several modes, photo3d recenters the head onto the photo viewpoint, the Y/B menu and in-VR
  scene switching, snap turn / smooth locomotion / arc teleport, and in AR: hit-test reticle,
  trigger placement on the floor, anchors, grip-drag, two-hand scaling, reset scale and the
  passthrough toggle; an app-made billboard stands on the floor at its real height; a VR-only
  scene opened inside AR turns passthrough off (and back on); a refused session shows a
  dismissible message; Enter VR while loading explains itself; Enter VR turns video sound on.
  A small floor-only "synthetic environment" answers hit-test rays.

IWER 2.5 note: its `XRReferenceSpace.getOffsetReferenceSpace()` takes a raw matrix instead of
the spec's `XRRigidTransform`, so `specs/helpers.mjs` installs a one-line shim. The viewer code
follows the WebXR spec as implemented by the Quest Browser.

### Running locally without browser downloads

Where Playwright cannot download browsers (e.g. sandboxed containers), the npm package
`@sparticuz/chromium` ships a Chromium binary; the config uses it automatically if present:

```bash
cd tests/viewer
npm ci
npm install --no-save @sparticuz/chromium@153.0.0
npx playwright test
```

Alternatively point `CHROMIUM_PATH` at any Chromium/Chrome executable.

### From pytest

```bash
VRPHOTO_BROWSER_TESTS=1 python -m pytest -q tests/viewer/test_viewer_browser.py
```

## Fixtures by hand

```bash
python tests/viewer/make_fixtures.py /tmp/vr-lib            # write the fixture library
python tests/viewer/serve_fixtures.py --library /tmp/vr-lib --port 8765
# open http://127.0.0.1:8765/ in a browser
```
