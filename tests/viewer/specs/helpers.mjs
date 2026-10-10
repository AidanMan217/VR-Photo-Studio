// Shared helpers for the viewer browser tests.

import { expect } from '@playwright/test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

/** Console noise that is not a viewer problem. */
const BENIGN = [
  /GPU stall due to ReadPixels/i, // SwiftShader perf note from our pixel read-back
  /Could not find \w+ in the model/i, // upstream controller asset quirk (three's XRControllerModelFactory)
  /Requested frame rate is the same/i, // IWER info
  /Automatic fallback to software WebGL has been deprecated/i,
];

/** Record console errors / warnings and uncaught page errors. */
export function watchConsole(page) {
  const log = { errors: [], warnings: [] };
  page.on('console', (msg) => {
    const text = msg.text();
    if (BENIGN.some((re) => re.test(text))) return;
    if (msg.type() === 'error') log.errors.push(text);
    else if (msg.type() === 'warning') log.warnings.push(text);
  });
  page.on('pageerror', (err) => log.errors.push(`pageerror: ${err.message}`));
  return log;
}

export async function sceneList(request) {
  const res = await request.get('/api/scenes');
  expect(res.ok()).toBeTruthy();
  return res.json();
}

/** Wait until view.html reports the scene as loaded (or fail with its error). */
export async function waitReady(page) {
  await page.waitForFunction(() => window.__vrphoto?.state?.ready || window.__vrphoto?.state?.error, null, {
    timeout: 60_000,
  });
  const state = await page.evaluate(() => window.__vrphoto.state);
  expect(state.error, `viewer error: ${state.error}`).toBeNull();
  return state;
}

/** Pixel statistics of the next rendered frame; polls until something visible is drawn. */
export async function waitForPixels(page, minLit = 0.02) {
  let stats = null;
  await expect
    .poll(
      async () => {
        stats = await page.evaluate(() => window.__vrphoto.capture());
        return stats?.litFraction ?? 0;
      },
      { timeout: 30_000, intervals: [100, 250, 500] },
    )
    .toBeGreaterThan(minLit);
  return stats;
}

export const state = (page) => page.evaluate(() => window.__vrphoto.state);

/**
 * Install IWER (Meta's WebXR emulator, test-only) as an emulated Quest 3 before the page loads,
 * plus a tiny synthetic environment whose floor (y = 0) answers hit-test rays.
 */
export async function installQuest3(page) {
  await page.addInitScript({ path: require.resolve('iwer/build/iwer.min.js') });
  await page.addInitScript(() => {
    class FloorEnvironment {
      constructor() {
        this.version = 'floor-test';
        this.planesVisible = false;
        this.boundingBoxesVisible = false;
        this.meshesVisible = false;
        this._canvas = document.createElement('canvas');
        this._planes = new Set();
        this._meshes = new Set();
      }
      render() {}
      loadEnvironment() {}
      loadDefaultEnvironment() {
        return Promise.resolve();
      }
      deleteAll() {}
      get environmentCanvas() {
        return this._canvas;
      }
      get trackedPlanes() {
        return this._planes;
      }
      get trackedMeshes() {
        return this._meshes;
      }
      computeDepthBuffer() {
        return null;
      }
      /** Ray (column-major 4x4, looking down -z) against the floor plane y = 0. */
      computeHitTestResults(m) {
        const o = [m[12], m[13], m[14]];
        const d = [-m[8], -m[9], -m[10]];
        if (d[1] > -1e-4) return [];
        const t = -o[1] / d[1];
        if (t <= 0) return [];
        return [new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, o[0] + t * d[0], 0, o[2] + t * d[2], 1])];
      }
    }
    // eslint-disable-next-line no-undef
    const device = new IWER.XRDevice(IWER.metaQuest3);
    device.installRuntime({ forceInstall: true });
    device.installSEM(FloorEnvironment);
    window.__iwer = device;
    // IWER 2.5 bug: XRReferenceSpace.getOffsetReferenceSpace() expects a raw 4x4 matrix instead of
    // the XRRigidTransform the WebXR spec (and Quest Browser) uses, so offsets were silently ignored.
    // Convert here - test-only shim, the viewer itself follows the spec.
    const RefSpace = window.XRReferenceSpace;
    const original = RefSpace.prototype.getOffsetReferenceSpace;
    RefSpace.prototype.getOffsetReferenceSpace = function getOffsetReferenceSpace(transform) {
      return original.call(this, transform && transform.matrix ? transform.matrix : transform);
    };
  });
}

/** Rotation quaternion [x, y, z, w] pitching a controller down by ``deg`` degrees. */
export function pitchDown(deg) {
  const a = (-deg * Math.PI) / 180 / 2;
  return [Math.sin(a), 0, 0, Math.cos(a)];
}

/** Press and release an emulated controller button (IWER applies values on the next frame). */
export async function pressButton(page, hand, button, holdMs = 250) {
  await page.evaluate(([h, b]) => window.__iwer.controllers[h].updateButtonValue(b, 1), [hand, button]);
  await page.waitForTimeout(holdMs);
  await page.evaluate(([h, b]) => window.__iwer.controllers[h].updateButtonValue(b, 0), [hand, button]);
  await page.waitForTimeout(200);
}

export async function enterXR(page, which) {
  const button = page.locator(which === 'ar' ? '#enter-ar' : '#enter-vr');
  await expect(button).toBeVisible();
  await button.click();
  await page.waitForFunction(
    (mode) => window.__vrphoto.state.presenting && window.__vrphoto.state.sessionMode === mode && window.__vrphoto.state.xrFrames > 20,
    which === 'ar' ? 'immersive-ar' : 'immersive-vr',
    { timeout: 30_000 },
  );
}
