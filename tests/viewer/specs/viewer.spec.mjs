import { expect, test } from '@playwright/test';
import { sceneList, state, waitForPixels, waitReady, watchConsole } from './helpers.mjs';

const MODE_LABELS = {
  photo3d: '3D Scene',
  stereo: '3D Screen',
  sphere: 'VR 180 / 360',
  object: 'Object',
  splat: '3D World',
};

test.describe('viewer desktop preview (view.html)', () => {
  test('every fixture scene loads without console errors and draws pixels', async ({ page, request }) => {
    test.setTimeout(240_000);
    const scenes = await sceneList(request);
    for (const s of scenes) {
      await test.step(s.id, async () => {
        const log = watchConsole(page);
        await page.goto(s.url);
        const st = await waitReady(page);
        expect(st.mode).toBe(s.mode);
        await expect(page.locator('#scene-title')).toHaveText(s.title);
        await expect(page.locator('#scene-mode')).toHaveText(MODE_LABELS[s.mode]);
        await expect(page.locator('#loading')).toBeHidden();
        await expect(page.locator('#error')).toBeHidden();
        const px = await waitForPixels(page, 0.05);
        expect(px.distinctColors, `${s.id} looks like a flat colour`).toBeGreaterThan(8);
        // No headset in the test browser: the preview explains how to get to VR.
        await expect(page.locator('#xr-status')).toContainText('Preview mode');
        expect(log.errors, `${s.id} console errors`).toEqual([]);
      });
    }
  });

  test('stereo preview shows the left eye; mono content is on the shared layer', async ({ page }) => {
    await page.goto('/view.html?scene=fx-stereo');
    await waitReady(page);
    const layers = await page.evaluate(() => {
      const app = window.__vrphoto.app;
      const get = (n) => app.root.getObjectByName(n);
      return {
        left: get('screen-left').layers.mask,
        right: get('screen-right').layers.mask,
        camera: app.camera.layers.mask,
      };
    });
    expect(layers.left).toBe(1 << 1);
    expect(layers.right).toBe(1 << 2);
    expect(layers.camera & layers.left).toBeTruthy();
    expect(layers.camera & layers.right).toBe(0);
  });

  test('VR180 is centred straight ahead (-z)', async ({ page }) => {
    await page.goto('/view.html?scene=fx-sphere180');
    await waitReady(page);
    // The fixture has a red marker at the image centre: it must be in the middle of the view.
    const centre = await page.evaluate(async () => {
      await window.__vrphoto.capture();
      const app = window.__vrphoto.app;
      const gl = app.renderer.getContext();
      app.renderer.render(app.scene, app.camera);
      const px = new Uint8Array(4);
      gl.readPixels(gl.drawingBufferWidth >> 1, gl.drawingBufferHeight >> 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      return Array.from(px);
    });
    expect(centre[0]).toBeGreaterThan(180);
    expect(centre[1]).toBeLessThan(90);
  });

  test('keyboard WASD moves the desktop camera', async ({ page }) => {
    await page.goto('/view.html?scene=fx-object-glb');
    await waitReady(page);
    const before = await page.evaluate(() => window.__vrphoto.app.camera.position.toArray());
    await page.locator('#stage canvas').click({ position: { x: 20, y: 300 } });
    await page.keyboard.down('w');
    await page.waitForTimeout(400);
    await page.keyboard.up('w');
    const after = await page.evaluate(() => window.__vrphoto.app.camera.position.toArray());
    expect(after[2]).toBeLessThan(before[2] - 0.1);
  });

  test('video scene autoplays muted and the media buttons work', async ({ page, request }) => {
    const scenes = await sceneList(request);
    test.skip(!scenes.some((s) => s.id === 'fx-stereo-video'), 'ffmpeg with libvpx not available for the video fixture');
    await page.goto('/view.html?scene=fx-stereo-video');
    await waitReady(page);
    await expect(page.locator('#media-bar')).toBeVisible();
    await expect.poll(async () => (await state(page)).handle.video.time).toBeGreaterThan(0);
    expect((await state(page)).handle.video.muted).toBe(true);
    await page.locator('#media-play').click();
    await expect.poll(async () => (await state(page)).handle.video.paused).toBe(true);
    await page.locator('#media-mute').click();
    await expect.poll(async () => (await state(page)).handle.video.muted).toBe(false);
  });

  test('missing scene shows a friendly error with a way back', async ({ page }) => {
    await page.goto('/view.html?scene=does-not-exist');
    await expect(page.locator('#error')).toBeVisible();
    await expect(page.locator('#error-title')).toHaveText('Scene not found');
    await expect(page.locator('#error a[href="./"]')).toBeVisible();
    expect((await state(page)).ready).toBe(false);
  });

  test('damaged scene.json is reported clearly', async ({ page }) => {
    await page.route('**/scenes/fx-stereo/scene.json', (route) => route.fulfill({ body: '{"mode": "stereo", ', contentType: 'application/json' }));
    await page.goto('/view.html?scene=fx-stereo');
    await expect(page.locator('#error-title')).toHaveText('The scene file is damaged');
  });

  test('invalid manifest is reported clearly', async ({ page }) => {
    await page.route('**/scenes/fx-stereo/scene.json', (route) =>
      route.fulfill({ json: { schema: 'vrphoto.scene/1', id: 'fx-stereo', title: 'x', mode: 'stereo' } }),
    );
    await page.goto('/view.html?scene=fx-stereo');
    await expect(page.locator('#error-title')).toHaveText('This scene cannot be shown');
    await expect(page.locator('#error-text')).toContainText('"stereo" section is missing');
  });

  test('missing asset names the file', async ({ page }) => {
    await page.route('**/scenes/fx-stereo/photo_LR.jpg', (route) => route.fulfill({ status: 404, body: 'nope' }));
    await page.goto('/view.html?scene=fx-stereo');
    await expect(page.locator('#error-title')).toHaveText('A file is missing');
    await expect(page.locator('#error-text')).toContainText('photo_LR.jpg');
    await expect(page.locator('#error-retry')).toBeVisible();
  });

  test('no scene parameter', async ({ page }) => {
    await page.goto('/view.html');
    await expect(page.locator('#error-title')).toHaveText('No scene selected');
  });

  test('a superseded splat load leaves no splat renderer behind', async ({ page }) => {
    // Hold back the (lazily imported) splat renderer so the scene switch happens mid-load.
    let release;
    const held = new Promise((resolve) => {
      release = resolve;
    });
    await page.route('**/vendor/spark/dist/spark.module.js', async (route) => {
      await held;
      await route.continue();
    });
    const log = watchConsole(page);
    await page.goto('/view.html?scene=fx-splat');
    await page.waitForFunction(() => window.__vrphoto?.app?.handle === null && window.__vrphoto.app.renderer);
    await page.waitForTimeout(300);
    await page.evaluate(() => window.__vrphoto.loadScene('fx-stereo'));
    await waitReady(page);
    expect((await state(page)).sceneId).toBe('fx-stereo');
    const sparkLoaded = page.waitForResponse('**/vendor/spark/dist/spark.module.js');
    release();
    await sparkLoaded;
    // Give the stale load time to evaluate Spark and try to add its renderer.
    await page.waitForTimeout(1500);
    const leftovers = await page.evaluate(() => {
      const app = window.__vrphoto.app;
      return {
        spark: app.scene.children.filter((o) => o.name === 'spark-renderer').length,
        extra: app.extra.length,
        mode: window.__vrphoto.state.mode,
      };
    });
    expect(leftovers).toEqual({ spark: 0, extra: 0, mode: 'stereo' });
    expect(log.errors).toEqual([]);
  });

  test('insecure origin explains HTTPS / USB mode instead of offering VR', async ({ page }) => {
    await page.addInitScript(() => Object.defineProperty(window, 'isSecureContext', { value: false }));
    await page.goto('/view.html?scene=fx-photo3d');
    await waitReady(page);
    await expect(page.locator('#xr-note')).toBeVisible();
    await expect(page.locator('#xr-note')).toContainText('https://');
    await expect(page.locator('#xr-note')).toContainText('certificate');
    await expect(page.locator('#enter-vr')).toBeHidden();
  });
});
