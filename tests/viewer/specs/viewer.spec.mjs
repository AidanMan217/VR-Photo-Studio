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

  test('portrait 3D screens are at most 2.2 m tall and never wider than the scene says', async ({ page }) => {
    // F5 / F7: a 9:16 phone video on the default 3 m wide screen used to be 5.3 m tall,
    // reaching below the floor; the parallax was limited for screen_width_m, never go wider.
    const screen = async () =>
      page.evaluate(() => {
        const app = window.__vrphoto.app;
        const left = app.root.getObjectByName('screen-left');
        const group = app.root.getObjectByName('stereo-screen');
        const p = left.geometry.parameters;
        return { width: p.width, height: p.height, y: group.position.y, handle: window.__vrphoto.state.handle };
      });
    await page.goto('/view.html?scene=fx-stereo');
    await waitReady(page);
    let s = await screen();
    expect(s.width).toBeCloseTo(2.5, 5); // landscape: exactly the manifest width
    expect(s.height).toBeCloseTo(2.5 * 192 / 256, 3);
    await page.route('**/scenes/fx-stereo/scene.json', (route) =>
      route.fulfill({
        json: {
          schema: 'vrphoto.scene/1',
          id: 'fx-stereo',
          title: 'Portrait',
          mode: 'stereo',
          stereo: { media: 'photo_LR.jpg', kind: 'image', layout: 'sbs', aspect: 0.5625 },
        },
      }),
    );
    await page.reload();
    await waitReady(page);
    s = await screen();
    expect(s.height).toBeCloseTo(2.2, 5);
    expect(s.width).toBeCloseTo(2.2 * 0.5625, 5);
    expect(s.handle.width).toBeCloseTo(s.width, 5);
    expect(s.y - s.height / 2).toBeGreaterThanOrEqual(0.5 - 1e-6); // bottom edge well above the floor
  });

  test('a real-size capture keeps its size, stands on its own floor and ignores stray splats', async ({ page }) => {
    // F2: the capture is already in metres (object.scale = 1): height_m (the person, 1.7 m)
    // must not squeeze the whole splat (person + floor + floaters, 5 m tall) to 1.7 m, and
    // stray floater splats must not lift it off the floor or shift it sideways.
    await page.goto('/view.html?scene=fx-capture-object');
    await waitReady(page);
    const h = (await state(page)).handle;
    expect(h.fitScale).toBe(1);
    expect(h.realHeight).toBeCloseTo(1.7, 6); // "Reset 1:1 scale" names the person's height
    expect(Math.abs(h.pivot[1])).toBeLessThan(0.02); // its floor (y = 0) on the real floor
    expect(Math.abs(h.pivot[0])).toBeLessThan(0.05); // centred on the person, not the floaters
    expect(Math.abs(h.pivot[2])).toBeLessThan(0.05);
  });

  test('stray splats do not turn a small capture into a room you start inside', async ({ page, request }) => {
    await page.goto('/view.html?scene=fx-splat');
    await waitReady(page);
    expect((await state(page)).handle.inside).toBe(true); // the 8 m room fixture
    const capture = await (await request.get('/scenes/fx-capture-object/capture.splat')).body();
    await page.route('**/scenes/fx-splat/room.splat', (route) => route.fulfill({ body: capture }));
    await page.reload();
    await waitReady(page);
    const h = (await state(page)).handle;
    expect(h.inside).toBe(false); // 3 m person + floor disc; floaters 6 m away are ignored
    expect(h.bounds[1][0]).toBeLessThan(2); // robust x extent: the floor disc, not the floaters
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

  test('a video whose picture cannot be decoded is reported even when its sound plays', async ({ page }) => {
    // E2E-4: e.g. HEVC + AAC in a browser without HEVC fires 'loadeddata' with a 0 x 0 picture.
    // A sound-only file reproduces that deterministically in every Chromium build.
    const rate = 8000;
    const samples = rate; // 1 s mono 16-bit sine
    const wav = Buffer.alloc(44 + samples * 2);
    wav.write('RIFF', 0);
    wav.writeUInt32LE(36 + samples * 2, 4);
    wav.write('WAVEfmt ', 8);
    wav.writeUInt32LE(16, 16);
    wav.writeUInt16LE(1, 20); // PCM
    wav.writeUInt16LE(1, 22); // mono
    wav.writeUInt32LE(rate, 24);
    wav.writeUInt32LE(rate * 2, 28);
    wav.writeUInt16LE(2, 32);
    wav.writeUInt16LE(16, 34);
    wav.write('data', 36);
    wav.writeUInt32LE(samples * 2, 40);
    for (let i = 0; i < samples; i++) wav.writeInt16LE(Math.round(8000 * Math.sin((i * 2 * Math.PI * 440) / rate)), 44 + i * 2);
    await page.route('**/scenes/fx-stereo/scene.json', (route) =>
      route.fulfill({
        json: {
          schema: 'vrphoto.scene/1',
          id: 'fx-stereo',
          title: 'Sound but no picture',
          mode: 'stereo',
          stereo: { media: 'clip_LR.mp4', kind: 'video', layout: 'sbs' },
        },
      }),
    );
    await page.route('**/scenes/fx-stereo/clip_LR.mp4', (route) => route.fulfill({ body: wav, contentType: 'video/mp4' }));
    await page.goto('/view.html?scene=fx-stereo');
    await expect(page.locator('#error')).toBeVisible();
    await expect(page.locator('#error-title')).toHaveText('This video cannot be played here');
    await expect(page.locator('#error-text')).toContainText('Compatible (H.264)');
    expect((await state(page)).ready).toBe(false);
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
