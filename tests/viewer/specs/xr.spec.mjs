// Immersive-session tests with IWER (Meta's Immersive Web Emulation Runtime) emulating a
// Quest 3.  IWER is injected by the test only - it is never shipped with the viewer.

import { expect, test } from '@playwright/test';
import { enterXR, installQuest3, pitchDown, pressButton, sceneList, state, waitForPixels, waitReady, watchConsole } from './helpers.mjs';

test.describe('WebXR on an emulated Quest 3', () => {
  test.beforeEach(async ({ page }) => {
    await installQuest3(page);
  });

  for (const id of ['fx-photo3d', 'fx-stereo', 'fx-sphere180', 'fx-splat']) {
    test(`Enter VR renders frames: ${id}`, async ({ page }) => {
      const log = watchConsole(page);
      await page.goto(`/view.html?scene=${id}`);
      await waitReady(page);
      await expect(page.locator('#enter-ar')).toBeVisible({ visible: id !== 'fx-sphere180' });
      await enterXR(page, 'vr');
      const st = await state(page);
      expect(st.presenting).toBe(true);
      await waitForPixels(page, 0.02);
      const frames = st.xrFrames;
      await expect.poll(async () => (await state(page)).xrFrames).toBeGreaterThan(frames + 10);
      // Exit returns to the desktop preview.
      await page.evaluate(() => window.__vrphoto.app.xr.exit());
      await expect.poll(async () => (await state(page)).presenting).toBe(false);
      await expect(page.locator('#enter-vr')).toBeVisible();
      expect(log.errors).toEqual([]);
    });
  }

  test('photo3d recenters the head onto the photo viewpoint', async ({ page }) => {
    await page.goto('/view.html?scene=fx-photo3d');
    await waitReady(page);
    await page.evaluate(() => {
      window.__iwer.position.set(0.7, 1.75, 0.4); // user standing somewhere else, taller than eye_height
    });
    await enterXR(page, 'vr');
    await expect.poll(() => page.evaluate(() => window.__vrphoto.pendingRecenter)).toBe(false);
    await page.waitForTimeout(300);
    const head = await page.evaluate(() => window.__vrphoto.app.xr.head.position.toArray());
    expect(head[0]).toBeCloseTo(0, 1);
    expect(head[1]).toBeCloseTo(1.5, 1); // eye_height_m of the fixture
    expect(head[2]).toBeCloseTo(0, 1);
  });

  test('menu opens with B, lists actions and switches scenes inside VR', async ({ page }) => {
    await page.goto('/view.html?scene=fx-stereo');
    await waitReady(page);
    await enterXR(page, 'vr');
    await pressButton(page, 'right', 'b-button');
    await expect.poll(async () => (await state(page)).menuOpen).toBe(true);
    await expect.poll(() => page.evaluate(() => window.__vrphoto.menuItems())).toEqual(
      expect.arrayContaining(['Recenter', '‹ Previous scene', 'Next scene ›', 'Exit VR']),
    );
    await page.evaluate(() => window.__vrphoto.activateMenu('next'));
    await expect.poll(async () => (await state(page)).sceneId).not.toBe('fx-stereo');
    await expect.poll(async () => (await state(page)).ready).toBe(true);
    expect((await state(page)).presenting).toBe(true);
    await waitForPixels(page, 0.01);
    await page.evaluate(() => window.__vrphoto.activateMenu('exit'));
    await expect.poll(async () => (await state(page)).presenting).toBe(false);
  });

  test('snap turn and smooth locomotion move the rig in VR', async ({ page }) => {
    await page.goto('/view.html?scene=fx-splat');
    await waitReady(page);
    await enterXR(page, 'vr');
    const yaw0 = await page.evaluate(() => window.__vrphoto.app.xr.rigYaw);
    await page.evaluate(() => window.__iwer.controllers.right.updateAxes('thumbstick', 1, 0));
    await page.waitForTimeout(300);
    await page.evaluate(() => window.__iwer.controllers.right.updateAxes('thumbstick', 0, 0));
    await expect.poll(() => page.evaluate(() => window.__vrphoto.app.xr.rigYaw)).toBeCloseTo(yaw0 - Math.PI / 6, 3);
    const p0 = await page.evaluate(() => window.__vrphoto.app.xr.head.position.toArray());
    await page.evaluate(() => window.__iwer.controllers.left.updateAxes('thumbstick', 0, -1));
    await page.waitForTimeout(600);
    await page.evaluate(() => window.__iwer.controllers.left.updateAxes('thumbstick', 0, 0));
    const p1 = await page.evaluate(() => window.__vrphoto.app.xr.head.position.toArray());
    expect(Math.hypot(p1[0] - p0[0], p1[2] - p0[2])).toBeGreaterThan(0.2);
  });

  test('right trigger arc-teleports to the floor in VR', async ({ page }) => {
    await page.goto('/view.html?scene=fx-object-glb');
    await waitReady(page);
    await enterXR(page, 'vr');
    const before = await page.evaluate(() => window.__vrphoto.app.xr.head.position.toArray());
    await page.evaluate((q) => window.__iwer.controllers.right.quaternion.set(...q), pitchDown(20));
    await page.evaluate(() => window.__iwer.controllers.right.updateButtonValue('trigger', 1));
    await expect.poll(() => page.evaluate(() => window.__vrphoto.app.xr.teleport.valid)).toBe(true);
    await page.evaluate(() => window.__iwer.controllers.right.updateButtonValue('trigger', 0));
    await expect
      .poll(async () => {
        const p = await page.evaluate(() => window.__vrphoto.app.xr.head.position.toArray());
        return Math.hypot(p[0] - before[0], p[2] - before[2]);
      })
      .toBeGreaterThan(0.5);
  });

  test('Enter AR exposes hit-test, select places the object, grip moves and both grips scale it', async ({ page }) => {
    const log = watchConsole(page);
    await page.goto('/view.html?scene=fx-object-glb');
    await waitReady(page);
    await expect(page.locator('#enter-ar')).toBeVisible();
    await enterXR(page, 'ar');
    const features = await page.evaluate(() => window.__vrphoto.app.xr.session.enabledFeatures);
    expect(features).toEqual(expect.arrayContaining(['local-floor', 'hit-test', 'anchors']));
    // Passthrough: transparent background, environment hidden, object waits for placement.
    expect(await page.evaluate(() => window.__vrphoto.app.scene.background)).toBeNull();
    let h = (await state(page)).handle;
    expect(h.placed).toBe(false);
    await expect.poll(async () => (await state(page)).handle.hitTest).toBe(true);

    // Point the right controller at the floor: the reticle appears there.
    await page.evaluate((q) => window.__iwer.controllers.right.quaternion.set(...q), pitchDown(45));
    await expect.poll(async () => (await state(page)).handle.reticle).toBe(true);
    await pressButton(page, 'right', 'trigger');
    await expect.poll(async () => (await state(page)).handle.placed).toBe(true);
    h = (await state(page)).handle;
    expect(h.placedBy).toBe('hit-test');
    expect(h.visible).toBe(true);
    expect(Math.abs(h.position[1])).toBeLessThan(0.01); // on the floor
    expect(h.position[2]).toBeLessThan(-0.3); // in front of the user
    await expect.poll(async () => (await state(page)).handle.anchored).toBe(true);
    await waitForPixels(page, 0.01);

    // Grip-grab and drag sideways.
    const x0 = h.position[0];
    await page.evaluate(() => window.__iwer.controllers.right.updateButtonValue('squeeze', 1));
    await expect.poll(async () => (await state(page)).handle.grabs).toBe(1);
    await page.evaluate(() => {
      const c = window.__iwer.controllers.right;
      c.position.set(c.position.x + 0.3, c.position.y, c.position.z);
    });
    await expect.poll(async () => (await state(page)).handle.position[0]).toBeGreaterThan(x0 + 0.25);
    // Second grip: pull the hands apart to scale up.
    await page.evaluate(() => window.__iwer.controllers.left.updateButtonValue('squeeze', 1));
    await expect.poll(async () => (await state(page)).handle.grabs).toBe(2);
    await page.evaluate(() => {
      const c = window.__iwer.controllers.left;
      c.position.set(c.position.x - 0.4, c.position.y, c.position.z);
    });
    await expect.poll(async () => (await state(page)).handle.scale).toBeGreaterThan(1.2);
    await page.evaluate(() => {
      window.__iwer.controllers.left.updateButtonValue('squeeze', 0);
      window.__iwer.controllers.right.updateButtonValue('squeeze', 0);
    });
    await expect.poll(async () => (await state(page)).handle.grabs).toBe(0);
    await page.evaluate(() => window.__vrphoto.activateMenu('reset-scale'));
    await expect.poll(async () => (await state(page)).handle.scale).toBe(1);

    // Passthrough toggle shows the neutral VR room and back.
    await page.evaluate(() => window.__vrphoto.activateMenu('passthrough'));
    expect(await page.evaluate(() => window.__vrphoto.app.scene.background !== null)).toBe(true);
    await page.evaluate(() => window.__vrphoto.activateMenu('passthrough'));
    expect(await page.evaluate(() => window.__vrphoto.app.scene.background)).toBeNull();
    expect(log.errors).toEqual([]);
  });

  test('a failed XR start shows a message that can be dismissed', async ({ page }) => {
    await page.goto('/view.html?scene=fx-photo3d');
    await waitReady(page);
    await page.evaluate(() => {
      navigator.xr.requestSession = () => Promise.reject(new DOMException('The user denied permission', 'NotAllowedError'));
    });
    await page.locator('#enter-vr').click();
    await expect(page.locator('#error')).toBeVisible();
    await expect(page.locator('#error-title')).toHaveText('Could not start VR');
    await expect(page.locator('#error-text')).toContainText('permission');
    const ok = page.locator('#error-retry');
    await expect(ok).toHaveText('OK');
    await ok.click();
    await expect(page.locator('#error')).toBeHidden();
    await expect(page.locator('#enter-vr')).toBeEnabled();
    expect((await state(page)).presenting).toBe(false);
    expect((await state(page)).ready).toBe(true); // the desktop preview is still there
  });

  test('Enter VR while the scene is still loading explains why nothing happens', async ({ page }) => {
    let release;
    const held = new Promise((resolve) => {
      release = resolve;
    });
    await page.route('**/scenes/fx-photo3d/bg.glb', async (route) => {
      await held;
      await route.continue();
    });
    await page.goto('/view.html?scene=fx-photo3d');
    await expect(page.locator('#enter-vr')).toBeVisible();
    await page.locator('#enter-vr').click();
    await expect(page.locator('#toast')).toContainText('still loading');
    expect((await state(page)).presenting).toBe(false);
    release();
    await waitReady(page);
    await enterXR(page, 'vr');
  });

  test('Enter VR on a video scene turns the sound on (the click is the user gesture)', async ({ page, request }) => {
    const scenes = await sceneList(request);
    test.skip(!scenes.some((s) => s.id === 'fx-stereo-video'), 'ffmpeg with libvpx not available for the video fixture');
    await page.goto('/view.html?scene=fx-stereo-video');
    await waitReady(page);
    expect((await state(page)).handle.video.muted).toBe(true);
    await enterXR(page, 'vr');
    expect((await state(page)).handle.video.muted).toBe(false);
  });

  test('switching to a VR-only scene inside AR turns passthrough off, and back on afterwards', async ({ page }) => {
    const log = watchConsole(page);
    await page.goto('/view.html?scene=fx-object-glb');
    await waitReady(page);
    await enterXR(page, 'ar');
    expect(await page.evaluate(() => window.__vrphoto.app.scene.background)).toBeNull();
    await page.evaluate(() => window.__vrphoto.loadScene('fx-sphere180'));
    expect((await state(page)).sceneId).toBe('fx-sphere180');
    expect(await page.evaluate(() => window.__vrphoto.app.passthrough)).toBe(false);
    expect(await page.evaluate(() => window.__vrphoto.app.scene.background !== null)).toBe(true);
    expect(await page.evaluate(() => window.__vrphoto.menuItems())).toContain('Passthrough: off');
    await page.evaluate(() => window.__vrphoto.loadScene('fx-object-glb'));
    expect((await state(page)).sceneId).toBe('fx-object-glb');
    // The object's placement instructions are what the user sees (not just the scene title).
    expect((await state(page)).vrToast).toContain('Point at the floor');
    expect(await page.evaluate(() => window.__vrphoto.app.passthrough)).toBe(true);
    expect(await page.evaluate(() => window.__vrphoto.app.scene.background)).toBeNull();
    // A choice made in the menu is kept: off stays off for the next AR scene.
    await page.evaluate(() => window.__vrphoto.activateMenu('passthrough'));
    await page.evaluate(() => window.__vrphoto.loadScene('fx-object-splat'));
    expect(await page.evaluate(() => window.__vrphoto.app.passthrough)).toBe(false);
    expect(log.errors).toEqual([]);
  });

  test('an object made by the app (billboard GLB) stands on the floor at its real height in AR', async ({ page, request }) => {
    const scenes = await sceneList(request);
    test.skip(!scenes.some((s) => s.id === 'fx-real-object'), 'the app geometry modules are not available');
    const log = watchConsole(page);
    await page.goto('/view.html?scene=fx-real-object');
    await waitReady(page);
    await enterXR(page, 'ar');
    await page.evaluate((q) => window.__iwer.controllers.right.quaternion.set(...q), pitchDown(45));
    await expect.poll(async () => (await state(page)).handle.reticle).toBe(true);
    await pressButton(page, 'right', 'trigger');
    await expect.poll(async () => (await state(page)).handle.placed).toBe(true);
    const h = (await state(page)).handle;
    expect(h.realHeight).toBeCloseTo(1.7, 2);
    expect(Math.abs(h.position[1])).toBeLessThan(0.01);
    await waitForPixels(page, 0.01);
    expect(log.errors).toEqual([]);
  });

  test('splat object can be placed in AR too', async ({ page }) => {
    await page.goto('/view.html?scene=fx-object-splat');
    await waitReady(page);
    await enterXR(page, 'ar');
    await page.evaluate((q) => window.__iwer.controllers.right.quaternion.set(...q), pitchDown(40));
    await expect.poll(async () => (await state(page)).handle.reticle).toBe(true);
    await pressButton(page, 'right', 'trigger');
    await expect.poll(async () => (await state(page)).handle.placed).toBe(true);
    expect((await state(page)).handle.placedBy).toBe('hit-test');
  });
});
