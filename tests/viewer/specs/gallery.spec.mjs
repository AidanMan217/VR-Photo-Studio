import { expect, test } from '@playwright/test';
import { sceneList, watchConsole } from './helpers.mjs';

test.describe('gallery (index.html)', () => {
  test('lists every exported scene with thumbnail, title and mode', async ({ page, request }) => {
    const scenes = await sceneList(request);
    expect(scenes.length).toBeGreaterThanOrEqual(7);
    const log = watchConsole(page);
    await page.goto('/');
    const cards = page.locator('.card');
    await expect(cards).toHaveCount(scenes.length);
    for (const [i, s] of scenes.entries()) {
      const card = cards.nth(i);
      await expect(card.locator('h2')).toHaveText(s.title);
      await expect(card).toHaveAttribute('href', s.url);
      await expect(card.locator('.badge')).not.toBeEmpty();
      if (s.thumbnail) await expect(card.locator('img')).toHaveAttribute('src', s.thumbnail);
    }
    // Big touch targets for the Quest Browser.
    const box = await cards.first().boundingBox();
    expect(box.height).toBeGreaterThan(150);
    await expect(page.locator('#secure-banner')).toBeHidden();
    expect(log.errors).toEqual([]);
  });

  test('a card opens the viewer for that scene', async ({ page, request }) => {
    const [first] = await sceneList(request);
    await page.goto('/');
    await page.locator('.card').first().click();
    await expect(page).toHaveURL(new RegExp(`view\\.html\\?scene=${first.id}$`));
    await expect(page.locator('#scene-title')).toHaveText(first.title);
  });

  test('empty library explains how to export from the desktop app', async ({ page }) => {
    await page.route('**/api/scenes', (route) => route.fulfill({ json: [] }));
    await page.goto('/');
    await expect(page.locator('#empty')).toBeVisible();
    await expect(page.locator('#empty')).toContainText('No scenes yet');
    await expect(page.locator('#empty')).toContainText('Add to VR viewer');
    await expect(page.locator('#grid')).toBeHidden();
  });

  test('server problems show a friendly message and Try again recovers', async ({ page }) => {
    let fail = true;
    await page.route('**/api/scenes', (route) => (fail ? route.abort() : route.continue()));
    await page.goto('/');
    await expect(page.locator('#error')).toBeVisible();
    await expect(page.locator('#error')).toContainText("Can't reach VR Photo Studio");
    fail = false;
    await page.locator('#retry').click();
    await expect(page.locator('.card').first()).toBeVisible();
    await expect(page.locator('#error')).toBeHidden();
  });

  test('insecure origin shows how to get HTTPS', async ({ page }) => {
    await page.addInitScript(() => Object.defineProperty(window, 'isSecureContext', { value: false }));
    await page.goto('/');
    await expect(page.locator('#secure-banner')).toBeVisible();
    await expect(page.locator('#secure-banner')).toContainText('https://');
    await expect(page.locator('#secure-banner')).toContainText('USB mode');
  });

  test('works at phone width', async ({ page }) => {
    await page.setViewportSize({ width: 380, height: 800 });
    await page.goto('/');
    await expect(page.locator('.card').first()).toBeVisible();
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(scrollWidth).toBeLessThanOrEqual(380);
  });
});
