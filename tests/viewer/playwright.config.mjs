// Playwright configuration for the WebXR viewer browser tests.
//
// CI (GitHub Actions):  cd tests/viewer && npm ci && npx playwright install --with-deps chromium && npx playwright test
// Locally without browser downloads: `npm install --no-save @sparticuz/chromium` (bundles a
// Chromium binary) or set CHROMIUM_PATH=/path/to/chrome - see README.md.
//
// The tests start tests/viewer/serve_fixtures.py, a stand-in for the app's ViewerServer that
// serves the viewer plus a freshly generated fixture library (needs `pip install -e .`).

import { defineConfig, devices } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.VIEWER_TEST_PORT || 8765);
const PYTHON = process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3');

// Software WebGL so the tests run on GPU-less CI machines.
const GL_ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];

async function launchOptions() {
  if (process.env.CHROMIUM_PATH) return { executablePath: process.env.CHROMIUM_PATH, args: GL_ARGS };
  try {
    const { default: chromium } = await import('@sparticuz/chromium');
    const executablePath = await chromium.executablePath();
    // Drop flags meant for AWS Lambda that break multi-page test runs.
    const args = chromium.args.filter((a) => !/^--(single-process|headless|no-zygote)/.test(a));
    return { executablePath, args: [...new Set([...args, ...GL_ARGS])] };
  } catch {
    return { args: GL_ARGS }; // Playwright-managed Chromium (npx playwright install chromium)
  }
}

export default defineConfig({
  testDir: path.join(here, 'specs'),
  testMatch: /.*\.spec\.mjs$/,
  timeout: 90_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: process.env.CI ? 2 : 1,
  retries: process.env.CI ? 1 : 0,
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never', outputFolder: path.join(here, 'playwright-report') }]] : 'list',
  outputDir: path.join(here, 'test-results'),
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    viewport: { width: 960, height: 600 },
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'], viewport: { width: 960, height: 600 }, launchOptions: await launchOptions() },
    },
  ],
  webServer: {
    // VIEWER_TEST_SERVER=app runs the same tests against the real vrphoto.webxr.server.ViewerServer.
    // VIEWER_TEST_LIBRARY=<folder> serves an existing library (e.g. scenes exported by the app)
    // instead of the generated fixtures - use it with `-g "every fixture scene"`.
    command:
      `${PYTHON} serve_fixtures.py --port ${PORT}` +
      (process.env.VIEWER_TEST_SERVER === 'app' ? ' --app-server' : '') +
      (process.env.VIEWER_TEST_LIBRARY ? ` --library "${process.env.VIEWER_TEST_LIBRARY}"` : ''),
    cwd: here,
    url: `http://127.0.0.1:${PORT}/api/scenes`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
