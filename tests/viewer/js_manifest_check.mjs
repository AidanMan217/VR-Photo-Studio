#!/usr/bin/env node
// Runs the viewer's own (browser) manifest validator under Node so pytest can cross-check it
// against the Python validator.  Reads a JSON array of manifests on stdin and prints a JSON
// array of { ok, errors, manifest } results.  manifest.js has no browser dependencies.
//
// Usage: node js_manifest_check.mjs [VIEWER_DIR] < cases.json

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(process.argv[2] || path.join(here, '..', '..', 'src', 'vrphoto', 'webxr', 'viewer'));
const { validateManifest, formatRelativeTime, MODES, SCHEMA_ID } = await import(
  pathToFileURL(path.join(root, 'js', 'manifest.js')).href
);

const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
const results = cases.map((c) => {
  try {
    return validateManifest(c);
  } catch (err) {
    return { ok: false, errors: [`threw: ${err.message}`], manifest: null };
  }
});
process.stdout.write(
  JSON.stringify({
    modes: MODES,
    schema: SCHEMA_ID,
    relative: formatRelativeTime(new Date(Date.now() - 2 * 3600 * 1000).toISOString()),
    results,
  }),
);
