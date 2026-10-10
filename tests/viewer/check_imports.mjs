#!/usr/bin/env node
// Static checks for the offline WebXR viewer (no browser needed):
//  * every module specifier in every .html / .js file under the viewer resolves to an existing
//    local file (import maps, static imports, re-exports and literal dynamic import() strings),
//  * no runtime file reachable from the pages references a CDN (the viewer must work offline),
//  * every <script src>, stylesheet and icon referenced by the pages exists.
// Static imports are read with V8's own module parser (vm.SourceTextModule), so comments and
// strings can never produce false positives.
//
// Usage: node check_imports.mjs [VIEWER_DIR] [--json]
// Exit code 0 = all good, 1 = problems found (listed on stderr), 2 = usage / internal error.

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const __filename = fileURLToPath(import.meta.url);

if (typeof vm.SourceTextModule !== 'function') {
  // Re-run with the flag that exposes V8's module parser.
  const r = spawnSync(process.execPath, ['--experimental-vm-modules', '--no-warnings', __filename, ...process.argv.slice(2)], {
    stdio: 'inherit',
  });
  process.exit(r.status ?? 2);
}

const args = process.argv.slice(2);
const asJson = args.includes('--json');
const positional = args.filter((a) => !a.startsWith('--'));
const ROOT = path.resolve(positional[0] || path.join(path.dirname(__filename), '..', '..', 'src', 'vrphoto', 'webxr', 'viewer'));

/** Hosts that must never be contacted at runtime (CDNs / remote asset stores). */
export const CDN_HOSTS = [
  'cdn.jsdelivr.net',
  'unpkg.com',
  'www.unpkg.com',
  'cdnjs.cloudflare.com',
  'esm.sh',
  'esm.run',
  'cdn.skypack.dev',
  'ga.jspm.io',
  'jspm.dev',
  'sparkjs.dev',
  'storage.googleapis.com',
  'fonts.googleapis.com',
  'fonts.gstatic.com',
  'www.gstatic.com',
  'raw.githubusercontent.com',
  'huggingface.co',
  'threejs.org',
  'cdn.babylonjs.com',
  'immersive-web.github.io',
  'meta-quest.github.io',
];
/** URLs our own (non-vendored) files may contain: XML namespaces and localhost (USB-mode help text). */
const OWN_URL_ALLOW = [/^https?:\/\/www\.w3\.org\//, /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?(\/|$)/];

const problems = [];
const problem = (file, msg) => problems.push(`${path.relative(ROOT, file) || file}: ${msg}`);

function walk(dir, out = []) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    if (ent.name === 'node_modules' || ent.name.startsWith('.')) continue;
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) walk(p, out);
    else out.push(p);
  }
  return out;
}

const isVendor = (file) => path.relative(ROOT, file).split(path.sep)[0] === 'vendor';
const exists = (file) => {
  try {
    return fs.statSync(file).isFile();
  } catch {
    return false;
  }
};

// ------------------------------------------------------------------ HTML parsing
function attr(tag, name) {
  const m = tag.match(new RegExp(`\\b${name}\\s*=\\s*("([^"]*)"|'([^']*)'|([^\\s>]+))`, 'i'));
  return m ? (m[2] ?? m[3] ?? m[4]) : null;
}

function parseHtml(file) {
  const html = fs.readFileSync(file, 'utf8');
  const res = { importMap: { imports: {} }, moduleSrcs: [], inlineModules: [], classicSrcs: [], links: [] };
  const scriptRe = /<script\b([^>]*)>([\s\S]*?)<\/script>/gi;
  let m;
  while ((m = scriptRe.exec(html))) {
    const tag = `<script ${m[1]}>`;
    const type = (attr(tag, 'type') || '').toLowerCase();
    const src = attr(tag, 'src');
    if (type === 'importmap') {
      try {
        const map = JSON.parse(m[2]);
        Object.assign(res.importMap.imports, map.imports || {});
      } catch (err) {
        problem(file, `invalid import map JSON: ${err.message}`);
      }
    } else if (type === 'module') {
      if (src) res.moduleSrcs.push(src);
      else if (m[2].trim()) res.inlineModules.push(m[2]);
    } else if (src) {
      res.classicSrcs.push(src);
    }
  }
  const linkRe = /<link\b[^>]*>/gi;
  while ((m = linkRe.exec(html))) {
    const href = attr(m[0], 'href');
    if (href) res.links.push(href);
  }
  return res;
}

// ------------------------------------------------------------------ resolution
function resolveUrlPath(spec, fromFile, importMap, kind) {
  // Returns an absolute file path or null (and records a problem).
  if (/^[a-z][a-z0-9+.-]*:/i.test(spec)) {
    if (spec.startsWith('data:') || spec.startsWith('blob:')) return 'virtual';
    problem(fromFile, `${kind} "${spec}" is a remote URL - the viewer must work offline`);
    return null;
  }
  let target = spec;
  if (!spec.startsWith('./') && !spec.startsWith('../') && !spec.startsWith('/')) {
    // Bare specifier: exact import-map entry or longest matching prefix entry ("three/addons/").
    const imports = importMap?.imports || {};
    if (Object.prototype.hasOwnProperty.call(imports, spec)) {
      target = imports[spec];
    } else {
      const prefix = Object.keys(imports)
        .filter((k) => k.endsWith('/') && spec.startsWith(k))
        .sort((a, b) => b.length - a.length)[0];
      if (!prefix) {
        problem(fromFile, `${kind} "${spec}" is a bare specifier with no import-map entry`);
        return null;
      }
      target = imports[prefix] + spec.slice(prefix.length);
    }
    // Import-map targets are relative to the HTML page (the viewer root).
    return checkPath(resolveFrom(target, path.join(ROOT, 'index.html')), fromFile, spec, kind);
  }
  return checkPath(resolveFrom(target, fromFile), fromFile, spec, kind);
}

function resolveFrom(target, fromFile) {
  const clean = target.split('#')[0].split('?')[0];
  if (clean.startsWith('/')) return path.join(ROOT, decodeURIComponent(clean));
  return path.resolve(path.dirname(fromFile), decodeURIComponent(clean));
}

function checkPath(abs, fromFile, spec, kind) {
  const rel = path.relative(ROOT, abs);
  if (rel.startsWith('..') || path.isAbsolute(rel)) {
    problem(fromFile, `${kind} "${spec}" points outside the viewer folder`);
    return null;
  }
  if (!exists(abs)) {
    problem(fromFile, `${kind} "${spec}" -> ${rel.split(path.sep).join('/')} does not exist`);
    return null;
  }
  return abs;
}

const parsedCache = new Map();
function moduleSpecifiers(file, source = null) {
  if (source === null && parsedCache.has(file)) return parsedCache.get(file);
  const code = source ?? fs.readFileSync(file, 'utf8');
  const out = { static: [], dynamic: [], unverifiable: 0 };
  try {
    const mod = new vm.SourceTextModule(code, { identifier: file });
    const reqs = mod.moduleRequests || mod.dependencySpecifiers.map((s) => ({ specifier: s }));
    out.static = reqs.map((r) => r.specifier);
  } catch (err) {
    problem(file, `cannot be parsed as an ES module: ${err.message}`);
  }
  const dyn = findDynamicImports(code);
  out.dynamic = dyn.literals;
  out.unverifiable = dyn.computed;
  if (source === null) parsedCache.set(file, out);
  return out;
}

/**
 * Find ``import(...)`` calls outside comments and strings with a small JS lexer.
 * Returns literal specifiers and the number of calls whose argument is computed.
 */
export function findDynamicImports(code) {
  const literals = [];
  let computed = 0;
  const n = code.length;
  let i = 0;
  let prev = ''; // previous significant character (for regex-vs-division)
  let prevWord = '';
  const regexAfter = '(,=:[!&|?{};+-*%<>~^';
  const regexWords = new Set(['return', 'typeof', 'case', 'do', 'else', 'in', 'of', 'new', 'delete', 'void', 'throw', 'yield', 'await']);
  const skipString = (q) => {
    i++;
    while (i < n && code[i] !== q) {
      if (code[i] === '\\') i++;
      else if (q === '`' && code[i] === '$' && code[i + 1] === '{') {
        // template expression: skip balanced braces (strings inside handled recursively)
        i += 2;
        let depth = 1;
        while (i < n) {
          const c = code[i];
          if (c === "'" || c === '"' || c === '`') {
            skipString(c); // leaves i just after the closing quote
            continue;
          }
          if (c === '{') depth++;
          else if (c === '}' && --depth === 0) break; // i stays on the closing brace
          i++;
        }
      }
      i++;
    }
    i++;
  };
  while (i < n) {
    const c = code[i];
    if (c === '/' && code[i + 1] === '/') {
      while (i < n && code[i] !== '\n') i++;
      continue;
    }
    if (c === '/' && code[i + 1] === '*') {
      const end = code.indexOf('*/', i + 2);
      i = end < 0 ? n : end + 2;
      continue;
    }
    if (c === "'" || c === '"' || c === '`') {
      skipString(c);
      prev = 'a';
      prevWord = '';
      continue;
    }
    if (c === '/' && (prev === '' || regexAfter.includes(prev) || regexWords.has(prevWord))) {
      // regular expression literal
      i++;
      let inClass = false;
      while (i < n && (code[i] !== '/' || inClass) && code[i] !== '\n') {
        if (code[i] === '\\') i++;
        else if (code[i] === '[') inClass = true;
        else if (code[i] === ']') inClass = false;
        i++;
      }
      i++;
      while (i < n && /[a-z]/i.test(code[i])) i++;
      prev = 'a';
      prevWord = '';
      continue;
    }
    if (/[A-Za-z_$]/.test(c)) {
      let j = i;
      while (j < n && /[\w$]/.test(code[j])) j++;
      const word = code.slice(i, j);
      if (word === 'import' && prev !== '.') {
        let k = j;
        while (k < n && /\s/.test(code[k])) k++;
        if (code[k] === '(') {
          k++;
          while (k < n && /\s/.test(code[k])) k++;
          const q = code[k];
          const m = (q === "'" || q === '"') ? code.slice(k).match(/^(['"])([^'"\n\\]+)\1\s*[),]/) : null;
          if (m) literals.push(m[2]);
          else computed++;
        }
      }
      prev = 'a';
      prevWord = word;
      i = j;
      continue;
    }
    if (!/\s/.test(c)) {
      prev = c;
      prevWord = '';
    }
    i++;
  }
  return { literals, computed };
}

function scanUrls(file, own) {
  const text = fs.readFileSync(file, 'utf8');
  for (const m of text.matchAll(/https?:\/\/[^\s"'`)<>\\]+/g)) {
    const url = m[0];
    let host = '';
    try {
      host = new URL(url).host.toLowerCase();
    } catch {
      continue;
    }
    if (CDN_HOSTS.includes(host)) problem(file, `references CDN URL ${url}`);
    else if (own && !OWN_URL_ALLOW.some((re) => re.test(url))) problem(file, `contains a remote URL ${url}`);
  }
}

// ------------------------------------------------------------------ main
function main() {
  if (!fs.existsSync(ROOT)) {
    console.error(`Viewer folder not found: ${ROOT}`);
    process.exit(2);
  }
  const files = walk(ROOT);
  const htmlFiles = files.filter((f) => f.endsWith('.html'));
  const jsFiles = files.filter((f) => f.endsWith('.js') || f.endsWith('.mjs'));
  const unionMap = { imports: {} };
  const reachable = new Set();

  function visit(file, importMap) {
    const key = `${file}`;
    if (reachable.has(key)) return;
    reachable.add(key);
    const specs = moduleSpecifiers(file);
    for (const spec of [...specs.static, ...specs.dynamic]) {
      const target = resolveUrlPath(spec, file, importMap, 'import');
      if (target && target !== 'virtual') visit(target, importMap);
    }
    if (specs.unverifiable && !isVendor(file)) {
      problem(file, `has ${specs.unverifiable} dynamic import() with a computed specifier (cannot be verified)`);
    }
  }

  for (const html of htmlFiles) {
    const page = parseHtml(html);
    Object.assign(unionMap.imports, page.importMap.imports);
    for (const [key, target] of Object.entries(page.importMap.imports)) {
      if (/^[a-z]+:/i.test(target)) {
        problem(html, `import map entry "${key}" points to remote URL ${target}`);
        continue;
      }
      const abs = resolveFrom(target, html);
      if (key.endsWith('/')) {
        if (!fs.existsSync(abs) || !fs.statSync(abs).isDirectory()) problem(html, `import map folder "${key}" -> ${target} does not exist`);
      } else {
        checkPath(abs, html, key, 'import map entry');
      }
    }
    for (const src of page.classicSrcs) resolveUrlPath(src.startsWith('.') || src.startsWith('/') ? src : `./${src}`, html, null, 'script');
    for (const href of page.links) {
      if (href.startsWith('data:')) continue;
      resolveUrlPath(href.startsWith('.') || href.startsWith('/') ? href : `./${href}`, html, null, 'link');
    }
    for (const src of page.moduleSrcs) {
      const abs = resolveUrlPath(src.startsWith('.') || src.startsWith('/') ? src : `./${src}`, html, page.importMap, 'module script');
      if (abs) visit(abs, page.importMap);
    }
    page.inlineModules.forEach((code, i) => {
      const specs = moduleSpecifiers(`${html}#inline${i}`, code);
      for (const spec of [...specs.static, ...specs.dynamic]) {
        const target = resolveUrlPath(spec, html, page.importMap, 'import');
        if (target && target !== 'virtual') visit(target, page.importMap);
      }
    });
    scanUrls(html, true);
  }

  // Every .js file, even if no page loads it, must parse and have resolvable imports.
  for (const js of jsFiles) {
    if (!reachable.has(js)) {
      const specs = moduleSpecifiers(js);
      for (const spec of [...specs.static, ...specs.dynamic]) resolveUrlPath(spec, js, unionMap, 'import');
    }
  }

  // No CDN at runtime: scan every reachable module (own and vendored) and our CSS.
  for (const file of reachable) scanUrls(file, !isVendor(file));
  for (const css of files.filter((f) => f.endsWith('.css') && !isVendor(f))) scanUrls(css, true);

  const summary = {
    root: ROOT,
    html: htmlFiles.length,
    js: jsFiles.length,
    reachable: [...reachable].map((f) => path.relative(ROOT, f).split(path.sep).join('/')).sort(),
    problems,
  };
  if (asJson) process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
  else {
    console.log(`Checked ${htmlFiles.length} HTML and ${jsFiles.length} JS files; ${reachable.size} modules reachable from the pages.`);
  }
  if (problems.length) {
    console.error(`${problems.length} problem(s):\n  ${problems.join('\n  ')}`);
    process.exit(1);
  }
  if (!asJson) console.log('All module specifiers resolve to local files; no CDN URLs at runtime.');
}

main();
