// Gallery page: lists exported scenes from GET api/scenes and links to view.html.
// No three.js here so the page is instant even on the Quest Browser.

import { MODE_INFO, formatRelativeTime } from './manifest.js';

const API_URL = 'api/scenes';
const POLL_MS = 10000;

const $ = (id) => document.getElementById(id);

let lastPayload = '';
let pollTimer = 0;
let loadedOnce = false;

function show(id) {
  for (const key of ['loading', 'error', 'empty', 'grid']) {
    $(key).classList.toggle('hidden', key !== id);
  }
}

function cardFor(scene) {
  const info = MODE_INFO[scene.mode] || { label: scene.mode || 'Scene', icon: '\u{1F5BC}' };
  const a = document.createElement('a');
  a.className = 'card';
  a.href = scene.url || `view.html?scene=${encodeURIComponent(scene.id)}`;
  a.dataset.sceneId = scene.id;

  const thumb = document.createElement('div');
  thumb.className = 'thumb';
  if (scene.thumbnail) {
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.decoding = 'async';
    img.alt = '';
    img.src = scene.thumbnail;
    img.addEventListener('error', () => {
      img.remove();
      thumb.append(placeholder(info));
    }, { once: true });
    thumb.append(img);
  } else {
    thumb.append(placeholder(info));
  }

  const meta = document.createElement('div');
  meta.className = 'meta';
  const h2 = document.createElement('h2');
  h2.textContent = scene.title || scene.id;
  h2.title = scene.title || scene.id;
  const row = document.createElement('div');
  row.className = 'row';
  const badge = document.createElement('span');
  badge.className = `badge mode-${scene.mode}`;
  badge.textContent = info.label;
  badge.title = info.hint || '';
  const time = document.createElement('time');
  if (scene.created) {
    time.dateTime = scene.created;
    time.textContent = formatRelativeTime(scene.created);
    time.title = new Date(scene.created).toLocaleString();
  }
  row.append(badge, time);
  meta.append(h2, row);
  a.append(thumb, meta);
  return a;
}

function placeholder(info) {
  const span = document.createElement('span');
  span.className = 'placeholder';
  span.setAttribute('aria-hidden', 'true');
  span.textContent = info.icon || '\u{1F5BC}';
  return span;
}

function render(scenes) {
  const grid = $('grid');
  if (!scenes.length) {
    grid.replaceChildren();
    show('empty');
    return;
  }
  grid.replaceChildren(...scenes.map(cardFor));
  show('grid');
}

async function load({ quiet = false } = {}) {
  if (!quiet && !loadedOnce) show('loading');
  try {
    const res = await fetch(API_URL, { cache: 'no-store', headers: { Accept: 'application/json' } });
    if (!res.ok) throw new Error(`The viewer server answered with HTTP ${res.status}.`);
    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error('The viewer server sent an unexpected answer (not a scene list).');
    }
    const scenes = Array.isArray(data) ? data : (Array.isArray(data?.scenes) ? data.scenes : null);
    if (!scenes) throw new Error('The viewer server sent an unexpected answer (not a scene list).');
    loadedOnce = true;
    if (text !== lastPayload) {
      lastPayload = text;
      render(scenes.filter((s) => s && typeof s.id === 'string'));
    }
  } catch (err) {
    if (quiet && loadedOnce) return; // keep showing the last good list during a hiccup
    const offline = err instanceof TypeError; // fetch() network failure
    $('error-text').textContent = offline
      ? 'Make sure VR Photo Studio is running on your PC and the VR viewer server is started, ' +
        'and that this device is on the same network. Then press "Try again".'
      : `${err.message} Restart the viewer server in the desktop app and press "Try again".`;
    show('error');
  }
}

function schedulePoll() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    if (document.visibilityState === 'visible') await load({ quiet: true });
    schedulePoll();
  }, POLL_MS);
}

function init() {
  if (!window.isSecureContext) $('secure-banner').classList.remove('hidden');
  $('refresh').addEventListener('click', () => load());
  $('retry').addEventListener('click', () => load());
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') load({ quiet: true });
  });
  load().then(schedulePoll);
}

init();
