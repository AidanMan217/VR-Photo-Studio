// Small shared helpers: friendly errors, fetching with progress, resource disposal.

/** An error whose message is meant for the user (shown as-is in the UI). */
export class FriendlyError extends Error {
  constructor(title, message, { details = '', cause } = {}) {
    super(message, cause ? { cause } : undefined);
    this.name = 'FriendlyError';
    this.title = title;
    this.details = details;
  }
}

export class AbortedError extends Error {
  constructor() {
    super('Loading was cancelled');
    this.name = 'AbortError';
  }
}

export function throwIfAborted(signal) {
  if (signal?.aborted) throw new AbortedError();
}

export function isAbort(err) {
  return err?.name === 'AbortError';
}

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Prefix log lines so they are easy to find in the Quest Browser remote console. */
export const log = {
  info: (...a) => console.info('[vrphoto]', ...a),
  warn: (...a) => console.warn('[vrphoto]', ...a),
  error: (...a) => console.error('[vrphoto]', ...a),
};

/** True on a Meta Quest 3 / 3S (used for render-scale choices). */
export function isQuest3() {
  return /Quest 3/i.test(navigator.userAgent);
}

function httpError(url, res) {
  const name = decodeURIComponent(String(url).split('?')[0].split('/').pop() || url);
  if (res.status === 404) {
    return new FriendlyError(
      'A file is missing',
      `This scene needs "${name}", but it is not on the PC any more. Export the scene again from the desktop app.`,
      { details: `HTTP 404 for ${url}` },
    );
  }
  return new FriendlyError(
    'Could not load the scene',
    `The PC answered with an error (HTTP ${res.status}) while loading "${name}". Try again, or restart the viewer in the desktop app.`,
    { details: `HTTP ${res.status} ${res.statusText} for ${url}` },
  );
}

function networkError(url, err) {
  return new FriendlyError(
    'Lost connection to your PC',
    'Could not reach VR Photo Studio. Make sure the app is still running on your PC, the VR viewer is started, and this headset is on the same network.',
    { details: `${err?.message || err} (${url})`, cause: err },
  );
}

/** fetch() that turns HTTP/network failures into FriendlyErrors. */
export async function fetchChecked(url, { signal, cache = 'no-cache' } = {}) {
  let res;
  try {
    res = await fetch(url, { signal, cache });
  } catch (err) {
    if (isAbort(err)) throw new AbortedError();
    throw networkError(url, err);
  }
  if (!res.ok) throw httpError(url, res);
  return res;
}

export async function fetchJSON(url, opts = {}) {
  const res = await fetchChecked(url, opts);
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch (err) {
    throw new FriendlyError(
      'The scene file is damaged',
      'scene.json could not be read (it is not valid JSON). Export the scene again from the desktop app.',
      { details: String(err?.message || err), cause: err },
    );
  }
}

/**
 * Download ``url`` into an ArrayBuffer, reporting progress as a 0..1 fraction (or -1 when the
 * size is unknown).
 */
export async function fetchArrayBuffer(url, { signal, onProgress } = {}) {
  const res = await fetchChecked(url, { signal });
  const total = Number(res.headers.get('Content-Length')) || 0;
  if (!res.body || !onProgress) return res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      onProgress(total ? received / total : -1);
    }
  } catch (err) {
    if (isAbort(err) || signal?.aborted) throw new AbortedError();
    throw networkError(url, err);
  }
  const out = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.byteLength;
  }
  return out.buffer;
}

const disposed = new WeakSet();

/** Dispose every geometry / material / texture below ``root`` (safe to call twice). */
export function disposeObject3D(root) {
  if (!root) return;
  const textures = new Set();
  const objects = [];
  root.traverse((obj) => objects.push(obj));
  for (const obj of objects) {
    if (disposed.has(obj)) continue;
    disposed.add(obj);
    if (obj.geometry && typeof obj.geometry.dispose === 'function') obj.geometry.dispose();
    const mats = Array.isArray(obj.material) ? obj.material : obj.material ? [obj.material] : [];
    for (const mat of mats) {
      for (const value of Object.values(mat)) {
        if (value && value.isTexture) textures.add(value);
      }
      if (mat.uniforms) {
        for (const u of Object.values(mat.uniforms)) if (u?.value?.isTexture) textures.add(u.value);
      }
      mat.dispose?.();
    }
    // SplatMesh (Spark) and similar objects own GPU/worker resources and expose dispose().
    if (obj !== root && typeof obj.dispose === 'function' && !obj.isScene) {
      try {
        obj.dispose();
      } catch (err) {
        log.warn('dispose failed', err);
      }
    }
  }
  for (const t of textures) t.dispose();
}
