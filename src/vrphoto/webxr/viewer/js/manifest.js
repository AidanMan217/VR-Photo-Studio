// Scene manifest helpers shared by the gallery and the viewer (no three.js dependency).
// The authoritative validator is vrphoto.webxr.scene.validate_manifest (Python, run at export);
// this is a friendly, forgiving check so a damaged or hand-edited scene.json produces a clear
// message instead of a black screen.

export const SCHEMA_ID = 'vrphoto.scene/1';
export const MODES = ['photo3d', 'stereo', 'sphere', 'object', 'splat'];

/** Human labels for each mode (gallery badges, viewer title bar). */
export const MODE_INFO = {
  photo3d: { label: '3D Scene', icon: '\u{1F304}', hint: 'Step into the photo and lean to look behind things' },
  stereo: { label: '3D Screen', icon: '\u{1F3AC}', hint: 'A big 3D photo or video screen' },
  sphere: { label: 'VR 180 / 360', icon: '\u{1F310}', hint: 'Surrounds you - look around' },
  object: { label: 'Object', icon: '\u{1F9CD}', hint: 'Place it in your room and walk around it' },
  splat: { label: '3D World', icon: '\u{2728}', hint: 'A Gaussian-splat world you can walk through' },
};

/** Default values for optional fields (kept in sync with docs/CONTRACTS.md section 8). */
export const DEFAULTS = {
  photo3d: { anchor_distance_m: 2.0, eye_height_m: 1.5 },
  stereo: { kind: 'image', layout: 'sbs', aspect: null, screen_width_m: 3.0, distance_m: 4.0 },
  sphere: { kind: 'image', coverage: 180, layout: 'mono', depth: null, radius_m: 20, depth_near_m: 1.5, depth_far_m: null },
  object: { height_m: null, pivot: 'base', placement: 'hit-test', distance_m: 1.5, up: 'y' },
  splat: { up: 'y', scale: 1.0 },
};

const ENUMS = {
  'stereo.kind': ['image', 'video'],
  'stereo.layout': ['sbs', 'tb', 'mono'],
  'sphere.kind': ['image', 'video'],
  'sphere.coverage': [180, 360],
  'sphere.layout': ['mono', 'sbs', 'tb'],
  'object.format': ['glb', 'ply', 'splat', 'spz'],
  'object.pivot': ['base', 'center'],
  'object.placement': ['hit-test', 'front-floor'],
  'object.up': ['y', '-y', 'z'],
  'splat.format': ['ply', 'splat', 'spz'],
  'splat.up': ['y', '-y', 'z'],
  'env.prefer': ['ar', 'vr'],
};
const REQUIRED = {
  photo3d: ['meshes'],
  stereo: ['media'],
  sphere: ['media'],
  object: ['asset'],
  splat: ['asset'],
};
const POSITIVE = {
  photo3d: ['anchor_distance_m'],
  stereo: ['aspect', 'screen_width_m', 'distance_m'],
  sphere: ['radius_m', 'depth_near_m', 'depth_far_m'],
  object: ['height_m', 'distance_m'],
  splat: ['scale'],
};
const FILE_FIELDS = { stereo: ['media'], sphere: ['media', 'depth'], object: ['asset'], splat: ['asset'] };

const VIDEO_EXT = /\.(mp4|m4v|mov|webm|mkv)$/i;

/** True when ``name`` is a plain file name inside the scene folder (no paths, no URLs). */
export function isPlainFileName(name) {
  return typeof name === 'string' && name.length > 0 && name.length < 256 &&
    !/[\\/:]/.test(name) && name !== '.' && name !== '..' && !name.startsWith('..');
}

/**
 * Validate a parsed scene.json and return ``{ ok, errors, manifest }`` where ``manifest`` has the
 * defaults filled in.  ``errors`` are short human sentences.
 */
export function validateManifest(raw) {
  const errors = [];
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
    return { ok: false, errors: ['scene.json does not contain a scene description.'], manifest: null };
  }
  const m = structuredClone(raw);
  if (m.schema !== undefined && m.schema !== SCHEMA_ID) {
    if (typeof m.schema === 'string' && m.schema.startsWith('vrphoto.scene/')) {
      errors.push(`This scene was made by a newer VR Photo Studio (${m.schema}). Update the app on your PC.`);
    } else {
      errors.push('scene.json is not a VR Photo Studio scene.');
    }
  }
  if (!MODES.includes(m.mode)) {
    errors.push(m.mode === undefined
      ? 'The scene does not say what kind of scene it is ("mode" is missing).'
      : `Unknown scene type "${String(m.mode)}".`);
    return { ok: false, errors, manifest: m };
  }
  if (typeof m.title !== 'string' || !m.title.trim()) m.title = m.id || 'Untitled scene';
  const block = m[m.mode];
  if (block === null || typeof block !== 'object' || Array.isArray(block)) {
    errors.push(`The "${m.mode}" section is missing from scene.json.`);
    return { ok: false, errors, manifest: m };
  }
  m[m.mode] = { ...DEFAULTS[m.mode], ...block };
  const b = m[m.mode];
  for (const key of REQUIRED[m.mode]) {
    if (b[key] === undefined || b[key] === null || b[key] === '') errors.push(`${m.mode}.${key} is missing.`);
  }
  for (const key of POSITIVE[m.mode]) {
    if (b[key] === null || b[key] === undefined) continue;
    if (typeof b[key] !== 'number' || !Number.isFinite(b[key]) || b[key] <= 0) {
      errors.push(`${m.mode}.${key} must be a positive number.`);
    }
  }
  for (const [path, allowed] of Object.entries(ENUMS)) {
    const [sec, key] = path.split('.');
    if (sec !== m.mode && sec !== 'env') continue;
    const val = sec === 'env' ? m.env?.[key] : b[key];
    if (val !== undefined && val !== null && !allowed.includes(val)) {
      errors.push(`${path} must be one of: ${allowed.join(', ')}.`);
    }
  }
  if (m.mode === 'photo3d') {
    if (!Array.isArray(b.meshes) || !b.meshes.length) {
      errors.push('photo3d.meshes must list at least one .glb file.');
    } else if (!b.meshes.every(isPlainFileName)) {
      errors.push('photo3d.meshes must be plain file names inside the scene folder.');
    }
    if (typeof b.eye_height_m !== 'number' || !Number.isFinite(b.eye_height_m)) {
      errors.push('photo3d.eye_height_m must be a number.');
    }
  }
  for (const key of FILE_FIELDS[m.mode] || []) {
    if (b[key] !== null && b[key] !== undefined && !isPlainFileName(b[key])) {
      errors.push(`${m.mode}.${key} must be a plain file name inside the scene folder.`);
    }
  }
  if ((m.mode === 'stereo' || m.mode === 'sphere') && !raw[m.mode].kind && typeof b.media === 'string') {
    b.kind = VIDEO_EXT.test(b.media) ? 'video' : 'image';
  }
  if (m.mode === 'object' && !raw.object.format && typeof b.asset === 'string') {
    b.format = formatFromName(b.asset);
  }
  if (m.mode === 'splat' && !raw.splat.format && typeof b.asset === 'string') {
    b.format = formatFromName(b.asset);
  }
  if ((m.mode === 'object' || m.mode === 'splat') && !b.format) {
    errors.push(`${m.mode}.format is missing (glb, ply, splat or spz).`);
  }
  m.env = { ...(m.env && typeof m.env === 'object' ? m.env : {}) };
  if (!m.env.prefer) m.env.prefer = m.mode === 'object' ? 'ar' : 'vr';
  if (m.notices !== undefined && !Array.isArray(m.notices)) m.notices = [String(m.notices)];
  return { ok: errors.length === 0, errors, manifest: m };
}

export function formatFromName(name) {
  const ext = String(name).split('.').pop().toLowerCase();
  return ['glb', 'ply', 'splat', 'spz'].includes(ext) ? ext : null;
}

/** Background colour for VR from manifest.env.background ("black", "#rrggbb", "passthrough"). */
export function backgroundColor(env, fallback) {
  const bg = env?.background;
  if (typeof bg === 'string' && /^#[0-9a-f]{6}$/i.test(bg)) return bg;
  if (bg === 'black') return '#000000';
  return fallback;
}

/** "2 hours ago"-style text for ISO timestamps (falls back to the locale date). */
export function formatRelativeTime(iso, now = Date.now()) {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return '';
  const s = Math.round((now - t) / 1000);
  if (s < 45) return 'just now';
  const units = [['minute', 60], ['hour', 3600], ['day', 86400]];
  if (s < 7 * 86400) {
    let unit = units[0];
    for (const u of units) if (s >= u[1]) unit = u;
    const n = Math.round(s / unit[1]);
    return `${n} ${unit[0]}${n === 1 ? '' : 's'} ago`;
  }
  return new Date(t).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}
