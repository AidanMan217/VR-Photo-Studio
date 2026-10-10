// Asset loading shared by the modes: GLB (unlit), stereo images split per eye, video, Spark splats.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { AbortedError, FriendlyError, fetchArrayBuffer, fetchChecked, log, throwIfAborted } from './util.js';

/** Largest texture side we upload (keeps Quest 3 at 72-90 fps and inside its memory budget). */
export const MAX_TEXTURE = 4096;

let gltfLoader = null;

/** The shared GLTFLoader (also used for controller and hand models). */
export function getGLTFLoader() {
  if (!gltfLoader) gltfLoader = new GLTFLoader();
  return gltfLoader;
}

const fileName = (url) => decodeURIComponent(String(url).split('?')[0].split('/').pop());

/** Download and parse a .glb file. */
export async function loadGLB(url, { signal, onProgress } = {}) {
  const buf = await fetchArrayBuffer(url, { signal, onProgress });
  throwIfAborted(signal);
  const base = url.substring(0, url.lastIndexOf('/') + 1);
  try {
    return await getGLTFLoader().parseAsync(buf, base);
  } catch (err) {
    const text = String(err?.message || err);
    // Draco / meshopt geometry or KTX2 textures need decoders the offline viewer does not ship.
    const compressed = /draco|meshopt|ktx2|basisu/i.test(text);
    throw new FriendlyError(
      'Could not open a 3D model',
      compressed
        ? `"${fileName(url)}" uses a compressed GLB format (Draco / meshopt / KTX2) that this viewer cannot read. Save it as a plain GLB (no compression) and export it again.`
        : `"${fileName(url)}" is not a valid 3D model (GLB). Export the scene again from the desktop app.`,
      { details: text, cause: err },
    );
  }
}

/**
 * Replace lit materials with unlit MeshBasicMaterial (photos already contain their lighting,
 * and unlit rendering is the cheapest path on Quest).  Returns the number of meshes changed.
 */
export function makeUnlit(root, { anisotropy = 1 } = {}) {
  let n = 0;
  root.traverse((obj) => {
    if (!obj.isMesh || !obj.material) return;
    const list = Array.isArray(obj.material) ? obj.material : [obj.material];
    const out = list.map((m) => {
      if (m.map) {
        m.map.colorSpace = THREE.SRGBColorSpace;
        m.map.anisotropy = Math.max(m.map.anisotropy || 1, anisotropy);
      }
      if (m.isMeshBasicMaterial) {
        m.toneMapped = false;
        return m;
      }
      const useEmissive = !m.map && m.emissiveMap;
      const basic = new THREE.MeshBasicMaterial({
        name: m.name,
        map: (useEmissive ? m.emissiveMap : m.map) || null,
        color: m.color ? m.color.clone() : new THREE.Color(0xffffff),
        alphaMap: m.alphaMap || null,
        transparent: m.transparent,
        opacity: m.opacity,
        alphaTest: m.alphaTest,
        side: m.side,
        vertexColors: m.vertexColors,
        depthWrite: m.depthWrite,
        depthTest: m.depthTest,
        toneMapped: false,
      });
      m.dispose();
      n += 1;
      return basic;
    });
    obj.material = Array.isArray(obj.material) ? out : out[0];
  });
  return n;
}

function decodeError(url, err) {
  return new FriendlyError(
    'Could not open an image',
    `"${fileName(url)}" could not be decoded by this browser. Export the scene again (JPEG or PNG).`,
    { details: String(err?.message || err), cause: err },
  );
}

/** Copy a rectangle of ``src`` into a canvas-backed texture no larger than ``maxSize``. */
export function cropToTexture(src, sx, sy, sw, sh, maxSize = MAX_TEXTURE) {
  const s = Math.min(1, maxSize / Math.max(sw, sh));
  const w = Math.max(1, Math.round(sw * s));
  const h = Math.max(1, Math.round(sh * s));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const g = canvas.getContext('2d');
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = 'high';
  g.drawImage(src, sx, sy, sw, sh, 0, 0, w, h);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  // Free the CPU copy once the GPU has it (large VR180 frames are ~60 MB each).
  tex.onUpdate = () => {
    canvas.width = 1;
    canvas.height = 1;
  };
  return tex;
}

/**
 * Load a (possibly stereo) image and return one texture per eye.
 * ``layout``: "sbs" (left | right), "tb" (left on top) or "mono".
 * Returns ``{ left, right, eyeWidth, eyeHeight, stereo }``; for mono, left === right.
 */
export async function loadEyeTextures(url, layout, { signal, onProgress, maxSize = MAX_TEXTURE } = {}) {
  const buf = await fetchArrayBuffer(url, { signal, onProgress });
  throwIfAborted(signal);
  let bmp;
  try {
    bmp = await createImageBitmap(new Blob([buf]));
  } catch (err) {
    throw decodeError(url, err);
  }
  throwIfAborted(signal);
  const W = bmp.width;
  const H = bmp.height;
  let rects;
  if (layout === 'sbs') rects = [[0, 0, W / 2, H], [W / 2, 0, W / 2, H]];
  else if (layout === 'tb') rects = [[0, 0, W, H / 2], [0, H / 2, W, H / 2]];
  else rects = [[0, 0, W, H]];
  const textures = rects.map((r) => cropToTexture(bmp, r[0], r[1], r[2], r[3], maxSize));
  bmp.close?.();
  return {
    left: textures[0],
    right: textures[1] || textures[0],
    eyeWidth: rects[0][2],
    eyeHeight: rects[0][3],
    stereo: textures.length === 2,
  };
}

/** Load a grayscale image (e.g. depth.png) into a Float32Array of size w*h in [0,1]. */
export async function loadGrayscale(url, w, h, { signal } = {}) {
  const buf = await fetchArrayBuffer(url, { signal });
  throwIfAborted(signal);
  let bmp;
  try {
    bmp = await createImageBitmap(new Blob([buf]));
  } catch (err) {
    throw decodeError(url, err);
  }
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const g = canvas.getContext('2d', { willReadFrequently: true });
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = 'high';
  g.drawImage(bmp, 0, 0, w, h);
  bmp.close?.();
  const px = g.getImageData(0, 0, w, h).data;
  const out = new Float32Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = px[i * 4] / 255;
  return out;
}

const VIDEO_UNDECODABLE = 'The browser cannot decode this video format. Export it again with the "Compatible (H.264)" video format.';

/**
 * Create a looping, initially muted video element and its texture.  Browsers (and WebXR) only
 * allow sound after a user gesture, so callers unmute on the first click / trigger press.
 * A video whose picture cannot be decoded is reported even when its sound track plays (the
 * browser then fires 'loadeddata' with a 0 x 0 picture instead of an error).
 */
export async function loadVideo(url, { signal } = {}) {
  const video = document.createElement('video');
  video.crossOrigin = 'anonymous';
  video.loop = true;
  video.muted = true;
  video.playsInline = true;
  video.setAttribute('playsinline', '');
  video.preload = 'auto';
  video.src = url;
  await new Promise((resolve, reject) => {
    const cleanup = () => {
      video.removeEventListener('loadeddata', onLoaded);
      video.removeEventListener('error', onError);
      signal?.removeEventListener('abort', onAbort);
    };
    const onLoaded = () => {
      cleanup();
      if (!video.videoWidth || !video.videoHeight) {
        // e.g. HEVC + AAC in a browser without HEVC: sound plays, the picture stays black.
        disposeVideo(video, null);
        reject(new FriendlyError('This video cannot be played here', VIDEO_UNDECODABLE, {
          details: 'The video has no picture the browser can decode (0 x 0 after loading).',
        }));
        return;
      }
      resolve();
    };
    const onError = async () => {
      cleanup();
      const code = video.error?.code;
      disposeVideo(video, null);
      // Distinguish "missing file" from "unsupported codec" for a helpful message.
      try {
        await fetchChecked(url, { signal });
      } catch (err) {
        reject(err);
        return;
      }
      reject(new FriendlyError(
        'This video cannot be played here',
        code === 4 || code === 3
          ? VIDEO_UNDECODABLE
          : 'The video could not be loaded. Try again, or export it again from the desktop app.',
        { details: `MediaError code ${code}: ${video.error?.message || ''}` },
      ));
    };
    const onAbort = () => {
      cleanup();
      disposeVideo(video, null); // stop downloading / decoding a video nobody will see
      reject(new AbortedError());
    };
    if (signal?.aborted) {
      onAbort();
      return;
    }
    video.addEventListener('loadeddata', onLoaded);
    video.addEventListener('error', onError);
    signal?.addEventListener('abort', onAbort);
  });
  video.play().catch((err) => log.info('Autoplay blocked until the user interacts:', err?.name || err));
  const texture = new THREE.VideoTexture(video);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.generateMipmaps = false;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  return { video, texture, width: video.videoWidth, height: video.videoHeight };
}

/** Stop a video and release its decoder. */
export function disposeVideo(video, texture) {
  try {
    video.pause();
    video.removeAttribute('src');
    video.load();
  } catch (err) {
    log.warn('video cleanup', err);
  }
  texture?.dispose();
}

let sparkPromise = null;

/** Lazily import Spark (2.8 MB) only for scenes that contain splats. */
export function loadSpark() {
  if (!sparkPromise) {
    sparkPromise = import('@sparkjsdev/spark').catch((err) => {
      sparkPromise = null;
      throw new FriendlyError(
        'The splat renderer failed to start',
        'Gaussian splats need WebGL2. Try the Meta Quest Browser or a current Chrome / Edge.',
        { details: String(err?.message || err), cause: err },
      );
    });
  }
  return sparkPromise;
}

/**
 * Create a SparkRenderer for ``renderer`` (one per splat scene).  The caller adds it to the
 * scene with ``ctx.addToScene`` - which refuses (and disposes it) when the load was superseded
 * meanwhile - so it is removed and disposed together with the mode.
 */
export async function createSparkRenderer(renderer) {
  const { SparkRenderer } = await loadSpark();
  // sqrt(5) std devs: visually ~identical to the default but noticeably faster in VR.
  const spark = new SparkRenderer({ renderer, maxStdDev: Math.sqrt(5) });
  spark.name = 'spark-renderer';
  return spark;
}

/** Download a splat file and build a SplatMesh (fileType: "ply" | "splat" | "spz"). */
export async function loadSplatMesh(url, fileType, { signal, onProgress } = {}) {
  const { SplatMesh } = await loadSpark();
  const bytes = await fetchArrayBuffer(url, { signal, onProgress });
  throwIfAborted(signal);
  const mesh = new SplatMesh({ fileBytes: bytes, fileType, fileName: fileName(url) });
  try {
    await mesh.initialized;
  } catch (err) {
    mesh.dispose?.();
    throw new FriendlyError(
      'Could not open the splat file',
      `"${fileName(url)}" is not a valid ${String(fileType).toUpperCase()} splat file. Export it again from the desktop app.`,
      { details: String(err?.message || err), cause: err },
    );
  }
  if (signal?.aborted) {
    mesh.dispose?.();
    throw new AbortedError();
  }
  return mesh;
}

/** Value at fraction ``q`` (0..1) of an ascending-sorted typed array. */
function quantile(sorted, q) {
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[i];
}

/**
 * Robust extent of a SplatMesh's centres in its local space: per axis the ``lo``..``hi``
 * fractions (default 0.5 % .. 99.5 %) of up to ``maxSamples`` evenly picked centres, so a few
 * stray "floater" splats (common in captures) neither inflate the size nor lift the object off
 * the floor or push the start position outside a room.  ``median`` is the median centre (the
 * app's splats are normalised to median x/z = 0, floor y = 0); ``full`` is the plain bounding
 * box of all centres (what ``height_m`` describes - the capture add-on records exactly that).
 * Empty boxes / null median when the splat has no centres.
 */
export function splatRobustBounds(mesh, { lo = 0.005, hi = 0.995, maxSamples = 200_000 } = {}) {
  let total = 0;
  try {
    total = mesh.packedSplats?.numSplats ?? mesh.splats?.numSplats ?? mesh.splats?.getNumSplats?.() ?? 0;
  } catch {
    total = 0;
  }
  const stride = total > maxSamples ? Math.ceil(total / maxSamples) : 1;
  const cap = total > 0 ? Math.ceil(total / stride) : 1024;
  let xs = new Float32Array(cap);
  let ys = new Float32Array(cap);
  let zs = new Float32Array(cap);
  let n = 0;
  const full = new THREE.Box3();
  try {
    mesh.forEachSplat((index, center) => {
      if (!Number.isFinite(center.x) || !Number.isFinite(center.y) || !Number.isFinite(center.z)) return;
      full.expandByPoint(center);
      if (index % stride !== 0) return;
      if (n === xs.length) {
        const grow = (a) => {
          const b = new Float32Array(a.length * 2);
          b.set(a);
          return b;
        };
        xs = grow(xs);
        ys = grow(ys);
        zs = grow(zs);
      }
      xs[n] = center.x;
      ys[n] = center.y;
      zs[n] = center.z;
      n += 1;
    });
  } catch (err) {
    log.warn('splat bounds', err);
  }
  if (n === 0) return { box: new THREE.Box3(), full, median: null, samples: 0 };
  const sx = xs.subarray(0, n).sort();
  const sy = ys.subarray(0, n).sort();
  const sz = zs.subarray(0, n).sort();
  const box = new THREE.Box3(
    new THREE.Vector3(quantile(sx, lo), quantile(sy, lo), quantile(sz, lo)),
    new THREE.Vector3(quantile(sx, hi), quantile(sy, hi), quantile(sz, hi)),
  );
  const median = new THREE.Vector3(quantile(sx, 0.5), quantile(sy, 0.5), quantile(sz, 0.5));
  return { box, full, median, samples: n };
}

/** Rotation that turns a splat with the given "up" axis into y-up (glTF frame). */
export function upAxisQuaternion(up) {
  const q = new THREE.Quaternion();
  if (up === '-y') q.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
  else if (up === 'z') q.setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
  return q;
}
