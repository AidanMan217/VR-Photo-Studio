// stereo: a big virtual screen showing a side-by-side / top-bottom 3D photo or video.
// Each eye sees its own half through camera layers (left eye = layer 1, right eye = layer 2);
// the desktop preview shows the left eye.  Mono media is shown on layer 0 to both eyes.

import * as THREE from 'three';
import { backgroundColor } from '../manifest.js';
import { disposeVideo, loadEyeTextures, loadVideo } from '../loaders.js';
import { videoControls } from '../media.js';

const SCREEN_Y = 1.6; // screen centre height; the head is recentered to this height

/** Plane geometry whose UVs show [u0,u1] x [v0,v1] of a texture. */
export function screenGeometry(width, height, u0 = 0, u1 = 1, v0 = 0, v1 = 1) {
  const geo = new THREE.PlaneGeometry(width, height);
  const uv = geo.attributes.uv;
  for (let i = 0; i < uv.count; i++) {
    uv.setXY(i, u0 + uv.getX(i) * (u1 - u0), v0 + uv.getY(i) * (v1 - v0));
  }
  uv.needsUpdate = true;
  return geo;
}

/** UV windows for the two eyes of a packed stereo frame (texture v = 1 is the top row). */
export function eyeWindows(layout) {
  if (layout === 'sbs') return { left: [0, 0.5, 0, 1], right: [0.5, 1, 0, 1] };
  if (layout === 'tb') return { left: [0, 1, 0.5, 1], right: [0, 1, 0, 0.5] };
  return { left: [0, 1, 0, 1], right: [0, 1, 0, 1] };
}

export async function load(ctx, scene) {
  const cfg = scene.stereo;
  const layout = cfg.layout;
  const stereo = layout === 'sbs' || layout === 'tb';
  const group = new THREE.Group();
  group.name = 'stereo-screen';
  ctx.root.add(group);

  let eyeAspect;
  let leftMat;
  let rightMat;
  let video = null;
  let videoTex = null;
  let leftGeo;
  let rightGeo;

  const width = cfg.screen_width_m;
  if (cfg.kind === 'video') {
    ctx.progress(-1, 'Opening video…');
    const v = await loadVideo(ctx.asset(cfg.media), { signal: ctx.signal });
    video = v.video;
    videoTex = v.texture;
    if (ctx.signal.aborted) disposeVideo(video, videoTex); // superseded: stop it (checkAborted throws below)
    const w = v.width || 16;
    const h = v.height || 9;
    eyeAspect = layout === 'sbs' ? w / 2 / h : layout === 'tb' ? w / (h / 2) : w / h;
    const aspect = cfg.aspect || eyeAspect;
    const win = eyeWindows(layout);
    leftGeo = screenGeometry(width, width / aspect, ...win.left);
    rightGeo = stereo ? screenGeometry(width, width / aspect, ...win.right) : null;
    leftMat = new THREE.MeshBasicMaterial({ map: videoTex, toneMapped: false });
    rightMat = leftMat;
  } else {
    const tex = await loadEyeTextures(ctx.asset(cfg.media), layout, {
      signal: ctx.signal,
      onProgress: (f) => ctx.progress(f < 0 ? -1 : f * 0.95, 'Loading 3D photo…'),
    });
    eyeAspect = tex.eyeWidth / tex.eyeHeight;
    const aspect = cfg.aspect || eyeAspect;
    leftGeo = screenGeometry(width, width / aspect);
    rightGeo = stereo ? screenGeometry(width, width / aspect) : null;
    leftMat = new THREE.MeshBasicMaterial({ map: tex.left, toneMapped: false });
    rightMat = stereo ? new THREE.MeshBasicMaterial({ map: tex.right, toneMapped: false }) : leftMat;
  }
  ctx.checkAborted();

  const aspect = cfg.aspect || eyeAspect;
  const height = width / aspect;
  const left = new THREE.Mesh(leftGeo, leftMat);
  left.name = 'screen-left';
  if (stereo) {
    left.layers.set(1);
    const right = new THREE.Mesh(rightGeo, rightMat);
    right.name = 'screen-right';
    right.layers.set(2);
    group.add(right);
  }
  group.add(left);
  // Thin dark bezel so the screen edge reads well against any background.
  const bezel = new THREE.Mesh(
    new THREE.PlaneGeometry(width * 1.02 + 0.04, height * 1.02 + 0.04),
    new THREE.MeshBasicMaterial({ color: 0x0a0a0c }),
  );
  bezel.position.z = -0.01;
  group.add(bezel);
  group.position.set(0, SCREEN_Y, -cfg.distance_m);

  const controls = video ? videoControls(video, ctx) : null;
  const d = cfg.distance_m;
  return {
    background: backgroundColor(scene.env, '#050608'),
    supportsAR: true,
    locomotion: (ar) => (ar ? { move: false, turn: false, teleport: false } : { move: true, turn: true, teleport: false }),
    recenterAnchor: { position: new THREE.Vector3(0, SCREEN_Y, 0), vertical: true },
    desktopView: {
      position: new THREE.Vector3(0, SCREEN_Y, 0),
      target: new THREE.Vector3(0, SCREEN_Y, -d),
      minDistance: Math.min(1, d * 0.5),
      maxDistance: d * 1.8,
      minAzimuthAngle: -0.7,
      maxAzimuthAngle: 0.7,
      minPolarAngle: Math.PI / 2 - 0.4,
      maxPolarAngle: Math.PI / 2 + 0.3,
    },
    media: video ? { video } : null,
    menuItems: controls ? controls.menuItems : [],
    onUserGesture: controls ? controls.onUserGesture : undefined,
    onButton: controls ? controls.onButton : undefined,
    update() {},
    dispose() {
      if (video) disposeVideo(video, videoTex);
    },
    debug: () => ({
      layout,
      kind: cfg.kind,
      aspect,
      width,
      height,
      stereo,
      video: video ? { paused: video.paused, muted: video.muted, time: video.currentTime } : null,
    }),
  };
}
