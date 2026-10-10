// photo3d: layered depth meshes (GLB) of a photo, positioned in metres relative to the original
// camera at (0, eye_height_m, 0) looking down -z.  On entering VR (and on Recenter) the user's
// head is moved exactly to that viewpoint, so the photo looks "right" and you can lean or step
// to look behind things.  Unlit materials: the photo already contains its lighting.

import * as THREE from 'three';
import { backgroundColor } from '../manifest.js';
import { loadGLB, makeUnlit } from '../loaders.js';

/** @returns {Promise<object>} mode handle (see js/app.js) */
export async function load(ctx, scene) {
  const cfg = scene.photo3d;
  const eye = cfg.eye_height_m;
  const anchor = cfg.anchor_distance_m;
  const group = new THREE.Group();
  group.name = 'photo3d';
  ctx.root.add(group);

  const files = cfg.meshes;
  const anisotropy = Math.min(4, ctx.renderer.capabilities.getMaxAnisotropy());
  let triangles = 0;
  for (let i = 0; i < files.length; i++) {
    const label = files.length > 1 ? `Loading 3D layer ${i + 1} of ${files.length}…` : 'Loading 3D photo…';
    const gltf = await loadGLB(ctx.asset(files[i]), {
      signal: ctx.signal,
      onProgress: (f) => ctx.progress((i + Math.max(0, f)) / files.length * 0.95, label),
    });
    ctx.checkAborted();
    makeUnlit(gltf.scene, { anisotropy });
    // Layers are listed back -> front; draw in that order so cut-out edges blend correctly.
    gltf.scene.traverse((obj) => {
      if (obj.isMesh) {
        obj.renderOrder = i;
        triangles += obj.geometry.index ? obj.geometry.index.count / 3 : obj.geometry.attributes.position.count / 3;
      }
    });
    gltf.scene.name = files[i];
    group.add(gltf.scene);
  }

  const viewpoint = new THREE.Vector3(0, eye, 0);
  return {
    background: backgroundColor(scene.env, '#000000'),
    supportsAR: true,
    // Walk/lean physically; a gentle stick move is allowed, teleporting would leave the photo.
    locomotion: (ar) => (ar ? { move: false, turn: false, teleport: false } : { move: true, turn: true, teleport: false }),
    recenterAnchor: { position: viewpoint, vertical: true },
    desktopView: {
      position: viewpoint.clone(),
      target: new THREE.Vector3(0, eye, -anchor),
      // Keep the preview near the original viewpoint where the reconstruction is complete.
      minDistance: anchor * 0.55,
      maxDistance: anchor * 1.15,
      minAzimuthAngle: -0.22,
      maxAzimuthAngle: 0.22,
      minPolarAngle: Math.PI / 2 - 0.16,
      maxPolarAngle: Math.PI / 2 + 0.16,
      enablePan: false,
      keyboardMove: true,
    },
    update() {},
    dispose() {
      // Meshes live under ctx.root, which the app disposes.
    },
    debug: () => ({ layers: group.children.length, triangles: Math.round(triangles), eyeHeight: eye }),
  };
}
