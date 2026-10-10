// splat: a full Gaussian-splat scene (e.g. a room captured with the COLMAP + Brush add-on)
// rendered with Spark.  You start at the scene origin looking down -z (or outside the scene
// when it is small) and can walk physically, use the left stick, snap turn and arc teleport.
// Units are metres with the floor at y = 0 (captures are scaled to real size by the app); the
// room / object decision uses robust bounds so stray "floater" splats do not count.

import * as THREE from 'three';
import { backgroundColor } from '../manifest.js';
import { createSparkRenderer, loadSplatMesh, splatRobustBounds, upAxisQuaternion } from '../loaders.js';

const ROOM_EXTENT_M = 4; // wider than this (and around the origin): a room you stand in

export async function load(ctx, scene) {
  const cfg = scene.splat;
  ctx.progress(0.01, 'Starting the splat renderer…');
  const spark = await createSparkRenderer(ctx.renderer);
  ctx.addToScene(spark);
  const mesh = await loadSplatMesh(ctx.asset(cfg.asset), cfg.format, {
    signal: ctx.signal,
    onProgress: (f) => ctx.progress(f < 0 ? -1 : 0.02 + f * 0.93, 'Loading 3D world…'),
  });
  ctx.checkAborted();

  const holder = new THREE.Group();
  holder.name = 'splat-world';
  holder.quaternion.copy(upAxisQuaternion(cfg.up));
  holder.scale.setScalar(cfg.scale);
  holder.add(mesh);
  ctx.root.add(holder);
  holder.updateMatrix();

  const box = splatRobustBounds(mesh).box.clone().applyMatrix4(holder.matrix);
  const size = box.isEmpty() ? new THREE.Vector3(4, 3, 4) : box.getSize(new THREE.Vector3());
  const center = box.isEmpty() ? new THREE.Vector3(0, 1, 0) : box.getCenter(new THREE.Vector3());
  const extent = Math.max(size.x, size.z);
  // A captured room surrounds the origin; a small capture is something to look at from outside.
  const inside = !box.isEmpty() && extent > ROOM_EXTENT_M && box.containsPoint(new THREE.Vector3(0, Math.min(1.2, box.max.y - 0.1), 0));
  const start = inside ? new THREE.Vector3(0, 1.6, 0) : new THREE.Vector3(center.x, 1.6, center.z + Math.max(1.5, extent * 1.2));

  return {
    background: backgroundColor(scene.env, '#000000'),
    supportsAR: true,
    heavy: true,
    locomotion: (ar) => (ar ? { move: false, turn: false, teleport: false } : { move: true, turn: true, teleport: true }),
    floorY: 0,
    recenterAnchor: { position: start, vertical: false },
    desktopView: inside
      ? { position: start.clone(), target: new THREE.Vector3(0, 1.6, -1.5), minDistance: 0.1, maxDistance: 30 }
      : {
          position: new THREE.Vector3(center.x, center.y + size.y * 0.3, center.z + Math.max(1.5, extent * 1.6)),
          target: center.clone(),
          minDistance: 0.2,
          maxDistance: Math.max(10, extent * 6),
        },
    update() {},
    dispose() {
      // SplatMesh is disposed with ctx.root; the SparkRenderer via ctx.addToScene.
    },
    debug: () => ({
      format: cfg.format,
      splats: mesh.packedSplats?.numSplats ?? mesh.splats?.getNumSplats?.() ?? null,
      inside,
      bounds: box.isEmpty() ? null : [box.min.toArray(), box.max.toArray()],
    }),
  };
}
