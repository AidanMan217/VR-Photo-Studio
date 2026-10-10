// Shared video behaviour for stereo / sphere modes: muted autoplay, unmute on the first user
// gesture (WebXR only allows audio after a trigger press / click), play-pause and loop controls.

import { log } from './util.js';

/**
 * @param {HTMLVideoElement} video
 * @param {object} ctx mode context (for toasts and menu refresh)
 * @returns {{ menuItems: object[], onUserGesture: Function, onButton: Function, togglePlay: Function }}
 */
export function videoControls(video, ctx) {
  let unmutedOnce = false;
  const play = () =>
    video.play().catch((err) => {
      log.info('video play blocked', err?.name || err);
      ctx.toast('Press play to start the video');
    });

  const togglePlay = () => {
    if (video.paused) play();
    else video.pause();
    ctx.refreshMenu();
  };

  return {
    togglePlay,
    onUserGesture() {
      if (unmutedOnce) return;
      unmutedOnce = true;
      video.muted = false;
      if (video.paused) play();
    },
    onButton(name) {
      if (name === 'primary') togglePlay();
    },
    menuItems: [
      { id: 'play', label: () => (video.paused ? '▶ Play' : '⏸ Pause'), action: togglePlay },
      {
        id: 'mute',
        label: () => (video.muted ? 'Sound: off' : 'Sound: on'),
        action: () => {
          video.muted = !video.muted;
          unmutedOnce = true;
          if (!video.muted && video.paused) play();
          ctx.refreshMenu();
        },
      },
      {
        id: 'loop',
        label: () => (video.loop ? 'Loop: on' : 'Loop: off'),
        action: () => {
          video.loop = !video.loop;
          if (!video.loop && video.ended) video.currentTime = 0;
          ctx.refreshMenu();
        },
      },
      {
        id: 'restart',
        label: '⏮ Restart',
        action: () => {
          video.currentTime = 0;
          play();
        },
      },
    ],
  };
}
