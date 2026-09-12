/* One service state for the top animation and badge; validity lives in events.js. */
(() => {
  'use strict';
  let config = null, configConfirmed = false, checked = false, polling = false;
  let forecast = {available:false, delayed:true, fetchFailed:false, detail:'Loading the latest forecast.'};
  const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)');
  let playPending = false, playBlocked = false, userPaused = false;
  const labels = {
    maintenance:'Maintenance', unknown:'Status unconfirmed', checking:'Checking status',
    connection_error:'Connection interrupted', unavailable:'Forecast unavailable',
    delayed:'Update delayed', operating:'Forecast operating'
  };
  function resolve() {
    const state = config?.mode === 'maintenance' ? 'maintenance'
      : !configConfirmed ? (checked ? 'unknown' : 'checking')
      : forecast.fetchFailed ? 'connection_error'
      : !forecast.available ? 'unavailable'
      : forecast.delayed ? 'delayed' : 'operating';
    return {state, label:labels[state], message:state === 'maintenance' ?
      (config.message || 'Forecast service maintenance. Previously published research remains available.') :
      state === 'unknown' ? 'Service status could not be confirmed. Retrying automatically.' :
      state === 'checking' ? 'Confirming service status and current forecasts.' : forecast.detail};
  }
  function render() {
    const status = resolve(), brand = document.getElementById('ef-brand');
    if (brand) { brand.dataset.state = status.state; brand.dataset.motion = document.hidden ? 'paused' : 'running'; }
    const headerLogo = document.getElementById('ef-header-logo');
    if (headerLogo) headerLogo.dataset.state = status.state;
    const badge = document.getElementById('event-status'), detail = document.getElementById('event-updated');
    if (badge) { badge.textContent = status.label; badge.className = `pill ${status.state === 'operating' ? 'good' : 'warn'}`; }
    if (detail) detail.textContent = status.message;
    syncVideo(status, brand);
    return status;
  }
  function canPlay() {
    return resolve().state === 'operating' && !document.hidden && !reducedMotion?.matches && !userPaused;
  }
  function syncVideo(status, holder) {
    const video = document.getElementById('ef-video');
    if (!holder || typeof video?.play !== 'function') return;
    const allowed = canPlay();
    holder.dataset.motion = allowed && !playBlocked ? 'running' : 'paused';
    const errorNote = document.getElementById('ef-video-error');
    if (errorNote) errorNote.hidden = !video.error;
    const toggle = document.getElementById('ef-video-toggle');
    if (toggle) {
      toggle.hidden = status.state !== 'operating' || !!reducedMotion?.matches || !!video.error;
      toggle.textContent = userPaused || playBlocked ? 'Play animation' : 'Pause animation';
    }
    if (!allowed || video.error) {
      video.pause(); holder.dataset.playback = 'paused';
      return;
    }
    if (!video.paused || playPending || playBlocked) return;
    video.muted = true;
    playPending = true;
    Promise.resolve(video.play()).then(() => {
      // A maintenance transition can arrive while play() is still pending.
      if (!canPlay()) { video.pause(); holder.dataset.playback = 'paused'; }
      else holder.dataset.playback = 'playing';
    }).catch(error => {
      if (canPlay() && error.name !== 'AbortError') playBlocked = true;
      holder.dataset.playback = 'paused';
    }).finally(() => { playPending = false; render(); });
  }
  async function refreshConfig() {
    if (polling) return;
    polling = true;
    try {
      const response = await fetch('site_status.json', {cache:'no-store', signal:globalThis.AbortSignal?.timeout?.(10000)});
      if (!response.ok) throw new Error('status unavailable');
      const next = await response.json();
      if (!next || next.schema_version !== 1 || !['auto','maintenance'].includes(next.mode) ||
          (next.message !== undefined && (typeof next.message !== 'string' || next.message.length > 500))) throw new Error('invalid status');
      config = next; configConfirmed = true;
    } catch (_) {
      // Retain a confirmed maintenance instruction until a valid replacement arrives.
      configConfirmed = false;
    } finally { checked = true; polling = false; render(); }
  }
  function mountVideo() {
    const video = document.getElementById('ef-video');
    if (typeof video?.play !== 'function') return;
    video.muted = true;
    video.addEventListener('playing', () => {
      if (!canPlay()) video.pause();
      else document.getElementById('ef-brand').dataset.playback = 'playing';
    });
    video.addEventListener('error', render);
    const toggle = document.getElementById('ef-video-toggle');
    if (toggle) toggle.addEventListener('click', () => {
      userPaused = playBlocked ? false : !userPaused;
      playBlocked = false; render();
    });
  }
  window.EtherForecastBrand = Object.freeze({
    updateForecast(value) { forecast = {...value}; return render(); },
    refreshConfig,
    getStatus:resolve
  });
  document.addEventListener('visibilitychange', render);
  reducedMotion?.addEventListener?.('change', render);
  document.addEventListener('DOMContentLoaded', () => {
    mountVideo(); render(); refreshConfig();
    setInterval(refreshConfig, 60000);
  });
})();
