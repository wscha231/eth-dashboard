/* One state for the badge and original-logo lighting; forecast validity lives in events.js. */
(() => {
  'use strict';
  let config = null, configConfirmed = false, checked = false, polling = false;
  let forecast = {available:false, delayed:true, fetchFailed:false, detail:'Loading the latest forecast.'};
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
    return status;
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
  async function mountLogo() {
    const holder = document.getElementById('ef-brand');
    if (!holder) return;
    try {
      const response = await fetch('assets/etherforecast-logo.svg', {signal:globalThis.AbortSignal?.timeout?.(10000)});
      if (!response.ok) throw new Error('logo unavailable');
      const parsed = new DOMParser().parseFromString(await response.text(), 'image/svg+xml');
      const svg = parsed.documentElement;
      if (svg.localName !== 'svg' || parsed.querySelector('parsererror,script,foreignObject')) throw new Error('invalid logo');
      // This is a fixed, versioned first-party asset, never a URL or markup from status JSON.
      svg.setAttribute('aria-hidden','true'); svg.removeAttribute('aria-labelledby');
      holder.replaceChildren(document.importNode(svg, true));
    } catch (_) { holder.dataset.asset = 'fallback'; }
  }
  window.EtherForecastBrand = Object.freeze({
    updateForecast(value) { forecast = {...value}; return render(); },
    refreshConfig,
    getStatus:resolve
  });
  document.addEventListener('visibilitychange', render);
  document.addEventListener('DOMContentLoaded', () => {
    render(); mountLogo(); refreshConfig();
    setInterval(refreshConfig, 60000);
  });
})();
