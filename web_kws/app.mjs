export class RollingAudioBuffer {
  constructor(size) {
    this.samples = new Float32Array(size);
    this.position = 0;
    this.count = 0;
  }
  push(chunk) {
    for (const sample of chunk) {
      this.samples[this.position] = sample;
      this.position = (this.position + 1) % this.samples.length;
      this.count++;
    }
  }
  get ready() { return this.count >= this.samples.length; }
  snapshot() {
    if (!this.ready) throw Error('The audio window is not full yet');
    const output = new Float32Array(this.samples.length);
    output.set(this.samples.subarray(this.position));
    output.set(this.samples.subarray(0, this.position), this.samples.length - this.position);
    return output;
  }
}

export function pcm16(samples) {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  for (let position = 0; position < samples.length; position++) {
    const sample = Math.min(1, Math.max(-1, samples[position]));
    view.setInt16(position * 2, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
  }
  return buffer;
}

export function microphoneConstraints(deviceId = '') {
  return {audio: {
    channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false,
    ...(deviceId ? {deviceId: {exact: deviceId}} : {})
  }, video: false};
}

export function wav(samples, rate) {
  const body = pcm16(samples);
  const buffer = new ArrayBuffer(44 + body.byteLength);
  const view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((character, position) => view.setUint8(offset + position, character.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, 36 + body.byteLength, true); text(8, 'WAVE'); text(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, body.byteLength, true);
  new Uint8Array(buffer, 44).set(new Uint8Array(body));
  return new Blob([buffer], {type: 'audio/wav'});
}

export class WakeGate {
  constructor() { this.reset(); }
  reset() {
    this.label = null;
    this.hits = 0;
    this.armed = true;
    this.quiet = 0;
    this.lastWake = -Infinity;
  }
  update(label, score, threshold, now) {
    if (score === null || score < threshold - .05) {
      this.quiet++;
      this.hits = 0;
      this.label = null;
      if (this.quiet >= 2) this.armed = true;
      return false;
    }
    this.quiet = 0;
    if (score < threshold) { this.hits = 0; this.label = null; return false; }
    this.hits = label === this.label ? this.hits + 1 : 1;
    this.label = label;
    if (this.armed && this.hits >= 2 && now - this.lastWake >= 1500) {
      this.armed = false;
      this.lastWake = now;
      return true;
    }
    return false;
  }
}

async function boot() {
  const element = id => document.getElementById(id);
  const fragment = new URLSearchParams(location.hash.slice(1));
  let savedToken = '';
  try { savedToken = sessionStorage.getItem('kws-token') || ''; } catch {}
  let token = fragment.get('token') || savedToken;
  try { if (token) sessionStorage.setItem('kws-token', token); } catch {}
  if (location.hash) history.replaceState(null, '', location.pathname);
  let state = null;
  let mode = 'idle';
  let generation = 0;
  let context = null;
  let stream = null;
  let source = null;
  let worklet = null;
  let ring = null;
  let hopSamples = 0;
  let sinceRequest = 0;
  let inFlight = false;
  let requestController = null;
  let playbackUrl = null;
  let enrollingLabel = '';
  const events = [];
  const gate = new WakeGate();

  function status(message, error = false) {
    element('status').textContent = message;
    element('status').className = error ? 'error' : '';
  }
  function controls() {
    const idle = mode === 'idle';
    element('enroll').disabled = !state || !idle;
    element('listen').disabled = !state?.keywords.length || (!idle && mode !== 'listen');
    element('listen').textContent = mode === 'listen' ? 'Stop listening' : 'Start listening';
    element('stop').disabled = idle;
    element('clear').disabled = !state?.keywords.length || !idle;
    element('keyword').disabled = !idle;
    element('microphone').disabled = !idle;
    element('capture-state').classList.toggle('active', !idle);
    element('capture-label').textContent = idle ? 'Microphone off' : mode === 'listen' ? 'Listening'
      : mode === 'prepare' ? 'Preparing; not recording yet' : mode === 'upload' ? 'Uploading' : 'Recording';
    for (const button of element('keywords').querySelectorAll('button')) button.disabled = !idle;
  }
  function renderState(next) {
    state = next;
    element('model').textContent = `CPU · ${next.encoder_type.toUpperCase()} / ${next.activation}`;
    element('model-detail').textContent = `${next.model} · ${next.dimensions}-D · fingerprint ${next.fingerprint.slice(0, 12)} · raw audio is not saved`;
    element('enroll').textContent = `Record enrollment (${next.window_seconds} s)`;
    element('window-info').textContent = `${next.window_seconds} s rolling window, matching every ${next.hop_seconds} s. Keep this page visible and your phone unlocked.`;
    const list = element('keywords');
    list.replaceChildren();
    if (!next.keywords.length) {
      const empty = document.createElement('p'); empty.className = 'muted'; empty.textContent = 'No enrolled keywords'; list.append(empty);
    }
    for (const item of next.keywords) {
      const row = document.createElement('div'); row.className = 'keyword row between';
      const description = document.createElement('div');
      const title = document.createElement('strong'); title.textContent = item.label;
      const detail = document.createElement('small'); detail.textContent = `${item.samples}/5 enrollment recordings${item.samples < 3 ? ' · more independent takes recommended' : ''}`;
      description.append(title, detail);
      const remove = document.createElement('button'); remove.className = 'secondary compact'; remove.textContent = 'Remove';
      remove.onclick = () => deleteKeyword(item.label);
      row.append(description, remove); list.append(row);
    }
    controls();
  }
  async function api(path, options = {}) {
    const response = await fetch(path, {...options, headers: {...(token ? {'X-KWS-Token': token} : {}), ...options.headers}});
    const data = await response.json();
    if (!response.ok) throw Error(data.error || `Request failed: ${response.status}`);
    return data;
  }
  async function stopCapture(message) {
    const stopped = ++generation;
    mode = 'idle';
    requestController?.abort(); requestController = null;
    if (worklet) { worklet.port.onmessage = null; worklet.disconnect(); }
    source?.disconnect();
    stream?.getTracks().forEach(track => track.stop());
    const previous = context;
    worklet = source = stream = context = ring = null;
    inFlight = false;
    gate.reset();
    element('level').style.width = '0%';
    element('level-meter').setAttribute('aria-valuenow', '0');
    controls();
    if (previous && previous.state !== 'closed') await previous.close().catch(() => {});
    if (message && stopped === generation) status(message);
  }
  async function openMicrophone() {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      throw Error('Microphone unavailable: install and trust the local CA, then open the HTTPS link in Chrome or Safari.');
    }
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!Context) throw Error('Web Audio is unavailable. Please update Chrome or Safari.');
    element('playback').pause();
    const selectedDevice = element('microphone').value;
    context = new Context();
    const openedContext = context;
    const current = generation;
    await context.resume();
    if (current !== generation) return false;
    const openedStream = await navigator.mediaDevices.getUserMedia(microphoneConstraints(selectedDevice));
    if (current !== generation) { openedStream.getTracks().forEach(track => track.stop()); return false; }
    stream = openedStream;
    const track = stream.getAudioTracks()[0];
    const settings = track.getSettings();
    element('capture-info').textContent = `Actual input: ${track.label || 'System default microphone'} · phone rate ${openedContext.sampleRate} Hz · resampled to 16 kHz on the computer · live monitoring is off`;
    element('input-settings').textContent = `Microphone: ${track.muted ? 'muted' : 'unmuted'} · browser noise suppression: ${String(settings.noiseSuppression ?? 'not reported')} · echo cancellation: ${String(settings.echoCancellation ?? 'not reported')} · automatic gain: ${String(settings.autoGainControl ?? 'not reported')}`;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      if (current !== generation) return false;
      const choices = element('microphone');
      choices.replaceChildren(new Option('System default microphone', ''));
      for (const device of devices.filter(item => item.kind === 'audioinput' && item.deviceId)) {
        choices.add(new Option(device.label || `Microphone ${choices.options.length}`, device.deviceId));
      }
      choices.value = selectedDevice;
      if (choices.selectedIndex < 0) choices.value = '';
    } catch {}
    if (current !== generation) return false;
    if (!openedContext.audioWorklet) throw Error('AudioWorklet is unavailable. Please update Chrome or Safari.');
    await openedContext.audioWorklet.addModule('/capture-worklet.js');
    if (current !== generation) return false;
    ring = new RollingAudioBuffer(Math.round(openedContext.sampleRate * state.window_seconds));
    hopSamples = Math.round(openedContext.sampleRate * state.hop_seconds);
    sinceRequest = 0;
    source = openedContext.createMediaStreamSource(stream);
    worklet = new AudioWorkletNode(openedContext, 'kws-capture', {channelCount: 1, channelCountMode: 'explicit'});
    worklet.port.onmessage = event => onAudio(event.data);
    source.connect(worklet).connect(openedContext.destination);
    stream.getTracks().forEach(track => track.addEventListener('ended', () => stopCapture('Microphone disconnected. Please start again.'), {once: true}));
    return true;
  }
  function onAudio(chunk) {
    let energy = 0;
    for (const sample of chunk) energy += sample * sample;
    const level = Math.min(100, Math.sqrt(energy / chunk.length) * 600);
    element('level').style.width = `${level}%`;
    element('level-meter').setAttribute('aria-valuenow', String(Math.round(level)));
    if (!['enroll', 'listen'].includes(mode) || !ring) return;
    ring.push(chunk);
    if (mode === 'enroll') {
      const remaining = Math.max(0, (ring.samples.length - ring.count) / context.sampleRate);
      element('capture-label').textContent = `Recording · ${remaining.toFixed(1)} s remaining`;
    }
    sinceRequest += chunk.length;
    if (!ring.ready) return;
    if (mode === 'enroll') {
      mode = 'upload';
      finishEnrollment(ring.snapshot(), context.sampleRate, generation);
    } else if (sinceRequest >= hopSamples) {
      sinceRequest %= hopSamples;
      if (!inFlight) detect(ring.snapshot(), context.sampleRate, generation);
    }
  }
  async function finishEnrollment(samples, rate, current) {
    const label = enrollingLabel;
    const stopped = current + 1;
    await stopCapture();
    if (stopped !== generation) return;
    mode = 'upload'; controls();
    const controller = new AbortController();
    requestController = controller;
    const timeout = setTimeout(() => controller.abort(), 10000);
    status('Recording complete. Creating enrollment…');
    if (playbackUrl) URL.revokeObjectURL(playbackUrl);
    playbackUrl = URL.createObjectURL(wav(samples, rate));
    element('playback').src = playbackUrl; element('playback').hidden = false;
    element('download-recording').href = playbackUrl; element('download-recording').hidden = false;
    let energy = 0;
    let peak = 0;
    for (const sample of samples) { energy += sample * sample; peak = Math.max(peak, Math.abs(sample)); }
    element('recording-info').textContent = `Last recording: ${(samples.length / rate).toFixed(2)} s · RMS ${Math.sqrt(energy / samples.length).toFixed(4)} · peak ${peak.toFixed(4)}. Playback comes directly from phone capture, without model processing or additional audio effects.`;
    try {
      const result = await api(`/api/enroll?rate=${rate}&label=${encodeURIComponent(label)}`, {
        method: 'POST', headers: {'Content-Type': 'application/octet-stream'}, body: pcm16(samples), signal: controller.signal
      });
      if (stopped !== generation) return;
      renderState(result.state);
      status(`Enrolled “${label}”. Record another independent take or start listening.`);
    } catch (error) {
      if (stopped === generation) status(`Enrollment failed: ${error.name === 'AbortError' ? 'Upload timed out; check the LAN connection' : error.message}`, true);
    } finally {
      clearTimeout(timeout);
      if (stopped === generation) { mode = 'idle'; requestController = null; controls(); }
    }
  }
  async function detect(samples, rate, current) {
    const threshold = Number(element('threshold').value);
    if (!element('threshold').value.trim() || !Number.isFinite(threshold) || threshold < -1 || threshold > 1) {
      await stopCapture('The threshold must be between -1 and 1.'); return;
    }
    inFlight = true;
    const controller = new AbortController();
    requestController = controller;
    const timeout = setTimeout(() => controller.abort(), 10000);
    const started = performance.now();
    try {
      const result = await api(`/api/detect?rate=${rate}&threshold=${threshold}`, {
        method: 'POST', headers: {'Content-Type': 'application/octet-stream'}, body: pcm16(samples), signal: controller.signal
      });
      if (current !== generation || mode !== 'listen') return;
      element('match').textContent = result.label || 'Low-energy window skipped';
      element('score').textContent = result.score === null ? '—' : result.score.toFixed(3);
      element('ranking').textContent = result.scores.slice(0, 3).map(item => `${item.label} ${item.cosine.toFixed(3)}`).join(' · ');
      element('timing').textContent = `CPU processing ${result.audio.inference_ms} ms · round trip ${Math.round(performance.now() - started)} ms · RMS ${result.audio.rms.toFixed(4)}`;
      if (gate.update(result.label, result.score, threshold, performance.now())) {
        const message = `Wake detected: ${result.label}`;
        element('wake').textContent = message; element('wake').hidden = false;
        events.unshift({label: result.label, score: result.score, time: new Date().toLocaleTimeString()});
        if (events.length > 30) events.pop();
        renderEvents();
        navigator.vibrate?.(120);
        status(`${message}. Pause or say something different to rearm.`);
      }
    } catch (error) {
      if (current === generation) {
        await stopCapture(); status(`Listening stopped: ${error.name === 'AbortError' ? 'Request timed out; check the LAN connection' : error.message}`, true);
      }
    } finally {
      clearTimeout(timeout);
      if (current === generation) { inFlight = false; requestController = null; }
    }
  }
  function renderEvents() {
    element('events').replaceChildren();
    for (const event of events) {
      const row = document.createElement('div'); row.className = 'event';
      row.textContent = `${event.time}　${event.label}　${event.score.toFixed(3)}`;
      element('events').append(row);
    }
    if (!events.length) element('events').textContent = 'No wake events yet';
  }
  async function deleteKeyword(label) {
    if (!confirm(label === null ? 'Delete all enrolled keywords?' : `Delete all enrollment samples for “${label}”?`)) return;
    try {
      const result = await api('/api/delete', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({label})});
      renderState(result.state); status('Enrollment templates deleted.');
    } catch (error) { status(error.message, true); }
  }
  element('enroll').onclick = async () => {
    const label = element('keyword').value.trim();
    if (!label) { status('Enter a keyword label first.', true); return; }
    generation++;
    mode = 'prepare'; controls(); gate.reset(); element('wake').hidden = true;
    enrollingLabel = label;
    const current = generation;
    try {
      if (!await openMicrophone()) return;
      for (let count = 2; count > 0; count--) {
        if (current !== generation) return;
        status(`Preparing “${label}”: ${count}… Not recording yet; wait for “Start speaking”.`);
        element('capture-label').textContent = `Prepare ${count} s · not recording yet`;
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
      if (current !== generation) return;
      mode = 'enroll'; controls(); status(`Start speaking: say “${label}” once within ${state.window_seconds} seconds.`);
      element('capture-label').textContent = `Start speaking · ${state.window_seconds} s`;
    } catch (error) { if (current === generation) { await stopCapture(); status(`Could not open the microphone: ${error.message}`, true); } }
  };
  element('listen').onclick = async () => {
    if (mode === 'listen') { await stopCapture('Listening stopped; microphone closed.'); return; }
    generation++;
    mode = 'prepare'; controls(); gate.reset(); element('wake').hidden = true;
    const current = generation;
    try {
      if (!await openMicrophone()) return;
      if (current !== generation) return;
      mode = 'listen'; controls(); status(`Listening; filling the initial ${state.window_seconds} s audio window…`);
    } catch (error) { if (current === generation) { await stopCapture(); status(`Could not open the microphone: ${error.message}`, true); } }
  };
  element('stop').onclick = () => stopCapture('Microphone stopped.');
  element('clear').onclick = () => deleteKeyword(null);
  element('clear-events').onclick = () => { events.length = 0; renderEvents(); element('wake').hidden = true; };
  element('threshold').onchange = () => gate.reset();
  element('reset-threshold').onclick = () => { element('threshold').value = state.threshold; gate.reset(); };
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && mode !== 'idle') stopCapture('Page moved to the background; microphone stopped. Return to this page and start again.');
  });
  window.addEventListener('pagehide', () => stopCapture());
  try {
    const result = await api('/api/state');
    if (!result.token_required) {
      token = '';
      try { sessionStorage.removeItem('kws-token'); } catch {}
    }
    renderState(result); element('threshold').value = result.threshold;
    element('reset-threshold').disabled = false;
    status(`Connected. Model: ${result.model}. Select “Record enrollment” and allow microphone access to begin.`);
  } catch (error) { status(error.message, true); }
}

if (typeof document !== 'undefined') boot();
