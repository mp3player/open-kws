import assert from 'node:assert/strict';
import test from 'node:test';
import {readFile} from 'node:fs/promises';
import {RollingAudioBuffer, WakeGate, microphoneConstraints, pcm16, wav} from '../web_kws/app.mjs';

test('release web interface uses English', async () => {
  const html = await readFile(new URL('../web_kws/index.html', import.meta.url), 'utf8');
  const script = await readFile(new URL('../web_kws/app.mjs', import.meta.url), 'utf8');
  assert.ok(html.includes('lang="en"'));
  assert.ok(html.includes('Enroll a keyword'));
  assert.ok(html.includes('Start listening'));
  assert.equal(/\p{Script=Han}/u.test(html + script), false);
});

test('rolling windows retain exactly the latest samples without aliasing', () => {
  const ring = new RollingAudioBuffer(4);
  ring.push([1, 2]);
  assert.equal(ring.ready, false);
  assert.throws(() => ring.snapshot());
  ring.push([3, 4, 5]);
  assert.deepEqual([...ring.snapshot()], [2, 3, 4, 5]);
  const previous = ring.snapshot();
  ring.push([6, 7, 8, 9, 10]);
  assert.deepEqual([...ring.snapshot()], [7, 8, 9, 10]);
  assert.deepEqual([...previous], [2, 3, 4, 5]);
});

test('PCM16 is little endian and clips safely', () => {
  const view = new DataView(pcm16([-2, -1, 0, .5, 1, 2]));
  assert.deepEqual(Array.from({length: 6}, (_, position) => view.getInt16(position * 2, true)),
    [-32768, -32768, 0, 16384, 32767, 32767]);
});

test('microphone constraints keep preprocessing off and can select an exact device', () => {
  const defaults = microphoneConstraints();
  assert.equal(defaults.audio.echoCancellation, false);
  assert.equal(defaults.audio.noiseSuppression, false);
  assert.equal(defaults.audio.autoGainControl, false);
  assert.equal(defaults.video, false);
  assert.equal('deviceId' in defaults.audio, false);
  assert.deepEqual(microphoneConstraints('phone-mic').audio.deviceId, {exact: 'phone-mic'});
});

test('playback WAV preserves the same PCM sent to the model', async () => {
  const samples = Float32Array.from([0, -.25, .5, -.75, 1, -1]);
  const view = new DataView(await wav(samples, 48000).arrayBuffer());
  assert.equal(view.getUint32(24, true), 48000);
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), samples.length * 2);
  assert.deepEqual(new Uint8Array(view.buffer, 44), new Uint8Array(pcm16(samples)));
});

test('wake requires consistent hits, quiet rearming, and cooldown', () => {
  const gate = new WakeGate();
  assert.equal(gate.update('first', .8, .5, 0), false);
  assert.equal(gate.update('second', .8, .5, 500), false);
  assert.equal(gate.update('second', .8, .5, 1000), true);
  assert.equal(gate.update('second', .9, .5, 5000), false);
  gate.update(null, null, .5, 5100);
  gate.update(null, null, .5, 5200);
  assert.equal(gate.update('first', .8, .5, 5300), false);
  assert.equal(gate.update('first', .8, .5, 5800), true);
  gate.update(null, null, .5, 5900);
  gate.update(null, null, .5, 6000);
  gate.update('first', .8, .5, 6100);
  assert.equal(gate.update('first', .8, .5, 6200), false);
  assert.equal(gate.update('first', .8, .5, 7400), true);
});

test('AudioWorklet downmixes to mono and emits silent output', async () => {
  let Processor;
  globalThis.AudioWorkletProcessor = class { constructor() { this.port = {postMessage: buffer => this.sent = buffer}; } };
  globalThis.registerProcessor = (name, value) => { assert.equal(name, 'kws-capture'); Processor = value; };
  await import('../web_kws/capture-worklet.js');
  const processor = new Processor();
  for (let position = 0; position < 16; position++) {
    const output = new Float32Array(128).fill(1);
    assert.equal(processor.process([[new Float32Array(128).fill(.5), new Float32Array(128).fill(-.1)]], [[output]]), true);
    assert.ok(output.every(sample => sample === 0));
  }
  assert.equal(processor.sent.length, 2048);
  assert.ok(processor.sent.every(sample => Math.abs(sample - .2) < 1e-6));
});
