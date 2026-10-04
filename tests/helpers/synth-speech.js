// Crude formant synthesiser for testing the speech pipeline without a
// microphone: produces "up", "down", "left" and "right"-like sounds with
// random speaker pitch, vocal-tract length, speed, loudness and room noise.

import { Rng } from '../../js/core/rng.js';

// Each word is a list of segments: [durationFraction, kind, f1a, f2a, f1b, f2b]
// kind: 'v' voiced (formants glide a → b), 'n' noise burst/fricative, 's' silence.
const WORDS = {
  up: [[0.7, 'v', 650, 1200, 600, 1150], [0.12, 's'], [0.18, 'n', 1500, 4000]],
  down: [[0.1, 'n', 2500, 5000], [0.65, 'v', 700, 1200, 380, 850], [0.25, 'v', 250, 1500, 250, 1500]],
  left: [[0.2, 'v', 350, 1000, 400, 1200], [0.4, 'v', 550, 1800, 550, 1750], [0.25, 'n', 3000, 7000], [0.15, 'n', 1500, 4000]],
  right: [[0.2, 'v', 420, 1100, 500, 1150], [0.55, 'v', 750, 1200, 400, 2200], [0.08, 's'], [0.17, 'n', 1500, 4000]],
};

export const SYNTH_WORDS = Object.keys(WORDS);

export function synthWord(word, rng, sampleRate = 16000) {
  const f0 = rng.uniform(95, 230);
  const tract = rng.uniform(0.9, 1.12);
  const dur = rng.uniform(0.4, 0.62);
  const gain = rng.uniform(0.15, 0.6);
  const lead = Math.round(sampleRate * rng.uniform(0.3, 0.45));
  const tail = Math.round(sampleRate * 0.4);
  const N = Math.round(dur * sampleRate);
  const out = new Float32Array(lead + N + tail);
  const noiseFloor = rng.uniform(0.001, 0.004);
  for (let i = 0; i < out.length; i++) out[i] = rng.gauss() * noiseFloor;
  let pos = 0;
  let phase = 0;
  for (const seg of WORDS[word]) {
    const len = Math.round(seg[0] * N);
    const kind = seg[1];
    for (let k = 0; k < len; k++) {
      const t = k / Math.max(1, len - 1);
      const env = Math.min(1, k / 160, (len - k) / 160);
      let v = 0;
      if (kind === 'v') {
        const f1 = (seg[2] + (seg[4] - seg[2]) * t) * tract;
        const f2 = (seg[3] + (seg[5] - seg[3]) * t) * tract;
        const pitch = f0 * (1 + 0.08 * Math.sin(t * 3));
        phase += (2 * Math.PI * pitch) / sampleRate;
        for (let hmn = 1; hmn * pitch < 5000; hmn++) {
          const f = hmn * pitch;
          const a = Math.exp(-(((f - f1) / 110) ** 2)) + 0.7 * Math.exp(-(((f - f2) / 140) ** 2)) + 0.25 * Math.exp(-(((f - 2600 * tract) / 200) ** 2));
          v += (a / hmn ** 0.3) * Math.sin(hmn * phase);
        }
        v *= 0.25;
      } else if (kind === 'n') {
        // Band-limited noise: difference of two smoothed noises.
        v = rng.gauss() * 0.35;
      }
      out[lead + pos + k] += v * env * gain;
    }
    pos += len;
  }
  // Simple high-pass "colour" for fricatives / bursts: emphasise differences.
  return out;
}
