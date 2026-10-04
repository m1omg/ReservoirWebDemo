// Voice activity detection: finds where an utterance starts and stops by
// comparing each frame's loudness with an adaptive estimate of the room's
// background noise.

export class Vad {
  constructor({ onUtterance, sensitivityDb = 12, preRoll = 12, hangover = 22, minFrames = 15, maxFrames = 160 } = {}) {
    this.onUtterance = onUtterance;
    this.sensitivityDb = sensitivityDb;
    this.preRoll = preRoll;
    this.hangover = hangover;
    this.minFrames = minFrames;
    this.maxFrames = maxFrames;
    this.floor = null;
    this.recent = new Float32Array(300);
    this.recentHead = 0;
    this.recentCount = 0;
    this.sinceFloor = 0;
    this.history = []; // recent frames for pre-roll
    this.active = null;
    this.loud = 0;
    this.quiet = 0;
  }

  get threshold() {
    return (this.floor ?? -60) + this.sensitivityDb;
  }

  push(frame, rawDb) {
    // Digital silence would drag the background estimate to −∞; clamp it.
    const db = Math.max(-100, rawDb);
    // Background estimate: a low percentile of the last ~3 s of loudness,
    // so it follows the room even while someone keeps talking.
    this.recent[this.recentHead] = db;
    this.recentHead = (this.recentHead + 1) % this.recent.length;
    this.recentCount = Math.min(this.recent.length, this.recentCount + 1);
    if (++this.sinceFloor >= 10 || this.floor === null) {
      this.sinceFloor = 0;
      const sorted = Array.from(this.recent.subarray(0, this.recentCount)).sort((a, b) => a - b);
      this.floor = sorted[Math.floor(sorted.length * 0.1)];
    }

    const entry = { frame, db };
    if (!this.active) {
      this.history.push(entry);
      if (this.history.length > this.preRoll) this.history.shift();
      if (db > this.threshold) this.loud++;
      else this.loud = 0;
      if (this.loud >= 3) {
        this.active = this.history.slice();
        this.history = [];
        this.quiet = 0;
        this.peak = db;
      }
      return;
    }
    this.active.push(entry);
    this.peak = Math.max(this.peak, db);
    // "Quiet" is relative both to the background and to how loud this word
    // was, so a rising background (e.g. automatic gain) can't keep it open.
    if (db < Math.max(this.threshold - 4, this.peak - 22)) this.quiet++;
    else this.quiet = 0;
    if (this.quiet >= this.hangover || this.active.length >= this.maxFrames) this.finish();
  }

  finish() {
    const seg = this.active;
    this.active = null;
    this.loud = 0;
    // Trim most of the trailing silence, keep a little tail.
    const keep = Math.max(0, seg.length - Math.max(0, this.quiet - 6));
    const utt = seg.slice(0, keep);
    this.quiet = 0;
    if (utt.length >= this.minFrames) this.onUtterance?.(utt.map((e) => e.frame), utt.map((e) => e.db));
  }

  get speaking() {
    return this.active !== null;
  }
}
