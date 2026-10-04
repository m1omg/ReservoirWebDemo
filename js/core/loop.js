// Fixed-timestep animation loop.
//
// Simulations advance a fixed number of steps per *second of wall-clock
// time*, never per frame, so a 30 Hz laptop and a 240 Hz gaming monitor see
// exactly the same speed. Rendering happens once per display frame.

export class FixedStepLoop {
  constructor({
    stepsPerSecond = 60,
    step,
    render = () => {},
    maxCatchUpSeconds = 0.25,
    maxStepsPerFrame = 5000,
    now = () => performance.now(),
    requestFrame = (cb) => requestAnimationFrame(cb),
    cancelFrame = (id) => cancelAnimationFrame(id),
  }) {
    this.stepsPerSecond = stepsPerSecond;
    this.stepFn = step;
    this.renderFn = render;
    this.maxCatchUpSeconds = maxCatchUpSeconds;
    this.maxStepsPerFrame = maxStepsPerFrame;
    this.now = now;
    this.requestFrame = requestFrame;
    this.cancelFrame = cancelFrame;
    this.running = false;
    this.accumulator = 0;
    this.last = 0;
    this.frameId = 0;
    this.totalSteps = 0;
    this._frame = (t) => this.tick(t);
  }

  setRate(stepsPerSecond) {
    this.stepsPerSecond = Math.max(0, stepsPerSecond);
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.accumulator = 0;
    this.last = this.now();
    this.frameId = this.requestFrame(this._frame);
  }

  stop() {
    this.running = false;
    if (this.frameId) this.cancelFrame(this.frameId);
    this.frameId = 0;
  }

  tick(timestamp) {
    if (!this.running) return;
    const t = typeof timestamp === 'number' ? timestamp : this.now();
    let elapsed = (t - this.last) / 1000;
    this.last = t;
    // After a hidden tab or a long stall, don't try to replay the gap.
    if (elapsed > this.maxCatchUpSeconds) elapsed = this.maxCatchUpSeconds;
    if (elapsed < 0) elapsed = 0;
    this.accumulator += elapsed * this.stepsPerSecond;
    let n = Math.floor(this.accumulator);
    this.accumulator -= n;
    if (n > this.maxStepsPerFrame) n = this.maxStepsPerFrame;
    for (let i = 0; i < n; i++) {
      this.stepFn();
      this.totalSteps++;
    }
    this.renderFn(this.accumulator);
    if (this.running) this.frameId = this.requestFrame(this._frame);
  }
}

// Runs a long computation in slices of at most `budgetMs`, yielding to the
// browser in between so the page stays responsive. `work(i)` is called for
// i = 0..total-1; `onProgress(fraction)` after every slice.
export async function runChunked(total, work, { budgetMs = 12, onProgress, now = () => performance.now() } = {}) {
  let i = 0;
  while (i < total) {
    const start = now();
    while (i < total) {
      work(i++);
      if ((i & 15) === 0 && now() - start > budgetMs) break;
    }
    if (onProgress) onProgress(i / total);
    if (i < total) await new Promise((resolve) => setTimeout(resolve, 0));
  }
}
