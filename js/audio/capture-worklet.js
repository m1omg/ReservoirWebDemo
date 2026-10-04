// Runs on the audio thread: forwards microphone samples to the page in
// batches. Nothing is recorded or sent anywhere else.

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.batch = new Float32Array(1024);
    this.fill = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      let i = 0;
      while (i < ch.length) {
        const n = Math.min(ch.length - i, this.batch.length - this.fill);
        this.batch.set(ch.subarray(i, i + n), this.fill);
        this.fill += n;
        i += n;
        if (this.fill === this.batch.length) {
          this.port.postMessage(this.batch.slice());
          this.fill = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor('rc-capture', CaptureProcessor);
