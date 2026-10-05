# Reservoir Computing Playground

**Machine learning without training the network.** Four interactive demos of
[reservoir computing](https://en.wikipedia.org/wiki/Reservoir_computing), each
on a task that's usually given to LSTMs or CNNs. Everything runs in your browser
and trains in milliseconds. There's no server, no GPU and nothing to install.

**▶ Live version:** <https://m1omg.github.io/ReservoirWebDemo/>
(available once GitHub Pages is enabled, see [Publishing](#publishing))

## What is reservoir computing?

A normal recurrent neural network learns all of its weights by backpropagation
through time, which is slow, fiddly and data-hungry. A reservoir computer takes
a different route:

1. Build a big recurrent network with **random, fixed** connections (the *reservoir*).
2. Feed the input signal into it. Every neuron ends up with its own nonlinear,
   fading echo of the recent past.
3. Train **only a linear readout** on top of those echoes. That is one ridge
   regression, a single linear solve, with no gradient descent.

Training takes milliseconds, so it can happen live in the page. The reservoir
also doesn't have to be software: real tanks of water, photonic chips and other
physical systems have been used as reservoirs.

## The demos

| Demo | What it does | Usually done with | Here |
|---|---|---|---|
| **Forecasting chaos** | Predicts the Lorenz, Rössler and Mackey–Glass systems, then keeps "dreaming" their attractors indefinitely. Includes a linear model with no reservoir for comparison. | LSTMs / GRUs trained with backpropagation | 300 random neurons and one ridge regression. Trains in about 0.3–0.5 s and predicts Lorenz about 8 Lyapunov times ahead. |
| **Gesture recognition** | Draw shapes with a mouse, finger or pen. It guesses while you draw, and you can teach it new gestures from 3 examples. | CNNs / RNNs on large labelled datasets | 200 random neurons. About 99% accurate on unseen drawings; retrains in well under a second. |
| **Spoken words** | Record "up", "down", "left" and "right" a few times, then steer a token around a grid with your voice. | CNNs trained on datasets like Google Speech Commands (105,829 clips) | About 5 of your own recordings per word, with a leave-one-out self-check. |
| **A bucket of water** | A simulated pond is the reservoir. Ripples read at 48 points are enough to tell a sine wave from a square wave and to remember past inputs. | A trained network does the processing | Physics does the processing; only the 97 readout weights are trained. |

Every demo lets you change the reservoir's settings (size, spectral radius, leak
rate, input scaling, regularisation) and see what happens. **↺ Reset settings**
puts them back to the defaults.

## Simple or detailed, English or Slovak

Every description comes in two versions: **Simple**, a plain-language "explain
like I'm five", and **Detailed**, the technical one. Switch between them right
under each title. The page remembers your choice.

The whole app is also available in **Slovak**. It opens in Slovak automatically
when your browser's preferred language is Slovak, and the **EN | SK** switch in
the header changes it at any time.

Both choices can be set in a link, which is handy for sharing:

| Link | Opens |
|---|---|
| `…/ReservoirWebDemo/?explain=simple` | Simple explanations (the default) |
| `…/ReservoirWebDemo/?explain=detailed` | Technical explanations |
| `…/ReservoirWebDemo/?lang=sk` | Slovak |
| `…/ReservoirWebDemo/?lang=en` | English |
| `…/ReservoirWebDemo/?lang=sk&explain=detailed#chaos` | Slovak, Detailed, straight into the chaos demo |

## Running it

**Online:** open the live link above. It works on desktop and mobile.

**Locally on Linux** (any distro: Debian/Ubuntu/Mint/LMDE, Arch/CachyOS,
Fedora/Bazzite, openSUSE, …):

```sh
git clone https://github.com/m1omg/ReservoirWebDemo.git
cd ReservoirWebDemo
./run.sh
```

`run.sh` starts a tiny local web server with Python 3, which almost every distro
ships, and opens <http://127.0.0.1:8000/> in your default browser. Pass a port
number if 8000 is taken: `./run.sh 9000`. Any other static file server works
too.

> Why not just double-click `index.html`? Browsers don't allow JavaScript
> modules or the microphone on `file://` pages, so a local server (or the hosted
> version) is needed.

**Browsers:** current Firefox, Chrome/Chromium, Edge and Safari. The speech demo
needs microphone permission and a secure origin (`https://`, `localhost` or
`127.0.0.1`).

## Privacy

Nothing leaves your device. Microphone audio is analysed inside the page and
never stored or uploaded, and the microphone switches off when you leave the
speech tab. Your own gestures and, only if you tick "remember", 24-number sound
summaries per 10 ms (not audio) are kept in your browser's `localStorage`. Each
demo has a button to clear them.

## Under the hood

- Plain JavaScript ES modules, HTML and Canvas. No dependencies, no build step,
  no CDN, so it also works offline.
- `js/core/`: the reservoir itself (`esn.js`: sparse random network scaled to a
  chosen spectral radius), ridge regression (`linalg.js`: streaming, cache-blocked
  normal equations, with an automatic primal/dual switch), the forecaster and the
  sequence classifier.
- `js/systems/`: chaotic systems (RK4 and a delay-equation integrator), gesture
  templates and preprocessing, and the wave-equation pond.
- `js/audio/`: FFT and mel filterbank, voice activity detection, and the
  AudioWorklet that delivers microphone samples.
- `js/demos/`: one file per demo.
- `js/content/en.js`, `js/content/sk.js`: every piece of on-screen text in both
  languages and both reading levels. Adding a language means adding one more
  table; `js/i18n.js` handles lookup, plurals and number formatting.
- **Refresh-rate independent:** every animation advances a fixed number of
  simulation steps per second of real time (`js/core/loop.js`), so a 60 Hz
  laptop and a 240 Hz monitor run at exactly the same speed. A test checks this
  at 30, 60, 75, 144 and 240 Hz.

## Development

```sh
npm test          # unit and behaviour tests (Node ≥ 22, no packages needed)
npm run e2e       # browser smoke test with screenshots (needs Playwright + Chromium)
```

The tests check the maths (ridge regression, spectral radius, echo-state
property) and that each demo actually works. Examples: Lorenz must be forecast
more than 5 Lyapunov times ahead while the linear model fails within 1, gestures
must reach at least 95% held-out accuracy, a new gesture must be learnt from 3
examples, the water must separate sine from square only with the nonlinear
sensor, and synthetic spoken words must be recognised from 5 examples.

## Publishing

`.github/workflows/pages.yml` deploys the site whenever `main` changes. Turn it
on once under **Settings → Pages → Build and deployment → Source: GitHub
Actions**. The site will then be at `https://<user>.github.io/ReservoirWebDemo/`.

## Further reading

- H. Jaeger & H. Haas (2004). [Harnessing nonlinearity: predicting chaotic systems and saving energy in wireless communication](https://doi.org/10.1126/science.1091277). *Science* 304, 78–80.
- W. Maass, T. Natschläger & H. Markram (2002). Real-time computing without stable states. *Neural Computation* 14(11).
- C. Fernando & S. Sojakka (2003). [Pattern recognition in a bucket](https://doi.org/10.1007/978-3-540-39432-7_63). *ECAL 2003*, LNCS 2801.
- J. Pathak, Z. Lu, B. R. Hunt, M. Girvan & E. Ott (2017). [Using machine learning to replicate chaotic attractors and calculate Lyapunov exponents from data](https://doi.org/10.1063/1.5010300). *Chaos* 27, 121102.
- P. R. Vlachas et al. (2020). [Backpropagation algorithms and reservoir computing in recurrent neural networks for the forecasting of complex spatiotemporal dynamics](https://doi.org/10.1016/j.neunet.2020.02.016). *Neural Networks* 126, 191–217.
- D. J. Gauthier, E. Bollt, A. Griffith & W. A. S. Barbosa (2021). [Next generation reservoir computing](https://doi.org/10.1038/s41467-021-25801-2). *Nature Communications* 12, 5564.
- M. Lukoševičius (2012). A practical guide to applying echo state networks. *Neural Networks: Tricks of the Trade*, LNCS 7700.

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
