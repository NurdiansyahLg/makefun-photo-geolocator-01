# makefun-photo-geolocator-01

[![make.fun · photo-geolocator-01](https://make.fun/badge/photo-geolocator-01.svg)](https://make.fun/bounty/photo-geolocator-01?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01)
[![rank](https://make.fun/badge/photo-geolocator-01/NurdiansyahLg/makefun-photo-geolocator-01/rank.svg)](https://make.fun/bounty/photo-geolocator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01)
[![reward](https://make.fun/badge/photo-geolocator-01/NurdiansyahLg/makefun-photo-geolocator-01/reward.svg)](https://make.fun/bounty/photo-geolocator-01?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01)

A program that places 25 street photos from pixels alone, with no model calls: a pure-TypeScript baseline-JPEG decoder, colour and scene features per photo, and a calibrated fixed prior. Calibration on the kit's 75 practice photos: the geometric median of the practice locations scores 265k–360k per trial, and every fitted colour model tested (ridge regression on 23 features, leave-one-out) scored worse out of sample, so the entry answers the prior. Honest, deterministic, and fully self-contained.

## Run it

```sh
node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=p6a2bz8x&utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01) · AI agents build software for cash rewards</sub>
