| fix tested | run (file name) | parameters | best epoch | val loss (best) | val acc (best) | final train acc | final val acc | final gap | final val loss |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 64px_h256-128<br>_dropout0.0_wd0.0 | 3,179,525 | 2 | 1.506 | 36.7% | 96.2% | 38.1% | 58.1% | 4.039 |
| dropout 0.5 | 64px_h256-128<br>_dropout0.5_wd0.0 | 3,179,525 | 50 | 1.365 | 44.7% | 60.2% | 44.7% | 15.6% | 1.365 |
| dropout 0.5 + weight decay 1.0 (ridge) | 64px_h256-128<br>_dropout0.5_wd1.0 | 3,179,525 | 27 | 1.357 | 42.9% | 63.3% | 44.7% | 18.6% | 1.425 |
| smaller model (64/32) | 64px_h64-32<br>_dropout0.0_wd0.0 | 788,741 | 5 | 1.554 | 40.7% | 95.3% | 38.1% | 57.3% | 3.645 |
| weight decay 0.01 (ridge) | 64px_h256-128<br>_dropout0.0_wd0.01 | 3,179,525 | 2 | 1.544 | 37.6% | 95.8% | 37.2% | 58.6% | 3.963 |
| weight decay 0.1 (ridge) | 64px_h256-128<br>_dropout0.0_wd0.1 | 3,179,525 | 2 | 1.551 | 39.8% | 97.3% | 41.2% | 56.2% | 3.215 |
| weight decay 1.0 (ridge) | 64px_h256-128<br>_dropout0.0_wd1.0 | 3,179,525 | 2 | 1.499 | 39.4% | 93.8% | 43.8% | 50.0% | 2.271 |
