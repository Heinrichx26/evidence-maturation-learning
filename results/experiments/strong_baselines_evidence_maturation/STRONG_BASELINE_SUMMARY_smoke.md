# Strong baseline comparison

## event_aircraft_core

| Method | Micro-F1 | Macro-F1 | Samples-F1 | Jaccard | Runtime s |
|---|---:|---:|---:|---:|---:|
| SIHTC-adapted | 0.397 | 0.249 | 0.390 | 0.257 | 2.4 |
| VCLDL-adapted | 0.395 | 0.250 | 0.388 | 0.255 | 2.3 |
| BR-TFIDF | 0.387 | 0.249 | 0.380 | 0.248 | 2.8 |
| DyLas-adapted | 0.387 | 0.249 | 0.380 | 0.248 | 2.3 |
| HALB-adapted | 0.387 | 0.249 | 0.380 | 0.248 | 2.2 |
| LSPCL-adapted | 0.328 | 0.248 | 0.325 | 0.203 | 1.0 |
| MatchXML-adapted | 0.328 | 0.249 | 0.324 | 0.201 | 1.0 |

## event_aircraft_weather

| Method | Micro-F1 | Macro-F1 | Samples-F1 | Jaccard | Runtime s |
|---|---:|---:|---:|---:|---:|
| SIHTC-adapted | 0.380 | 0.243 | 0.374 | 0.243 | 2.4 |
| VCLDL-adapted | 0.378 | 0.246 | 0.371 | 0.240 | 2.3 |
| BR-TFIDF | 0.373 | 0.247 | 0.364 | 0.235 | 0.8 |
| DyLas-adapted | 0.373 | 0.247 | 0.364 | 0.235 | 2.4 |
| HALB-adapted | 0.373 | 0.247 | 0.364 | 0.235 | 2.3 |
| MatchXML-adapted | 0.320 | 0.231 | 0.320 | 0.200 | 1.2 |
| LSPCL-adapted | 0.298 | 0.232 | 0.297 | 0.182 | 1.2 |

## preliminary

| Method | Micro-F1 | Macro-F1 | Samples-F1 | Jaccard | Runtime s |
|---|---:|---:|---:|---:|---:|
| SIHTC-adapted | 0.512 | 0.387 | 0.504 | 0.361 | 2.3 |
| DyLas-adapted | 0.510 | 0.382 | 0.501 | 0.359 | 2.3 |
| VCLDL-adapted | 0.505 | 0.381 | 0.495 | 0.352 | 2.3 |
| BR-TFIDF | 0.504 | 0.381 | 0.496 | 0.352 | 6.0 |
| HALB-adapted | 0.504 | 0.381 | 0.496 | 0.352 | 7.0 |
| MatchXML-adapted | 0.445 | 0.303 | 0.450 | 0.318 | 9.1 |
| LSPCL-adapted | 0.434 | 0.317 | 0.439 | 0.306 | 9.3 |

## mature_factual

| Method | Micro-F1 | Macro-F1 | Samples-F1 | Jaccard | Runtime s |
|---|---:|---:|---:|---:|---:|
| SIHTC-adapted | 0.632 | 0.480 | 0.630 | 0.509 | 2.4 |
| BR-TFIDF | 0.625 | 0.477 | 0.621 | 0.500 | 7.4 |
| DyLas-adapted | 0.625 | 0.477 | 0.621 | 0.500 | 2.6 |
| HALB-adapted | 0.625 | 0.477 | 0.621 | 0.500 | 8.6 |
| VCLDL-adapted | 0.625 | 0.477 | 0.621 | 0.500 | 2.3 |
| LSPCL-adapted | 0.474 | 0.347 | 0.462 | 0.336 | 12.7 |
| MatchXML-adapted | 0.465 | 0.337 | 0.454 | 0.327 | 12.7 |
