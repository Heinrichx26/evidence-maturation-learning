# Strong baseline comparison

## event_aircraft_core

| Method | Micro-F1 | Macro-F1 | Samples-F1 | Jaccard | Runtime s |
|---|---:|---:|---:|---:|---:|
| VCLDL-adapted | 0.423 | 0.231 | 0.418 | 0.284 | 12.7 |
| SIHTC-adapted | 0.422 | 0.233 | 0.417 | 0.282 | 12.8 |
| DyLas-adapted | 0.414 | 0.227 | 0.408 | 0.274 | 16.9 |
| HALB-adapted | 0.412 | 0.228 | 0.406 | 0.272 | 12.9 |
| BR-TFIDF | 0.405 | 0.227 | 0.399 | 0.266 | 4.8 |
| LSPCL-adapted | 0.362 | 0.212 | 0.358 | 0.231 | 4.1 |
| MatchXML-adapted | 0.351 | 0.210 | 0.350 | 0.224 | 4.2 |

## event_aircraft_weather

| Method | Micro-F1 | Macro-F1 | Samples-F1 | Jaccard | Runtime s |
|---|---:|---:|---:|---:|---:|
| SIHTC-adapted | 0.431 | 0.233 | 0.424 | 0.290 | 12.2 |
| VCLDL-adapted | 0.428 | 0.234 | 0.420 | 0.286 | 12.4 |
| HALB-adapted | 0.417 | 0.232 | 0.409 | 0.275 | 13.6 |
| DyLas-adapted | 0.414 | 0.236 | 0.407 | 0.273 | 18.0 |
| BR-TFIDF | 0.406 | 0.235 | 0.399 | 0.266 | 4.1 |
| LSPCL-adapted | 0.362 | 0.207 | 0.365 | 0.238 | 6.2 |
| MatchXML-adapted | 0.356 | 0.201 | 0.359 | 0.233 | 6.0 |

## preliminary

| Method | Micro-F1 | Macro-F1 | Samples-F1 | Jaccard | Runtime s |
|---|---:|---:|---:|---:|---:|
| VCLDL-adapted | 0.544 | 0.304 | 0.534 | 0.401 | 43.8 |
| DyLas-adapted | 0.542 | 0.321 | 0.531 | 0.398 | 103.4 |
| SIHTC-adapted | 0.542 | 0.318 | 0.532 | 0.398 | 43.8 |
| BR-TFIDF | 0.534 | 0.318 | 0.524 | 0.388 | 35.7 |
| HALB-adapted | 0.533 | 0.316 | 0.525 | 0.388 | 73.7 |
| LSPCL-adapted | 0.483 | 0.266 | 0.473 | 0.337 | 59.9 |
| MatchXML-adapted | 0.476 | 0.256 | 0.464 | 0.330 | 59.8 |

## mature_factual

| Method | Micro-F1 | Macro-F1 | Samples-F1 | Jaccard | Runtime s |
|---|---:|---:|---:|---:|---:|
| SIHTC-adapted | 0.681 | 0.438 | 0.671 | 0.558 | 53.8 |
| VCLDL-adapted | 0.681 | 0.441 | 0.669 | 0.558 | 53.7 |
| DyLas-adapted | 0.677 | 0.453 | 0.666 | 0.552 | 135.3 |
| BR-TFIDF | 0.675 | 0.452 | 0.665 | 0.550 | 45.7 |
| HALB-adapted | 0.675 | 0.452 | 0.665 | 0.550 | 92.4 |
| LSPCL-adapted | 0.524 | 0.304 | 0.488 | 0.360 | 80.7 |
| MatchXML-adapted | 0.498 | 0.292 | 0.471 | 0.342 | 81.7 |
