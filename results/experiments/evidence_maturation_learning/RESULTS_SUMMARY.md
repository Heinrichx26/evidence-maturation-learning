# Evidence maturation learning results

## Data and validation

- Unit of analysis: NTSB event-aircraft records.
- Period: 2020-2025.
- Label space: level-2 final finding taxonomy.
- Forward validation: train on earlier years, tune thresholds on the most recent training year, and test on 2023, 2024, and 2025.
- Leakage control: probable-cause fields, finding fields, and cause/finding trigger words are excluded from model inputs.

## Stage-level performance

| Evidence stage | Micro-F1 | Macro-F1 | Samples-F1 | Jaccard | Mean entropy |
|---|---:|---:|---:|---:|---:|
| Event-aircraft core | 0.405 | 0.227 | 0.399 | 0.266 | 0.684 |
| Event-aircraft plus weather | 0.406 | 0.235 | 0.399 | 0.266 | 0.691 |
| Preliminary evidence | 0.534 | 0.318 | 0.524 | 0.388 | 0.616 |
| Mature factual evidence | 0.675 | 0.452 | 0.665 | 0.550 | 0.551 |

## Evidence acquisition effects

| Added evidence type | Delta micro-F1 | Delta macro-F1 | Delta samples-F1 | Uncertainty reduction |
|---|---:|---:|---:|---:|
| Weather structured fields | 0.001 | 0.007 | -0.000 | -0.007 |
| Preliminary narrative | 0.128 | 0.083 | 0.125 | 0.075 |
| Mature factual narrative | 0.142 | 0.135 | 0.141 | 0.066 |

## Maturation classes

- Early-closable: `Personnel issues-Task performance`, `Aircraft-Aircraft oper/perf/capability`, and `Aircraft-Aircraft power plant`.
- Late-emerging: `Aircraft-Fluids/misc hardware`, `Environmental issues-Conditions/weather/phenomena`, `Aircraft-Aircraft systems`, `Environmental issues-Physical environment`, and `Personnel issues-Psychological`.
- Unstable or low-support classes require careful treatment in manuscript tables.

## Result assessment

The preliminary-to-final framing is empirically usable on the current data. The preliminary evidence stage substantially improves over structured accident metadata, and mature factual evidence adds a second large gain. This supports the evidence maturation gap as a measurable target: some final findings are already learnable from early evidence, while hardware, systems, environmental, and psychological finding groups often require later factual evidence.

Weather structured fields alone do not reduce uncertainty in this setup. This result should be written as an evidence-prioritization finding: adding generic weather metadata is weaker than adding narrative investigation evidence for this final finding taxonomy.
