# Data quality and preparation results

These are actual preprocessing diagnostics. They are not trained-model accuracy or verified physical ground truth.

| Partition | Trips | GPS observations | Candidate rows | Eligible observations | Training candidate rows |
|---|---:|---:|---:|---:|---:|
| train | 80 | 208,417 | 1,084,832 | 143,533 | 755,694 |
| student_test | 10 | 19,160 | 113,639 | 14,203 | 76,234 |

## Observed data issues

- All 90 trip inputs passed the strict shape, finite-coordinate, coordinate-range, timestamp ordering and node/arc reference checks. Zero rejected trips are recorded in validation_errors.json.
- Training contains 208,417 GPS observations, student testing 19,160. No trip IDs overlap between the two partitions.
- The largest training timestamp gap is 153,064 seconds in trip 00000090; the largest student-test gap is 210 seconds in trip 00000065. Gaps over 60 seconds split feature segments. This threshold is a documented initial choice.
- There are 13 nonjoining consecutive reference-arc pairs across training and one across student testing. The reference sequence is therefore not assumed to be perfectly connected.
- Road networks include multiple weak components. Raw GPS rows are preserved, including observations with no generated candidates.
- Coordinates span multiple countries and hemispheres, so each trip uses a local metric projection rather than UTM zone 32N globally.

## Approximate label coverage

### train
- Alignment statuses: {"aligned": 144443, "ambiguous": 40981, "unmatched": 19992, "distant": 2993, "restart": 8}.
- 18,230 observations have no inference candidates.
- 4,715 misses among 188,425 derived known labels (2.50%). This denominator excludes unknown reference labels and must be reported alongside coverage.
- 143,533 observations (68.87% of raw observations) are retained in the convenience labelled training table.

### student_test
- Alignment statuses: {"aligned": 14215, "ambiguous": 4918, "distant": 23, "unmatched": 3, "restart": 1}.
- 0 observations have no inference candidates.
- 82 misses among 19,157 derived known labels (0.43%). This denominator excludes unknown reference labels and must be reported alongside coverage.
- 14,203 observations (74.13% of raw observations) are retained in the convenience labelled training table.

## Interpretation

Ambiguous, distant, restart and unmatched observations remain in the observation/reference tables, but are excluded from the convenience fitting table. Member 2 must not report fitting-subset metrics as whole-trip accuracy. The student-test fitting-format file is for convenience of evaluation code, not permission to train on student testing. Alignment assumptions and the organiser-approved metric definition remain essential to interpreting results.

Generated outputs were checked for one positive candidate per eligible observation, unique observation/candidate keys, candidate radius limits, projection fractions, and matching observation counts. Read reports/member1/README.md for the interfaces, feature allowlist and method limitations.

## Trip 00000054 requires attention

This training trip accounts for 16,519 of the 18,230 observations with no candidate inside 50m. Its nearest-road distance has median 84.2m and 90th percentile 96.9m, calculated against its entire supplied road graph. Increasing candidate count alone cannot resolve missing road coverage at this radius. These observations remain in outputs; they are not silently removed. Investigate raw graph/trajectory coverage with organisers before changing the radius or treating this trip's missing labels as classification errors.
