# Project Challenges

## Irradiance UTC alignment

The solar irradiance source uses `Timestamp_UTC`, while the generation and weather files use local campus timestamps. The first pipeline version parsed `Timestamp_UTC` as a naive timestamp and merged it directly on `CampusKey` and `Timestamp`. This shifted GHI into the wrong part of the day: before correction, average `Ghi` peaked around hour 1 while `SolarGeneration` peaked around hour 12, producing a negative `SolarGeneration` to `Ghi` correlation of about `-0.296`.

The pipeline now converts irradiance timestamps from UTC to `Australia/Melbourne` local time in `DataTransformer.transform_irradiance` before merging. Because local daylight-saving transitions can create repeated timestamps, `DataMerger.resample_irradiance` also averages duplicate `CampusKey` and `Timestamp` rows before interpolating to 15-minute intervals. After this correction, both GHI and solar generation peak around hour 12, and the overall `SolarGeneration` to `Ghi` correlation becomes positive at about `0.549`.

## Incomplete site metadata

The site metadata table has substantial missing technical information. Fields such as `kWp`, `Number of panels`, panel details, inverter details, optimizer details, and related equipment metadata are not consistently available across campuses. Earlier profiling showed `kWp` and `Number of panels` missing for about 40% of site rows, with the most complete technical metadata concentrated in the Bundoora campus.

This limits the reliability of capacity-normalized comparisons such as `GenerationPerkWp`, especially when comparing campuses or sites with missing installed capacity. The cleaning pipeline keeps a `has_capacity_data` flag so downstream EDA and modeling can distinguish rows with known capacity from rows where technical metadata is unavailable. Equipment-detail columns with sparse coverage are dropped from the cleaned site table to avoid carrying high-missingness fields into the master dataset.
