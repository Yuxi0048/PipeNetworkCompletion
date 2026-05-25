# Data Publication and License Audit

Workstream: Codex

Date: 2026-05-24

Status: research data release guidance, not legal advice

## Bottom Line

Do not publish the current full `data/` directory as-is.

The surface/context layers are mostly publishable or reproducible from public
open-data sources with attribution. The utility/sewer truth layers and anchor
layers should be treated as restricted until the original Urban Utilities terms
are explicitly confirmed.

Recommended public release shape:

- publish source code, configs, scripts, tests, and documentation;
- publish synthetic sample data;
- publish instructions/scripts to download open context layers from official
  providers;
- publish aggregate metrics and non-sensitive plots;
- do not publish raw sewer shapefiles, manhole points, service laterals, raw
  utility anchors, processed truth GeoJSON, processed graph pickles containing
  utility geometry, or predicted/decoded utility GeoJSON layers unless written
  permission or a clear public redistribution licence is confirmed.

## Local Data Found

The repository contains these major data families:

- `data/raw/gis/sewer/`: sewer assets, including gravity mains, pressure mains,
  manholes, services, valves, vents, pumps, fittings, and treatment/pump
  facilities.
- `data/raw/gis/roads/`: road shapefile bundle.
- `data/raw/mh_road/`: manhole-road near table.
- `data/interim/*.pkl`: processed paper-pipeline data.
- `data/experiments/*.pkl`: experiment graph/data pickles.
- `data/processed/aois/`: AOI clips, roads, buildings, built-up areas, sewer
  truth, watercourses, generated OSM building-point truth configs.
- `data/raw/context/` and `data/processed/context/`: building, DEM,
  watercourse, OSM/Overture-derived, and processed context layers.

The local sewer metadata includes fields such as `ASSETID`, `OWNER`, and QUU
owner-domain values. It also contains ArcGIS export metadata from a local
utility-data workspace. Treat these files as utility asset data, not ordinary
open context layers.

## Source and License Evidence

### Queensland Government Open Data

Queensland Government states that, unless otherwise noted, copyright material
available through its website is licensed under Creative Commons Attribution
4.0 International. It also warns that some material may have other licences or
third-party rights.

Source:

https://www.qld.gov.au/legal/copyright

The Queensland open-data FAQ says open data can be used, modified, and shared,
but each dataset has its own licence; most are open, but some can be more
restrictive.

Source:

https://www.data.qld.gov.au/article/standards-and-guidance/faq

Implication:

Queensland roads, DEM, cadastre, hydrography, and topographic building layers
are likely publishable if their specific dataset/resource pages say CC BY 4.0.
Each layer still needs dataset-level attribution.

### Brisbane City Council Open Data

Brisbane City Council says Council material is licensed under Creative Commons
Attribution 4.0 unless otherwise indicated, excluding insignia, branding,
trademarks, and third-party material.

Sources:

https://www.brisbane.qld.gov.au/about-council/governance-and-strategy/privacy-and-legal

https://data.brisbane.qld.gov.au/terms/terms-and-conditions/

Implication:

Brisbane waterway/drainage/open-space/park/city-plan open-data layers are
probably publishable with attribution if downloaded from the BCC open-data
portal and if no dataset-specific exception appears.

### OSM

OpenStreetMap data is licensed under ODbL. OSM requires attribution, and if a
database is altered or built upon, distribution of the result may require the
same licence.

Source:

https://www.openstreetmap.org/copyright

Implication:

OSM building footprints/centroids can be used, but public release of an
OSM-derived database, feature table, or candidate graph should carry ODbL
attribution and may trigger share-alike obligations. For the cleanest release,
publish scripts that re-download OSM/Overture data rather than bundling large
OSM-derived layers.

### Overture Maps

Overture uses permissive licences where possible, but some themes include OSM
and therefore carry ODbL. Overture states that licensing varies by theme and
source, and users should consult the attribution page.

Sources:

https://overturemaps.org/about/faq/

https://docs.overturemaps.org/attribution/

Implication:

Overture-derived buildings may still carry ODbL depending on theme/source.
Treat them similarly to OSM unless the exact release/theme attribution page
shows a permissive licence for the downloaded subset.

### DEM

The GA SRTM 1 second DEM is listed as Creative Commons Attribution 4.0
International.

Source:

https://docs.dea.ga.gov.au/data/external-data/ga-srtm-1-second-dem/

Implication:

DEM-derived features such as elevation, slope, low-ground index, and flow
accumulation can be published with attribution, assuming the exact DEM source
is GA SRTM/DEM-H or another open DEM with compatible terms.

### Queensland Building/Settlement Context

The Queensland Buildings and Settlements service lists building points,
homesteads, building areas, and built-up areas and identifies copyright as
State of Queensland Department of Resources.

Source:

https://spatial-gis.information.qld.gov.au/arcgis/rest/services/Structure/BuildingsAndSettlements/MapServer

Implication:

These appear to be public Queensland topographic layers, but the publication
package should still cite the corresponding QSpatial/data.qld.gov.au dataset
licence, preferably CC BY 4.0, before bundling the raw GPKG extracts.

### Urban Utilities Sewer Data

Urban Utilities exposes an open GIS map for water and sewerage infrastructure.
The public pages and ArcGIS item found during the audit identify the map as open
GIS asset information, but I did not find a clean CC BY/ODbL-style licence text
for bulk redistribution of the raw sewer asset layers.

Sources found:

https://www.urbanutilities.com.au/about-us/who-we-are/asset-gis-information

https://www.urbanutilities.com.au/development/our-services/research-my-land

https://www.arcgis.com/home/item.html?id=36fdac21178a4364a04f9516aa0703e5

Implication:

Treat sewer truth and sewer asset layers as not redistributable until one of
these is obtained:

- a dataset page with an explicit open licence allowing redistribution and
  derivative use;
- written permission from Urban Utilities;
- a release agreement from the original project data provider.

This applies especially to:

- `data/raw/gis/sewer/*`;
- `data/raw/mh_road/*`;
- `data/interim/*` derived from sewer/manhole assets;
- `data/experiments/*.pkl` graph data containing sewer/manhole geometry or
  asset attributes;
- `data/processed/aois/**/utility_truth*.geojson`;
- output prediction GeoJSONs that reveal utility alignment hypotheses;
- plots that visibly overlay sewer truth on identifiable geography.

## Publication Decision Table

| Data family | Examples | Public release recommendation | Risk |
|---|---|---:|---:|
| Source code and configs | `pipe_network_completion/`, `scripts/`, `configs/` without secrets | Publish | Low |
| Synthetic smoke data | Generated grids/buildings/truth | Publish | Low |
| Queensland/BCC road, watercourse, DEM, building context | roads, drainage, DEM, building areas, built-up areas | Publish if dataset-level licence is recorded; otherwise publish download scripts | Low/Medium |
| OSM/Overture buildings | building footprints/centroids | Prefer reproducible download scripts; if bundling, include ODbL/Overture attribution and licence notes | Medium |
| Cadastre/parcels | Queensland DCDB | Publish only after confirming dataset-specific terms; otherwise publish download instructions | Medium |
| Raw Urban Utilities sewer assets | mains, manholes, services, pumps, valves | Do not publish without explicit licence/permission | High |
| Processed sewer truth AOIs | `utility_truth*.geojson` | Do not publish without explicit licence/permission | High |
| Anchor-based graph pickles | manhole graph, MH-road table, line labels | Do not publish without explicit licence/permission | High |
| Decoded/predicted utility GeoJSON | predicted sewer alignments | Do not publish as public data; publish metrics instead | High |
| Paper figures showing exact truth | AOI overlays with truth mains | Avoid or generalize/anonymize unless permission confirmed | Medium/High |

## Recommended Public Release Structure

Use a slim public repository:

```text
PipeNetworkCompletion/
  pipe_network_completion/
  scripts/
  configs/templates/
  tests/
  docs/
  data/synthetic/
  data/README.md
  data_license_manifest.csv
```

Do not include:

```text
data/raw/gis/sewer/
data/raw/mh_road/
data/interim/
data/experiments/*.pkl
data/processed/aois/
data/processed/context/    # unless each layer has confirmed licence
outputs/*.geojson          # if it reveals utility locations
```

For reproducibility, provide:

- scripts to download open context layers;
- scripts to process AOIs after the user supplies licensed sewer truth;
- synthetic test data;
- metrics tables without raw geometry;
- clear note that the sewer truth cannot be redistributed by the authors unless
  permission is obtained.

## Suggested `data_license_manifest.csv` Columns

```text
layer_name
local_path_pattern
source_url
provider
license
redistribution_allowed
derivative_allowed
ai_training_allowed
attribution_required
share_alike_required
publish_raw
publish_processed
used_for_training
used_for_evaluation_only
risk
notes
```

## Recommended Paper / README Language

Suggested wording:

```text
The code, synthetic examples, and scripts for generating public surface-context
features are released. Utility asset geometries used as labels/evaluation data
are not redistributed because redistribution rights for the source utility
asset data were not confirmed. Users should obtain any utility asset data from
the relevant infrastructure owner or public GIS service under its applicable
terms.
```

For OSM/Overture:

```text
OpenStreetMap/Overture-derived context features are used subject to their
respective open-data licences. Where possible, the repository provides scripts
to reproduce these inputs from the original services rather than bundling
derived databases.
```

For Queensland/BCC open data:

```text
Queensland Government and Brisbane City Council context layers are used under
their published open-data terms, generally Creative Commons Attribution 4.0
unless a dataset-specific licence states otherwise. Attribution is retained in
the data manifest.
```
