---
name: Geospatial research
description: Design and validation guide for spatial joins, geographic dependence, and map-based inference.
when_to_use: The research question uses coordinates, administrative areas, spatial joins, distance, neighborhoods, maps, or geographic clustering.
---
# Geospatial research

## Define the spatial representation

Declare the coordinate reference system, geographic unit, boundary vintage,
spatial join predicate, and treatment of points on borders. Check latitude and
longitude bounds and never compute planar distances on unprojected degrees when
the scale makes that inaccurate. Administrative areas change over time; a name
match is not a stable geographic key.

## Assumptions and diagnostics

Inspect spatial autocorrelation, uneven support, edge effects, modifiable areal
unit sensitivity, and geographic missingness. Standard independent-error models
can understate uncertainty when nearby observations share shocks. For prediction,
validate with spatial blocks rather than random rows.

## Validation checklist

- Verify CRS, axis order, validity, and boundary vintage.
- Quantify unmatched and multiply matched spatial joins.
- Check results under at least one defensible geographic aggregation or radius.
- Use spatially appropriate uncertainty or clustering.
- Suppress small-area outputs under the disclosure policy.
- Treat maps as analytical outputs with the same evidence and limitation rules.
