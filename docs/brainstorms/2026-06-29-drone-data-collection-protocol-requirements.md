---
date: 2026-06-29
topic: drone-data-collection-protocol
---

# Drone Data Collection Protocol — Cogongrass Training Data

## Summary

A field protocol for collecting cogongrass training imagery with the DJI Mavic 3M.
Each mission is a **nadir (straight-down) mapping flight at 30 ft AGL** capturing
**both RGB and multispectral**, flown with high overlap so frames stitch into a
georeferenced orthomosaic. Several sites are flown repeatedly across dates and
conditions. Ground truth is captured two ways: GPS-marked in the field during the
flight, then refined on the stitched map afterward. This document is the handoff
to whoever organizes and executes the collection.

## Problem Frame

The hard problem in this project is not classifying one flight — it is
generalizing to a flight the model has never seen. The existing tracks already
train on one collection (2026-06-06) and test on an entirely held-out one
(2026-04-22), and accuracy drops across that gap because the two flights differ
in lighting, color, and capture geometry, not just in where the grass is.

That makes the collection protocol itself the highest-leverage lever we have. If
every flight is captured the same way — same altitude, same angle, same lighting
window, same calibration, same georeferencing — then the differences the model
sees between flights are real ground variation it should learn, not equipment and
weather drift it should ignore. A loose protocol bakes domain shift into the data
before any model touches it. The whole document is organized around one rule:
**lock every variable that can be locked.**

Note one pivot this implies. The current models are trained on **oblique**
frames; this protocol collects **nadir**. That is deliberate — nadir stitches
cleanly, georeferences easily, and supports map-based labeling — but it means the
detector is retrained on nadir data and deployed on nadir imagery going forward.
Train-on-what-you-deploy-on only holds if future field inference is also nadir.

## Key Decisions

- **Nadir capture, pipeline pivots with it.** Camera points straight down.
  Existing oblique models are retrained on the new data; deployment imagery must
  also be nadir for the training distribution to match.
- **30 ft AGL, pending a calibration test.** Fixed low altitude gives very fine
  detail useful for species discrimination. The trade-off (below) is real enough
  that one short test flight should confirm it before the full campaign commits.
- **Stitch-grade overlap, even though this is training data.** High overlap isn't
  only for the map — it lets cogongrass be labeled once on the orthomosaic and
  projected back to individual frames, and gives redundant looks at each patch.
- **Consistency across flights outranks per-flight quality.** When a choice trades
  a nicer single map against a repeatable routine, repeatability wins — the data's
  value is in being comparable across dates.

## Requirements

### Flight configuration

- R1. Every mission flies **nadir** (gimbal straight down, 0°).
- R2. Flight altitude is **30 ft AGL (~9 m)**, held constant for all flights at a
  given site once confirmed by the calibration test (R15).
- R3. Each flight captures **both RGB and multispectral** on the same pass.
- R4. Frontlap (along-track) is **≥ 80%** and sidelap (across-track) is **≥ 75%**;
  target **85% / 80%** for the feature-poor uniform grass canopy, where low
  texture and the lower-resolution multispectral sensor make stitching fail at
  ordinary overlap.
- R5. The mission is flown from a **saved flight plan** in DJI's mapping app, and
  the *same saved plan* is reused for that site on every subsequent date.

### Multispectral integrity

- R6. A **reflectance calibration panel** is captured at a consistent distance
  **before and after every flight**, with the captures stored alongside the
  flight's imagery.
- R7. The **sunlight/irradiance sensor (DLS)** on top of the aircraft stays
  unobstructed for the whole flight.
- R8. Camera and exposure settings (R9) are identical across the RGB and
  multispectral capture and unchanged between flights at a site.

### Consistency across flights (the spine)

- R9. Capture settings are fixed and recorded once, then reused: exposure mode,
  flight speed, capture interval/overlap, altitude, gimbal angle.
- R10. All flights at a site happen inside the **same lighting window** — solar
  noon ± ~2 hours — under a **consistent sky** (clear, or uniform overcast; never
  broken/patchy cloud that flickers shadow across the canopy mid-flight).
- R11. Flights are skipped or rescheduled outside defined limits for **wind, rain,
  and wet canopy** (thresholds set in the organizer's checklist).
- R12. Each flight is **georeferenced repeatably** so the same ground point lands
  at the same coordinate across dates — via the **RTK module**, or ground control
  points if RTK is unavailable.

### Ground truth

- R13. During each flight, a field worker **GPS-marks cogongrass stands and clear
  negative areas** (bare ground, other species, water) with photos.
- R14. After stitching, an expert **refines cogongrass regions on the
  orthomosaic**, reconciled against the field marks. Labels are treated as
  **per-flight** — the same site re-flown later is re-labeled, because the grass
  changes between dates.

### Validation & logging

- R15. Before the full campaign, one **calibration test flight** confirms 30 ft +
  the chosen overlap actually stitches cleanly and resolves cogongrass; altitude
  or overlap is adjusted from its result.
- R16. Every flight produces a **metadata log entry**: date, time, site, pilot,
  sky/weather, wind, temperature, altitude, overlap, panel-capture confirmation,
  battery sets, and any anomalies.
- R17. Imagery is delivered in a **consistent folder structure** keyed by
  site and date, with RGB, multispectral, panel captures, GPS marks, and the log
  together per flight.

## Per-flight field procedure

- F1. Single mapping flight at one site
  - **Pre-flight:** confirm weather inside limits (R10, R11); power up; confirm
    RTK fix or GCPs placed (R12); load the site's saved plan (R5); confirm
    nadir + altitude + overlap settings (R1, R2, R4).
  - **Calibration:** capture the reflectance panel (R6); confirm the DLS is clear
    (R7).
  - **Fly:** execute the saved plan unattended; a field worker simultaneously
    GPS-marks cogongrass stands and negatives with photos (R13).
  - **Close-out:** capture the reflectance panel again (R6); fill the metadata log
    (R16); offload imagery into the per-flight folder structure (R17).
  - **Later (off-site):** stitch the orthomosaic; expert refines labels on the map
    against the field marks (R14).

## Equipment & readiness

The organizer should confirm availability before scheduling:

- Mavic 3M with **RTK module** (or a plan for ground control points).
- DJI **reflectance calibration panel**.
- Enough **batteries** for low-altitude flights (30 ft footprint is small, so a
  site needs many passes — see Risks).
- DJI's **mapping flight app** and a **stitching/orthomosaic tool** (e.g. DJI
  Terra or equivalent) — confirm exact app and version, since the multispectral
  workflow and panel calibration must be supported end to end.

## Scope Boundaries

- Model training, tiling, and heatmap inference are **out of scope** here — this
  document covers data acquisition and labeling handoff only.
- Real-time/onboard detection during flight is out of scope; collection is for
  offline training data.
- Oblique capture is explicitly **not** part of this protocol (see the Key
  Decisions pivot).

## Dependencies / Assumptions

- Assumes future field **deployment will also be nadir** at the same altitude; if
  deployment stays oblique, this training data won't match the deployment
  distribution.
- Assumes the chosen app supports a **saved, repeatable mapping plan** and the
  Mavic 3M multispectral + panel calibration workflow.
- Assumes target sites have **known cogongrass presence** and accessible negatives
  for ground-truthing.
- Assumes someone with **cogongrass identification expertise** is available both
  in-field (R13) and for map refinement (R14).

## Risks

- **30 ft is low for area mapping.** A small footprint means many photos, long
  flights, and more battery swaps per site; over uniform grass, low altitude also
  gives each frame less spatial context, which can make stitching harder. R15's
  test flight exists to catch this before the campaign scales.
- **Patchy cloud is the silent consistency killer.** Shadows moving across the
  canopy mid-flight create per-flight artifacts the model can mistake for ground
  signal — hence the strict sky requirement (R10), which is easy to relax under
  schedule pressure and shouldn't be.

## Outstanding Questions

### Resolve before planning the campaign

- Specific **wind / temperature / time-of-day thresholds** for the go/no-go
  checklist (R10, R11).
- Which **georeferencing method** is actually available — RTK module on hand, or
  GCPs (R12)?
- The exact **app and stitching tool** chain, confirmed to support Mavic 3M
  multispectral + panel calibration.

### Deferred to execution

- Final altitude and overlap numbers, set from the R15 calibration test.
- The concrete **site list** and the **revisit cadence** (how often each site is
  re-flown).
- Folder-structure and file-naming convention details (R17).
