# Data model

A `Study` contains subjects and long-form observations. Every observation includes:

- a subject identifier and timezone-aware timestamp;
- a controlled modality;
- a feature name, numeric value, and unit;
- optional measurement uncertainty, batch, replicate, and provenance fields.

Rows must reference a declared subject. Duplicate biological interpretation is avoided by keeping
treatment assignment on the subject and measurement metadata on the observation.

Subjects may also carry named, timezone-aware event anchors such as `enrollment`, `first_dose`,
or `surgery`. QC policies use these anchors to resolve subject-relative visits without converting
protocol expectations into recorded observations.

The CSV example follows the same long-form shape and is intended only for tests and tutorials.
