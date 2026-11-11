# Data access and release scope

- MIMIC-IV v3.1: https://physionet.org/content/mimiciv/3.1/ ; DOI 10.13026/kpb9-mt58.
- eICU-CRD v2.0: https://physionet.org/content/eicu-crd/2.0/ ; DOI 10.13026/C2WM1R.

Access requires the applicable PhysioNet credentialing, training and data-use agreement. This repository conveys no data authorization or institutional ethics approval.

Keep raw files, patient IDs, patient-level feature/assignment tables, learned medoid values, fitted transformations and caches outside version control. MIMIC's official guidance treats derived datasets/models as sensitive and routes their sharing through PhysioNet under the source agreement. A medoid represents an actual training patient, so omitting direct identifiers does not make its values a freely distributable artifact.

Code contains clinical field names and item IDs because they define extraction. It contains no released patient-level observations. Clinical access prerequisites cannot be bypassed by running synthetic examples. Use an authorized local copy and keep all generated clinical outputs private.
