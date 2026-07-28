# Growth reference data

The Excel workbooks in this directory are the only references used by the
backend's clinical HAZ and WHZ calculations. They contain WHO L, M, and S
parameters and are validated for required columns and table coverage when the
application starts.

`lhfa_children_0-to-5-years_lms.xlsx.b64` is a lossless Base64 representation of
the Excel workbook because the pull-request API rejects binary patches. The
service decodes it directly into memory and passes those original workbook bytes
to the Excel reader. It does not generate values or write a temporary fallback.

The legacy `who_haz_0_59m.csv`, `who_wfh_0_59m.csv`, and
`who_whz_reference.csv` files are retained solely as non-clinical research and
comparison fixtures for synthetic-data tooling. They must not be loaded by
`app/`, used for an assessment, or treated as a fallback when an Excel table does
not cover a measurement.
