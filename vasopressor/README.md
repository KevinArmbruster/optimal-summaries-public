
# Vasopressor

All scripts specifically for MIMIC data.

[Top-level README](../README.md)

## Data Source

* The MIMIC-III database can be downloaded [HERE](https://physionet.org/content/mimiciii/1.4/).
* Follow the steps of [MIMIC Export](https://github.com/MLforHealth/MIMIC_Extract) to export the database to CSV.
    * Use [my fork](https://github.com/KevinArmbruster/MIMIC_Extract) with an existing Dockerfile. 
    * Run ```mimic_direct_extract.py ``` as instructed.
    * Creates `all_hourly_data.h5` (among others).
* Use the [Preprocessing Notebook](/optimal-summaries-public/vasopressor/preprocess_MIMIC_data.ipynb) to create necessary files for [data.py](/optimal-summaries-public/models/data.py).

## Scripts

* Preprocessing Notebook to use after MIMIC Extract
* Notebooks where I explored the various settings
