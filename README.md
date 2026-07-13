# CA-AMT

Official research code for:

**Clinical-aware adaptive multimodal Transformer for patient-level detection of coronary artery disease using clinical variables, tongue images, and bilateral palm images**

## Project Overview

CA-AMT is a patient-level multimodal learning framework for coronary artery disease (CAD) detection. The study evaluates how conventional clinical risk variables and non-invasive surface image phenotypes from tongue and bilateral palm images can be integrated for CAD prediction.

The clinical rationale is that structured clinical variables provide established risk information, while body-surface images may contribute complementary visual phenotypes. Simple feature concatenation can combine these sources, but CA-AMT further improves fusion quality through patient-specific adaptive modality weighting and Transformer-based cross-modal interaction.

This repository provides the implementation used to train, evaluate, externally validate, and interpret the CA-AMT model and its comparators. Patient data, patient images, clinical spreadsheets, trained weights, and generated prediction files are not included.

## Highlights

- Patient-level CAD detection using clinical variables, tongue images, and left/right palm images.
- Clinical-aware multimodal architecture with a clinical MLP encoder and image encoders.
- Transformer fusion over patient-level modality tokens.
- Patient-specific adaptive modality weights for clinical and image modalities.
- Comparator workflows for clinical-only, image-only, and image-clinical concatenation baselines.
- External validation scripts for independent cohort assessment.
- Interpretability workflows for modality weights, token attention, Grad-CAM style visualization, and patient-level explanation cards.

## Framework Overview

The CA-AMT workflow follows four main stages:

1. Clinical variables are encoded by a lightweight MLP branch.
2. Left palm, right palm, and tongue images are encoded by convolutional image encoders.
3. The resulting patient-level tokens interact through a cross-modal Transformer.
4. A patient-specific adaptive gating module estimates modality weights before final CAD classification.

This design preserves the clinical main line of the manuscript: clinical variables provide traditional risk information, surface images provide complementary visual phenotype information, simple concatenation integrates both sources, and CA-AMT improves the fusion process through adaptive patient-level weighting and Transformer interaction.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- environment.yml
|-- LICENSE
|-- CITATION.cff
|-- clinical_only_lasso_xgboost_pycharm.py
|-- cv_train_patient_oof_camt_mlp_v1.py
|-- cv_train_patient_oof_concat_clinical.py
|-- external_validate_camt_mlp_v1.py
|-- make_camt_patient_explanation_cards.py
|-- make_camt_modality_ablation_and_token_attention_probability_shift_validation.py
|-- plot_figure2_table2_sci_5models_v4.py
|-- plot_figure3_external_main_and_table3_v4.py
|-- scr/
|   |-- datasets/
|   |-- models/
|   |-- train/
|   |-- explain/
|   `-- utils/
|-- training/
|-- evaluation/
|-- networks/
|-- interpretability/
`-- docs/
```

The `training/`, `evaluation/`, `networks/`, `interpretability/`, and `docs/` folders provide publication-facing workflow documentation. The executable research scripts are retained in their original locations to preserve import compatibility and avoid altering training logic.

## Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/ykdy159357-cell/CA-AMT.git
cd CA-AMT
conda env create -f environment.yml
conda activate ca-amt
```

Alternatively, install the Python dependencies with pip:

```bash
pip install -r requirements.txt
```

GPU training requires a PyTorch build compatible with the local CUDA driver. If the default install command does not match your CUDA version, install PyTorch and torchvision from the official PyTorch index first, then install the remaining packages from `requirements.txt`.

## Requirements

The main dependencies are:

- Python 3.8 or later
- PyTorch and torchvision
- NumPy, pandas, SciPy, scikit-learn
- Pillow and matplotlib
- XGBoost and SHAP for clinical-only modeling and interpretation
- openpyxl for spreadsheet export from figure/table scripts

The exact dependency list is provided in `requirements.txt` and `environment.yml`.

## Data Description

The repository expects a de-identified patient-level metadata table prepared outside version control. A typical table contains:

- `patient_uid`: de-identified patient identifier
- `cohort`: development or external cohort label
- `split`: train, validation, test, or external split label where applicable
- `label`: CAD label
- `hand_left_path`: local path to the left palm image
- `hand_right_path`: local path to the right palm image
- `tongue_path`: local path to the tongue image
- clinical variables such as age, sex, BMI, fasting glucose, lipid indices, smoking, alcohol use, hypertension, diabetes, dyslipidemia, chest pain, and chest tightness

Patient images and clinical tables must remain outside the public repository. The `.gitignore` file intentionally excludes local data folders, image files, clinical spreadsheets, trained weights, and generated outputs.

## Training

Primary CA-AMT cross-validation training:

```bash
python cv_train_patient_oof_camt_mlp_v1.py
```

Image-clinical concatenation comparator:

```bash
python cv_train_patient_oof_concat_clinical.py
```

Clinical-only LASSO logistic regression and XGBoost comparators:

```bash
python clinical_only_lasso_xgboost_pycharm.py
```

Before running, configure the input CSV path and output directory in the corresponding script. The scripts are intentionally kept close to the original experimental implementation to preserve the reported training protocol.

## External Validation

External validation is performed without model retraining or threshold tuning:

```bash
python external_validate_camt_mlp_v1.py
python external_validate_concat_clinical.py
python external_validate_patient.py
```

The CA-AMT external validation script loads the fold-specific development models and clinical preprocessors, evaluates the independent external cohort once, and exports ensemble-level predictions and metrics.

## Interpretability

Patient-level interpretability is supported through:

- adaptive modality weights for left palm, right palm, tongue, and clinical tokens
- Transformer token attention summaries
- probability shift and ablation analyses
- Grad-CAM style image visualization scripts
- patient-level explanation cards integrating modality and clinical-factor information

Representative commands:

```bash
python make_camt_modality_ablation_and_token_attention_probability_shift_validation.py
python make_camt_patient_explanation_cards.py
python Make_Figure4_Modality_Attention_AB.py
```

Interpretability outputs may contain patient-level identifiers or image derivatives and should not be committed unless they are fully de-identified and approved for public release.

## Citation

If you use this code, please cite the manuscript and repository. Citation metadata is provided in `CITATION.cff`. The DOI is currently a placeholder and should be updated after publication or archival release.

## License

This project is released under the MIT License. See `LICENSE` for details.

