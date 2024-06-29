## Project: Underground Utility Network Completion

This document describes the code, data, and results for the project "Underground Utility Network Completion based on Spatial Contextual Information of Ground Facilities and Utility Anchor Points using Graph Neural Networks".

### Main Program

* **File:** `Inductive.ipynb` (Contains functional but ugly code)

**Notes:**

* The code for different architecture variations is not saved. Refer to the commented sections within the notebook to reproduce them.
* You will need to manually modify file paths and names to save models and results on your local machine.

### Results

* **File:** `model_metricsxxx.xlsx` (Contains model performance metrics)

### Trained Models (Checkpoints)

* **Location:** `/Models/` (Directory containing trained model checkpoints)

### Dataset

* **Cleaned Data:** `/Data/` (Directory containing cleaned datasets)
* **Preprocessed Data:**
    * `MH_proc.pkl`
    * `Road_proc.pkl`
    * `MH_R_RL_proc.pkl`
    * `Line_proc.pkl`
    * `R_R_proc.pkl`
    * `split_mask.pkl` (Optional mask for train/validation/test split, needed for reproducing results)
* **Raw Data:**
    * `/Sewer_shp/SewerManholes_ExportFeatures.shp`
    * `/Sewer_shp/SewerGravityMa_ExportFeature2.shp`
    * `/Sewer_shp/SewerGravityMa_ExportFeature1.shp`
    * `/Sewer_shp/SewersqlSewerP_ExportFeature.shp`
    * `/Roads_shp/Roads_ExportFeatures.shp`
    * `MH_Road.pkl`
* **Preprocessing Script:** `process.py` (Contains functional but ugly code for data preprocessing)

### Contact

For any questions regarding implementation details, feel free to contact Yuxi Zhang at zhan2889@purdue.edu.

### Reference Paper

The research is based on the following paper:

* **Title:** Underground Utility Network Completion based on Spatial Contextual Information of Ground Facilities and Utility Anchor Points using Graph Neural Networks
* **Authors:** Yuxi Zhang and Hubo Cai
* **Link: https://www.iaarc.org/publications/fulltext/120_ISARC_2024_Paper_249.pdf**

### Citation

```markdown
@inproceedings{10.22260/ISARC2024/0121,
  doi = {10.22260/ISARC2024/0121},
  year = {2024},
  month = {June},
  author = {Zhang, Yuxi and Cai, Hubo},
  title = {Underground Utility Network Completion based on Spatial Contextual Information of Ground Facilities and Utility Anchor Points using Graph Neural Networks},
  booktitle = {Proceedings of the 41st International Symposium on Automation and Robotics in Construction},
  isbn = {978-0-6458322-1-1},
  issn = {2413-5844},
  publisher = {International Association for Automation and Robotics in Construction (IAARC)},
  editor = {Gonzalez-Moret, Vicente and Zhang, Jiansong and García de Soto, Borja and Brilakis, Ioannis},
  pages = {936-943},
  address = {Lille, France}
}

