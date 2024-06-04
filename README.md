Instruction:
- Main Program:
Inductive.ipynb (ugly codes but work)
There are some issues you need to resolve on your own (I am not sure if I have time to clean this file):
1. The codes for different architecture variants are not saved, you need to manipulate the commented lines to reproduce the work
2. You need to manually change the file names and paths to save the models and results on your local computers.

- Results:
model_metricsxxx.xlsx

- Trained Models (Checkpoints):
/Models/...

- Dataset(cleaned):
/Data/...

- Dataset(preprocessed):
MH_proc.pkl
Road_proc.pkl
MH_R_RL_proc.pkl
Line_proc.pkl
R_R_proc.pkl
split_mask.pkl # mask for splitting train/val/test (optional, required for reproducing the results)

- Dataset(raw):
/Sewer_shp/SewerManholes_ExportFeatures.shp
/Sewer_shp/SewerGravityMa_ExportFeature2.shp
/Sewer_shp/SewerGravityMa_ExportFeature1.shp
/Sewer_shp/SewersqlSewerP_ExportFeature.shp
/Roads_shp/Roads_ExportFeatures.shp
MH_Road.pkl
Preprocessing file (ugly codes but work): process.py

If you have any questions on implementation, feel free to contact me (zhan2889@purdue.edu).

Paper:
https://www.iaarc.org/publications/fulltext/120_ISARC_2024_Paper_249.pdf

Citation:
@inproceedings{10.22260/ISARC2024/0121,
	doi = {10.22260/ISARC2024/0121},
	year = 2024,
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

