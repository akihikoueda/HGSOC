# Histopathological subtyping of high-grade serous ovarian cancer

This repository contains codes and prediction results generated for analysis in "Artificial intelligence-based histopathological subtyping of high-grade serous ovarian cancer". 

codes details
- model.py : the codes to create the deep learning model based on the NASNet-A-Large model pre-trained by ImageNet.
- processing.py: the codes for analysis of the whole slide tissue to predict the spatial distribution of tPattern, tTIL, and the percentage of occupancy within tumours. The codes require kindai.py for the prediction of tile annotation.
- visualization.ipynb: the codes to construct AI-based histopathology subtype classification algorithm based on predicted tile labels within WSI.
