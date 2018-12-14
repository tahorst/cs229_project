# Identifying Transcription Unit Structure from Rend Sequencing Data
Final project for CS229 (Machine Learning).  Data comes from GEO accession number GSE95211 associated with Lalanne JB, et al.
(2018). Evolutionary Convergence of Pathway-Specific Enzyme Expression Stoichiometry. Cell 173(3) 749-761.

The goal of this project is to use sequencing data to identify transcription unit initiation and termination sites within a
genome to determine which genes are expressed together. Although partially known, identifying all transcription units in an
organism can help create more accurate models of biologic behavior by better capturing interactions between coexpressed genes
leading to increased predictive capacity.  Unsupervised and supervised methods were used to identify structure from transcript
sequencing data. Supervised learning methods performed better at identifying transcription start and stop sites and avoiding
false predictions; however, they did not generalize well to the test set.

## Running code
### Models
Python 3.6 was used with packages in requirements.txt.  Running the models included in the final report and poster can be done
with the following lines (changing to `test = False` in each file will do a search among parameters of interest, otherwise models
are run with selected parameters):
```
python src/dbscan.py
python src/hmm.py
python src/logreg.py
python src/nn.py
```

### Generating plots
Some analysis plots used to explore the data are generated with the following script:
```
python src/analysis.py
```

### Other files
Start and end positions for regions of genes used in the unsupervised methods were generated with the following:
```
python src/identify_regions.py
```

Many shared functions are defined in `src/utils.py`.  Other models that were briefly explored are also include in `src/`.
