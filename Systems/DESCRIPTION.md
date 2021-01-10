
The submission file IIITT.zip has the systems as follows:

- run 1 : Pre-trained Transformer Model (allenai/scibert_scivocab_uncased)
- run 2 : Average of probabities of predictions of ( BERT_base_uncased + RoBERTa_base + SciBERT)
- run 3 : Ensemble of probabilities of predictions by ranking the percentile of the result stored as a pandas DataFrame 
