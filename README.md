# MTT-Sentiment-Classification
Mutlimodal Transformer Translation for Sentiment Classification. Using Text and Audio. 

Checkout and run pip install requirements. Can run and train either in CPU or GPU.
To run,  the datasett CMU-MOSEI data files in folder data. You can find the files for the standart processing for training and benchmarking here http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/ In more detail, the mosei_senti_data.pkl is the aligned data file  for the MOSEI and mosi_data.pkl for MOSI(both 50 length).
Run with parameters --model  mctn, mtt_cyclic(default), mtt_fuse. For  parameter --dataset either mosei(default) or mosi.
