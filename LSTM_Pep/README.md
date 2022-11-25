Peptide finetune training and generating example

## Requirements
This model is built using Python 3.7, and utilizes the following packages;

* numpy 1.18.2
* tensorflow 2.1.0
* tqdm 4.43.0
* Bunch 1.0.1
* matplotlib 3.1.2
* RDKit 2019.09.3
* scikit-learn 0.22.2.post1


The input fasta for training the initial model was deposite in https://figshare.com/s/60ac3aaa96eb0942f8a2 .
The input fasta files for funetuning were putted in generated_data folder.
Finetune training the model and generating de novo peptide by following command:
python retrain.py  
