# XReg-Test
This repository details the steps needed to create a product-to-product recommendation dataset as per `./problem_statement.txt` and then evaluate an implementation of Parabel (XReg) on it.

_Note: `related.txt` and `sample_descriptions.txt` (as mentioned in `./problem_statement.txt`), being large files, are not uploaded to this repository._

XReg and Parabel are eXtreme Classification (XC) algorithms introduced in the papers below:

- [Extreme Regression for Dynamic Search Advertising](http://manikvarma.org/pubs/prabhu20.pdf), WSDM 2020
- [Parabel: Partitioned Label Trees for Extreme Classification with
Application to Dynamic Search Advertising](http://manikvarma.org/pubs/prabhu18b.pdf), WWW 2018

For an exhaustive list of XC methods and much more, please refer to [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html).

The next sections describe the process adopted for dataset creation, debugging and improving the precision@1 metric.

## Dataset Creation
[The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html) mentions the link to [PyXCLib](https://github.com/kunaldahiya/pyxclib). We note that PyXCLib contains the file [sparse_bow_features_from_raw_data.py](https://github.com/kunaldahiya/pyxclib/blob/master/xclib/examples/sparse_bow_features_from_raw_data.py) which can be used for preprocessing and creating the dataset.
It uses TfidfVectorizer (from scikit-learn) and additionally removes english stop words and punctuation.
However, it accepts the data in gzipped JSON format. So, we take the following steps:
1. Use `./filter_format.py` to filter unnecessary products from the dataset and create train-test (80/20) split into JSON format files. The train-test split files are then gzipped.
2. Using the gzipped files created in previous step in the `./sparse_bow_features_from_raw_data_new.py` (a modified version of the [file](https://github.com/kunaldahiya/pyxclib/blob/master/xclib/examples/sparse_bow_features_from_raw_data.py) from PyXCLib), we create `./features_labels/trn_processed` and `./features_labels/tst_processed`.
3. The `trn_processed` and `tst_processed` produced in previous step contain the multi-labels as well as data point's features in same file. XReg, however, uses a different file format that splits this information across two files. 
   We use `Tree_Extreme_Classifiers/Tools/convert_format.pl` from [Parabel's code](http://manikvarma.org/code/Parabel/download.html) for this conversion. This gives us `trn_X_Xf.txt`, `trn_X_Y.txt`, `tst_X_Xf.txt`, `tst_X_Y.txt` files as desired.
4. Finally, we train the XReg model on the train data, make predictions on the test data and evaluate them.

Steps 1 and 2 are further explained below.

### Filtering the dataset
We remove those products from the dataset which do not aid the learning process. These are of three types:
1. Products (in `related.txt`) that are not present in `sample_descriptions.txt`.
   - After such products are removed, some related products' lists in `related.txt` may become empty and the corresponding products must also be removed from `related.txt`.
   The cycle repeats and we keep removing products until convergence as done in `removeEmpty(..)` function in `./filter_format.py`.
2. Products (in `related.txt` and `sample_descriptions.txt`) that are not labels of any product.
3. Products (in `related.txt` and `sample_descriptions.txt`) that do not have any labels.
    - We first remove the required products from `sample_descriptions.txt` and follow a process similar to removal of type 1 products above to remove the products from `related.txt`. 
  After one round of such removal, `sample_descriptions.txt` may contain more products that need to be removed as they do not have any labels now.
The cycle repeats and we keep removing products until convergence as done in `type3Removal(..)` function in `./filter_format.py`.

We then split this dataset into train and test set files containing rows like the following JSON object:
```json
{"id": productId, "description": product description, "related_products" : [indices of related products]}
```

### Creating `sparse_bow_features_from_raw_data_new.py`
Two small changes are made to existing [`sparse_bow_features_from_raw_data.py`](https://github.com/kunaldahiya/pyxclib/blob/master/xclib/examples/sparse_bow_features_from_raw_data.py):
1. Using correct keys (viz. `description` and `related_products`) in `read(..)`method.
2. Using the `Statistics` class from [here](https://github.com/kunaldahiya/pyxclib/blob/master/xclib/data/data_statistics.py) to compute dataset statistics.

The created dataset had the following statistics:
```
avg_doc_length = 71.19258089976321
n_avg_labels_per_sample = 1.3290112018585205
n_avg_samples_per_label = 1.3290482759475708
n_features = 30086
n_labels = 8868
n_test_samples = 1774
n_train_samples = 7095
```

## Debugging
When the created dataset was used to train and make predictions using XReg, we got precision@1 as 0. 
It was because the scores from all the trees were not being collated and an empty output score file was generated. The reason was the line 113 in `implementation_question/XReg-master/Source/xreg_predict.cpp`:
```
string temp_score_file_name = string( argv[3] ) + to_string( param.start_tree + i );
```                                            
The line is creating a `temp_score_file_name` string but it is using `argv[3]` which is the input feature file name for test set. We need to use `argv[2]` here.
On making this change, we get the following results (P@1 = 7.72265):
```
pointwise metrics
prec
        1:      7.72265
psp
        1:      7.72272
ndcg
        1:      7.72272
labelwise metrics
prec
        1:      3.89038
psp
        1:      15.8908
ndcg
        1:      15.8908
```

## Training and testing details
Using the provided `Makefile` we compile the XReg sources in `implementation_question/XReg-master/Source` into binaries.

We use the following commands (default set of hyperparameters):
```
./xreg_train ~/implementation_question/xreg_model ~/implementation_question/trn_X_Xf.txt ~/implementation_question/trn_X_Y.txt -s 0 -T 1 -t 3 -w 0 -k 0 -kleaf 0 -c 1.0 -m 100 -tcl 0.05 -ecl 0.1 -n 20 -r 0
./xreg_predict ~/implementation_question/xreg_model ~/implementation_question/xreg_model/out_score ~/implementation_question/tst_X_Xf.txt -T 1 -s 0 -B 10 -p 0 -pf 10.0 -ps -0.05 -r 0 -a 1.0
./xreg_metric ~/implementation_question/xreg_model/out_score ~/implementation_question/tst_X_Y.txt 1 0
```
For the run mentioned under `Debugging` section, the model size, training time and prediction time are mentioned below:
```
model size : 0.0219133 GB
training time : 0.00141966 hrs

prediction time before taking ensemble : 0.408824 ms/point
...
prediction time : 0.683912 ms/point
```

## Improving the precision@1 metric
### v2
To check the role of random split, we create another train-test split. We also added two new preprocessing steps (limit on max document frequency and stripping of accents) by instantiating `BoWFeatures` in `./sparse_bow_features_from_raw_data_new.py` as:
```
BoWFeatures(encoding=encoding, max_df=0.7, min_df=2, strip_accents='unicode', dtype=dtype)
```
This increased the P@1 slightly (from 7.72 to 7.67):
```
pointwise metrics
prec
        1:      7.66628
psp
        1:      7.66635
ndcg
        1:      7.66635
labelwise metrics
prec
        1:      3.76634
psp
        1:      15.2786
ndcg
        1:      15.2786
```

### v3
The maximum improvement came with the following idea of 'intelligent' splits (mentioned under `Split Creation` [here](http://manikvarma.org/downloads/XC/XMLRepository.html)):

_Splits were not created randomly but instead created in a way that ensured that every label has at least one training point. This yielded more realistic train/test splits as compared to uniform sampling which could drop many of the infrequently occurring, and hard to classify, labels from the test set._

While we cannot ensure that each label gets covered, we can maximize the label coverage within our training set.
We do so by using the `getMaxLabelCoverageSplit(..)` method in `./filter_format.py` which maximizes the label coverage by preferentially picking those data points in train set that have more labels (i.e. more related products).

This change increases label coverage from 82.06% (random split) to 91.83% and results in a P@1 of 20.9695.
```
pointwise metrics
prec
        1:      20.9695
psp
        1:      20.9697
ndcg
        1:      20.9697
labelwise metrics
prec
        1:      5.32249
psp
        1:      21.8613
ndcg
        1:      21.8613
```

### v4
We also start performing stemming along with tokenization in `./sparse_bow_features_from_raw_data_new.py` which improves the P@1 to 22.5479

```
pointwise metrics
prec
        1:      22.5479
psp
        1:      22.5481
ndcg
        1:      22.5481
labelwise metrics
prec
        1:      5.44653
psp
        1:      22.3708
ndcg
        1:      22.3708
```
Training and prediction time for this run are:
```
model size : 0.0183866 GB
training time : 0.000827308 hrs

prediction time before taking ensemble : 0.442911 ms/point
...
prediction time : 0.717958 ms/point

```

## Hyper-parameter tuning
We also evaluate the effect of tuning the training hyper-parameters on P@1 value.

### Effect of number of trees (t)
```
t = 2 -->  22.0405
t = 4 -->  21.9842
t = 5 -->  22.4915
t = 6 -->  22.3224
t = 7 -->  21.9842
```

We find that increasing or decreasing the number of trees does not benefit precision.

### Effect of changing linear classifier type for internal nodes (k)
```
k = 1 --> 22.0969
k = 3 --> 17.8128
k = 4 --> segmentation fault
k = 5 --> segmentation fault
```

### Effect of changing linear classifier type for leaf nodes (kleaf)
```
kleaf = 1 --> 23.1679
kleaf = 3 --> 1.01466
kleaf = 4 --> 22.4915
kleaf = 5 --> 21.7023
```

Note that this improves the best observed P@1 to 23.1679

### Effect of changing maximum iterations (n)
```
n = 30 --> 22.5479
n = 40 --> 22.5479
n = 50 --> 22.5479
n = 60 --> 22.5479
n = 70 --> 22.5479
n = 80 --> 22.5479
n = 90 --> 22.5479
n = 100 --> 22.5479
n = 1000 --> 22.5479
```

### Effect of changing maximum no. of labels in a leaf node (m)
```
m = 10 --> 0
m = 50 --> 21.4768
m = 80 --> 22.5479
m = 150 --> 20.8568
m = 120 --> 22.5479
m = 200 --> 20.8568
```

## Best Model
We observe the best P@1 value to be 23.1679 which occurs when we change the value of `kleaf` hyper-parameter from 0 (default) to 1.

All files in this repository (`./filter_format.py`, `./sparse_bow_features_from_raw_data_new.py`, `./features_labels/*` and `./xreg_model/*`) correspond to this version.


Detailed Metrics:
```
pointwise metrics
prec
        1:      23.1679
psp
        1:      23.1681
ndcg
        1:      23.1681
labelwise metrics
prec
        1:      6.57418
psp
        1:      27.0024
ndcg
        1:      27.0024

```

Training and prediction time for this run are:
```
model size : 0.0190113 GB
training time : 0.00126041 hrs

prediction time before taking ensemble : 0.454016 ms/point
...
prediction time : 0.770244 ms/point
```