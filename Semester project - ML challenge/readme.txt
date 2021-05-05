Train data (train.csv):
1) The data is in a tabular format corresponding to 400 rows and 33 attributes (the first 33 columns) as well as a label (the last, 34th, column).
2) Missing values are marked by NA.
3) The data are anomynimzed:
- There are no attribute names, you can use X1, X2, X33 and Y, referring to the first 33 columns and the last column (the label).
- Nominal values are marked by the values V1, V2, V3, etc. which are unique only within a column. It means that, for example, V1 in one column is not equal to V1 in another column (it just simply refers to some value, V1, of a given attribute which, is equal to V1 in another row in case of the same attribute). In other words, V1 and V2 might refer to the values "yellow" and "red", respectively, in case of the attribute X1 while in case of X2 these might refer to the values "Budapest" and "Prague", repsectively.
- Numeric attributes are normalized.

Test data (test.csv):
- There are only 33 columns, i.e. no label, since this has to be predicted by you. The same applies to missing values and anonymization as in the case of train data (see above).

Submissions:
- Each of you has to send (by e-mail, to me at tomas.horvath@inf.elte.hu) one text file containing a single column with the predictions where each row contains one number, the label, related to the corresponding row (instance) in the set (test.csv file). It means, the submission file should have 100 rows and one column.
- The deadline for submissions is 30th of May 2021.
- Only the first submission will be considered, thus, please, be sure that you have sent your final predictions.

Evaluation:
- The Accuracy score will be computed as the number of correct predictions divided by 100 (the number of instances/rows in the test set).
- In case of same accuracy for more students, the order in which the submissions were received will be considered (the first submission gets more points as the later submissions with the same accuracy).

Summarized:
- You have to train your model on the train data and submit the predictions for the instances in the test data by 30th of May 2021 which will be evaluated according to the Accuracy score.
 
Submit 3 files:	
2 txt files:
- predictions (1 column, 100 rows; worth 30 points)
- used code (for fraud control)
1 pdf: 
- Report (6-8 pages; worth 10 points)