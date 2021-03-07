# debias_reappearance

1. 2016 - Recommendations as Treatments: Debiasing Learning and Evaluation

   Reappearance experiments with model "MF_IPS" and "MF_Naive" in pytorch.

   Dataset used is "Yahoo!R3".

   data file:

   * train.txt: origin train data in Yahoo!R3
   * test.txt: origin test data in Yahoo!R3
   * test1.txt: sample from test data of Yahoo!R3 and it contains 2700 interactions ( 5% of total interactions, and it is used for calculating propensity score ). ( This data is also used as S_t in CausEProd model )
   * test2.txt: remain 95% data of the test data of Yahoo!R3. ( This data is for test in experiments )

   Results is as follow:

   | Yahoo!R3             | MAE   | MSE   |
   | -------------------- | ----- | ----- |
   | MF_IPS( in paper )   | 0.810 | 0.989 |
   | MF_IPS               |       |       |
   | MF_Naive( in paper ) | 1.154 | 1.891 |
   | MF_Naive             |       |       |



2. 2020 - Improving Ad Click Prediction by Considering Non-displayed Events

   Reappearance experiments with model "CausEProd" in pytorch.

   The evaluation part of this reappearance is the same as below.
   
   | Yahoo!R3  | MAE  | MSE  |
   | --------- | ---- | ---- |
   | CausEProb |      |      |
   
   
