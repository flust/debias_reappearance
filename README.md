# debias_reappearance

1. 2016 - Recommendations as Treatments: Debiasing Learning and Evaluation

   Reappearance experiments with model "MF_IPS" and "MF_Naive" in pytorch.

   Dataset used is "Yahoo!R3".

   data file:

   * train.txt: origin train data in Yahoo!R3
   * test.txt: origin test data in Yahoo!R3
   * test1.txt: sample from test data of Yahoo!R3 and it contains 2700 interactions ( 5% of total interactions, and it is used for calculating propensity score ). ( This data is also used as S_t in CausEProd model )
   * test2.txt: remain 95% data of the test data of Yahoo!R3. ( This data is for test in experiments )

   

   Run this experiment:
   
   ```
   python main.py --model=MF_Naive
   python main.py --model=MF_IPS
   ```
   
   The parameters can be changed in config.py `DefaultConfig`
   
   
   
   Results is as follow:
   
   | Yahoo!R3             | MAE    | MSE    |
   | -------------------- | ------ | ------ |
   | MF_IPS( in paper )   | 0.810  | 0.989  |
   | MF_IPS               | 0.8787 | 1.3653 |
   | MF_Naive( in paper ) | 1.154  | 1.891  |
   | MF_Naive             | 1.0136 | 1.6804 |



2. Causal Embeddings for Recommendation

   Reappearance experiments with model "CausEProd" in pytorch.

   The evaluation part of this expriment is the same as before.
   
   
   
   Run this experiment:
   
   ```
   python main.py --model=CausEProd
   ```
   
   The parameters can be changed in config.py `DefaultConfig`
   
   
   
   | Yahoo!R3  | MAE    | MSE    |
   | --------- | ------ | ------ |
   | CausEProb | 0.9138 | 1.2734 |
   
   
   
3. Improving Ad Click Prediction by Considering Non-displayed Events

   | Yahoo!R3(10 epoch BCELoss) | NLL(in paper) | NLL     | AUC(in paper) | AUC     |
   | -------------------------- | ------------- | ------- | ------------- | ------- |
   | average($S_c$)             | +0.0%         | +0.0%   | +0.0%         | +0.0%   |
   | average($S_t$)             | +79.1%        | +22.69% | +0.0%         | +0.0%   |
   | FM($S_c$)                  | -7.7%         | +23.01% | +36.4%        | +34.76% |
   | FM($S_t$)                  | -20.7%        | -0.03%  | +4.6%         | +3.06%  |
   | FM($S_c \cup S_t$)         | +0.2%         | +23.15% | +36.4%        | +34.96% |
   | IPS                        | +62.2%        | +18.50% | +23.2%        | +33.57% |
   | CausE                      | +6.8%         | /       | +37.8%        | /       |
   | New(avg)                   | +79.1%        | +17.64% | +51.8%        | +33.06% |
   | New(item-avg)              | +76.8%        |         | +54.2%        |         |
   | New(complex)               | -0.2%         | /       | +37.4%        | /       |

   