# Change Point Detection in Time Series via Discrete Cosine Similarity with Confidence Level Estimation

## Introduction

This paper introduces DCS-CPD, a novel method for change point detection in time series using cosine similarity. The method constructs a cosine similarity series that represents the statistical properties of the original series, allowing for the detection of change points. When no change points are present, the series follows a specific distribution, while deviations indicate potential change points. 
DCS-CPD derives the relationship between the change point probability and cosine similarity, enabling precise localization and probability estimation. Compared to traditional methods, DCS-CPD offers superior computational efficiency and broader applicability. Experiments on synthetic and real-world datasets validate its accuracy, robustness, and ability to provide reliable probability estimates for detected change points. 

## Synthetic Datasets

The synthetic datasets are crafted by **CraftedData.py**. 

Run 

```cmd
python CraftedData.py --length 1000 --peroid 16
```

to creat the synthetic datasets with the length of 1000 and the peroid of 16. 

## Discrete cosine similarity (DCS)

![](./pic/DCS.png)

## DCS-CPD

The function to detect the change points is defined in **DCS_CPD.py**. 

## Illustration of DCS-CPD for synthetic series.

![](./pic/exp_plotmvp.png)

As shown in (a), changes in mean, variance, and period were introduced at position $n=100$, respectively. With the threshold $p_t = 0.1$, (b) displays the corresponding DCS series along with the DCS threshold $y_t$. (c) presents the confidence curve derived from the $y_t$. 

## Experiment results

![](./pic/result_detection.png)

![](./pic/result_slimness.png)

