# Thesis_DMLDiD

This repository includes Python code for estimating the Double/Debiased Machine Learning Difference-in-Differences estimator coded by Liudmila Gorkun-Voevoda for her Master thesis "Effect of Environmental Policy Changes on Consumer Waste Behaviour".

The Double/Debiased Machine Learning Difference-in-Differences estimator (DMLDiD) was proposed by Chang (2020) and is an orthogonal extension of the Abadie's semiparametric Difference-in-Differences estimator of ATET. Particularly, in the case of repeated cross-sections, under the according assumptions, the ATET estimator can be calculated as $$\hat{ATET} = \dfrac{1}{N}\sum_{i=1}^{N}\dfrac{T_i-\hat{\lambda}_i}{\hat{\lambda}_i (1-\hat{\lambda}_i)} \dfrac{Y_i}{\hat{\pi}} \dfrac{D_i-\hat{p}(X_i)}{1-\hat{p}(X_i)}$$.
