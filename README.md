# Double/Debiased Machine Learning Difference-in-Differences estimator

This repository includes Python code for estimating the Double/Debiased Machine Learning Difference-in-Differences estimator coded by Liudmila Gorkun-Voevoda for her Master thesis "Effect of Environmental Policy Changes on Consumer Waste Behaviour".

### Theoretical Background

The Double/Debiased Machine Learning Difference-in-Differences estimator (DMLDiD) was proposed by Chang (2020) and is an orthogonal extension of the Abadie's semiparametric Difference-in-Differences estimator of ATET. Particularly, in the case of repeated cross-sections, under the according assumptions, the ATET estimator can be calculated as $$\hat{ATET} = \dfrac{1}{N}\sum_{i=1}^{N}\dfrac{T_i-\hat{\lambda}_i}{\hat{\lambda}_i (1-\hat{\lambda}_i)} \dfrac{Y_i}{\hat{\pi}} \dfrac{D_i-\hat{p}(X_i)}{1-\hat{p}(X_i)}$$
where $\hat{\lambda}_i$ is the estimator of $\mathbb{P}(T_i=1)$, $\hat{\pi}$ is the estimator of $\mathbb{P}(D=1)$ and $\hat{p}(X_i)$ is the estimator of propensity score $\mathbb{P}(D=1|X=x)$. 
This estimator of ATET is constructed for the case when $\hat{p}(X_i)$ is estimated with classical non-parametric estimation techniques (such as kernel estimation). 
However, if machine learning methods are used for the first stage estimation, this estimator may loose its convenient properties of being $\sqrt{N}$-consistent and asymptotically normal, therefore producing a bias in the final estimation. There are two reasons for such behaviour. 
Firstly, a score function based on Abadie's estimator has a non-zero directional derivative with respect to the propensity score. Using estimators for which this derivative would be equal to zero (the Neyman orthogonality condition) is important to obtain valid estimators of ATET when machine learning methods are used for estimation of the nuisance parameters. 
Secondly, generally machine learning estimators in such a setting will not be $\sqrt{N}$-consistent since they often have a slower convergence rate due to regularization bias. Therefore, if machine learning estimates are directly used in the equation stated above, then the estimators will not be $\sqrt{N}$-consistent.
The DMLDiD estimator is able to correct bias obtained from machine learning-based first stage estimation by introducing zero-mean adjustment terms to the second stage estimator, which make the Neyman orthogonality condition hold. Author combines the new adjusted scores with the cross-fitting algorithm developed by Chernozhukov et al. (2018) to construct the DMLDiD estimator.

Therefore, the procedure is as follows: firstly, the whole sample is split into $K$ sub-samples of the equal size $n$. Here I am splitting the sample into two sub-samples following Chang (2020). The final ATET estimator is equal to the average of $K$ sub-sample ATET estimators, where observations from the initial sample are assigned randomly into each sub-sample $I_k$. Each of those sub-sample estimators is defined as 
$$ATET_k = \dfrac{1}{n}\sum_{i\in I_k} \dfrac{ D_i - \hat{p}_ {k}\left(X_i\right)} {\hat{\pi}_ {k}\hat{\lambda}_ {k} \left(1-\hat{\lambda}_ {k}\right) \left(1-\hat{p}_ {k} \left(X_i\right)\right)} \times \left(\left(T_i-\hat{\lambda}_ {k}\right) Y_i - \hat{\mathit{l}_ {2k}}(X_i)\right)$$

where:
* $\hat{p_k}(X_i)$ is a propensity score estimator which can be estimated using any machine learning method, for which the training set is the auxiliary sub-sample $I^c_k$ that includes all the other sub-samples of the initial sample apart from $k$;
* $\hat{\pi}_ {k} = \frac{1}{n} \sum_{i \in I^c_k} D_i$ is the estimator of the probability of treatment $\mathbb{P}(D=1)$;
* $\hat{\lambda}_ {k} = \frac{1}{n} \sum_{i \in I^c_k}T_i$ is the estimator of $\mathbb{P}(T = 1)$;
* $\hat{\mathit{l}}_ {2k}(X_i)$ is the estimator of the expected weighted outcomes $\mathit{l}_{20} = \mathbb{E}\left[\left( T -\lambda \right) Y | X, D = 0\right]$. Similarly to $\hat{p_k}(X_i)$, it can be estimated with any machine learning method, using $I^c_k$ for training.

For each sub-sample $I_k$, the auxiliary subsample $I^c_k$ is used for calculation of $\hat{\pi}_k$ and $\hat{\lambda}_k$.

I am using an Ensemble Learner for estimation of the propensity scores $\hat{p_k}(X_i)$ and the function $\hat{\mathit{l}}_{2k}(X_i)$. An Ensemble Learner is a combination of multiple different machine learning methods, results of which are weighted in a certain way to produce the final estimation. In my analysis, the Ensemble Learner is represented by a combination of Random Forest and Logistic LASSO models. Such a choice follows a paper of Zimmert (2020) who pointed out that ability of Random Forest to account for strong non-linearities together with smoothing properties of a LASSO can produce good estimates. 
