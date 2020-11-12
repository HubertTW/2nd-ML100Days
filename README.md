# 2nd-ML100Days
機器學習馬拉松 


## Day 18 FeatureType
數值型
類別型
時間型
## Day 19 MissingValue&Standardlization
對domain knowledge有一定瞭解為佳

填補統計值:
* mean:偏態不明顯時
* median:偏態明顯時
* mode:類別型

填補指定值:
* fill 0:猜測空缺值意同0
* 不可能出現的值

填補預測值:
* 注意overfitting

Standardlization:
* MinMaxScaler:適用於均於分布/不適用有極端值
* StandardScaler:適用於常態分布

樹狀模型:標準化後對預測值無影響
非樹狀模型:標準化後對預測值有影響

## Day 20 RemoveOutlier
須注意可能會刪除重要資訊

## Day 21 ReduceSkewness
老師想要調分數,希望讓全班成績愈接近**常態分佈**
### method:
* log1p:log(x+1)
* sqrt:sqrt(x-min)
* boxcox
```
from scipy import stats
y=stats.boxcox(x,0.5)
#lambda=0.5 意為開根號
```

## Day 22 LabelEncoding&OneHotEncoding
LabelEncoding:
計算空間小
不適用deep learning
OneHotEncoding:
計算空間大
適用deep learning
## Day 23 MeanEncoding&Smoothing
當發現類別型特徵與目標值有關係時(如地段與房價)
就使用目標值的平均當作原類別型特徵新編碼
(如所有大安區的房價取平均當作'大安區'的新特徵)
problem:易overfitting
有些類別筆數過少(如北投區只有五筆)容易產生誤差
因此必須smoothing
https://zhuanlan.zhihu.com/p/26308272

## Day 24 CountingEncoding&Hash

conunting:
使用該類別出現的次數當作新編碼
使用時機為目標平均值與類別筆數成正/負相關
hash:
使用時機為同一類別裡相異值過大時,如姓名
(embedding更佳)

## Day 25 TimeFeature
年週期:cos((月/6+日/180)pi)
週週期:sin((星期幾/3.5+小時/84)pi)
日週期:sin((小時/12+分/720+秒/43200)pi)

## Day 26 FeatureCombination

利用現有特徵組合成有意義的新特徵
需對領域知識有了解

## Day 27 GroupEncoding
groupencoding:數值特徵+類別特徵 組合成新特徵
(mean/max/min/median/mode/count...)
每個都可以試試看
不容易overfitting
## Day 28 特徵過濾
刪除一些比較不重要的特徵
* 相關係數過濾
* Lasso Regression 
* GDBT(主流) 可解決共線性問題且穩定

(多重共線性是指多變量線性回歸中，變量之間由於存在高度相關關係而使回歸估計不準確)

## Day 29 feature inportance
樹狀狀模型的特徵重要性，可以分為
分支次數、特徵覆蓋度、損失函數降低量
permutation importance
[what's Series](https://ithelp.ithome.com.tw/articles/10193394)
Regressor 用於目標欄位是連續型的資料，Classifier 用於目標欄位是離散型的資料。當離散型的資料只有兩種值，稱為 Binary Classifier；有超過兩個以上的可能稱為 Multi-Class Classifier。
## Day 30 leaf encoding

以decision tree nodes 當作leaf

```
# 梯度提升樹調整參數並擬合後, 再將葉編碼 (*.apply) 結果做獨熱 / 邏輯斯迴歸
# 調整參數的方式採用 RandomSearchCV 或 GridSearchCV, 以後的進度會再教給大家, 本次先直接使用調參結果
gdbt = GradientBoostingClassifier(subsample=0.93, n_estimators=320, min_samples_split=0.1, min_samples_leaf=0.3, 
                                  max_features=4, max_depth=4, learning_rate=0.16)
onehot = OneHotEncoder()
lr = LogisticRegression(solver='lbfgs', max_iter=1000)

gdbt.fit(train_X, train_Y)
onehot.fit(gdbt.apply(train_X)[:, :, 0])
lr.fit(onehot.transform(gdbt.apply(val_X)[:, :, 0]), val_Y)
```
```
# 將梯度提升樹+葉編碼+邏輯斯迴歸結果輸出
pred_gdbt_lr = lr.predict_proba(onehot.transform(gdbt.apply(test_X)[:, :, 0]))[:, 1]
fpr_gdbt_lr, tpr_gdbt_lr, _ = roc_curve(test_Y, pred_gdbt_lr)
# 將梯度提升樹結果輸出
pred_gdbt = gdbt.predict_proba(test_X)[:, 1]
fpr_gdbt, tpr_gdbt, _ = roc_curve(test_Y, pred_gdbt)
```
## Day 34 K-fold
隨機把資料平均分成k組 取一組作testing data 
剩下k-1組作trainging data
重複餵進model作訓練直到每一組都當過testing data


## Day 36 Evaluation Metircs

Regression:
1. MAE
2. RSE
3. R^2

Classification:
1. AUC/ROC [0,1]
p>0.5=1   p<0.5=0
https://www.jianshu.com/p/c61ae11cc5f6

2. F-1 Score [0,1]
透過beta調整recall&precision的權重
https://blog.csdn.net/matrix_space/article/details/50384518


4. Confusion Matrix
T/F:預測是否正確
P/N:預測方向
Accuracy:True Positive+True Negative
precision:模型判定瑕疵，樣本確實為瑕疵的比例
TP/TP+FP 愈高預測愈精準
recall:模型判定的瑕疵，佔樣本所有瑕疵的比例
TP/TP+FN 
在實際情形為正向的狀況下，預測「能召回多少」正向的答案


https://www.ycc.idv.tw/confusion-matrix.html
```
auc=metrics.roc_auc_score(y_test,y_pred)

```


## Day 37 Logistic Regression
Classification:
1. logistic regression(discriminative model)
 
3. generative model(based on probability distribution&assumption)
找出最有可能產生出樣本屬於c1的Gaussian Fuction(mean/cov. matrix)



benefit of  generative model compare to dis. model :
1. less training data is need
2. more robust to noise

優缺點:https://blog.csdn.net/qq_23269761/article/details/81778585

linear vs non-linear
1. 非線性是指兩個變數之間的關係，是不成簡單比例（即線性）的關係。
2. 所謂線性，從數學上來講，是指方程的解滿足線性疊加原理，即方程任意兩個解的線性疊加仍然是方程的一個解。
3. 自變量與變量之間不成線性關係，成曲線或拋物線關係或不能定量，這種關係叫非線性關係。


## Day 39 LASSO/Ridge Regression

Regularization:

larger w:w愈不平滑,larger L,model more complex,overfitting
larger lambda:new model with smaller w but too large would cause underfitting


* LASSO:L=F(x)+lambda*abs(w)

* RidgeRegression:lambda*pow(w,2)

"当使用最小二乘法计算线性回归模型参数的时候，如果数据集合矩阵（也叫做设计矩阵(design matrix)）X，存在多重共线性，那么最小二乘法对输入变量中的噪声非常的敏感，其解会极为不稳定"

## Day 41 Decision Tree
依據feature來切分資料,希望切分後資料相似程度高
相似程度:
* GINI:愈大愈適合當作節點(feature)
* Entropy:愈小愈適合當作節點(feature)

information gain

## Day 45 Gradient Boosting Machine
* bagging:RF用抽樣的資料與特徵生成每一棵樹 再取平均(**RF**)
* boosting:藉後面生成的樹來修正前面的樹 透過gradient修正(**GradientBoosting**) 

https://ifun01.com/84A3FW7.html

## Day 70 Multilayer-Perception
def:擁有多層layer多個perception的神經網路
再更多層就是DNN
regression and liner discriminant classifier只是

ref:
1. [MLP黃志勝](https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-%E5%A4%9A%E5%B1%A4%E6%84%9F%E7%9F%A5%E6%A9%9F-multilayer-perceptron-mlp-%E9%81%8B%E4%BD%9C%E6%96%B9%E5%BC%8F-f0e108e8b9af)
