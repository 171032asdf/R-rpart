# R-rpart
R rpart包 决策树 二分类
#读入数据
heart = as.data.frame(read.table(file.choose()))
#数据概况
install.packages('skimr')
library('skimr')
skim(heart)
#数据缺失情况
install.packages('DataExplorer')
library('DataExplorer')
plot_missing(heart)
colnames(heart) = c("age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","trget")
#剔除数据中含缺失值的样本
heart = na.omit(heart)
#变量类型纠正
for(i in c(2,3,6,7,9,11,13,14))
{
  heart[,i] = factor(heart[,i])
}
#数据处理后概况展示
skim(heart)
#因变量分布情况
table(heart$exang)


###########
#拆分数据
set.seed(42)
install.packages('caret')
library('caret')
trains = createDataPartition(
  y = heart$exang,#对于分类型因变量
  p = 0.75,
  list = F)
traindata = heart[trains,]
testdata = heart[-trains,]
#拆分后因变量分布
table(traindata$exang)
table(testdata$exang) 
 
############

#构建因变量自变量公式
colnames(heart)
form_cls = as.formula(
  paste0(
    "exang ~ ",
    paste(colnames(traindata)[c(1:8,11:14)],collapse = "+")
  )
)
form_cls

#构建模型
set.seed(42)  #固定交叉验证结果
#rpart包
install.packages('rpart')
library('rpart')
fit_dt_cls = rpart(
  form_cls,
  data = traindata,
  method = "class" , #分类模型
  parms = list(split = "gini"),
  control = rpart.control(cp = 0.001)
)
#原始分类树
fit_dt_cls
#复杂度相关数据
printcp(fit_dt_cls)
poltcp(fit_dt_cls,upper = "splits")
#后剪枝
fit_dt_cls_pruned = prune(fit_dt_cls,cp = 0.001)
print(fit_dt_cls_pruned)
#变量重要性
fit_dt_cls_pruned$variable.importance
#变量重要性图示
library(ggplot2)
varimpdata = data.frame(importance = fit_dt_cls_pruned$variable.importance)
ggplot(varimpdata,
       aes(x = as.factor(rownames(varimpdata)),y = importance)) +
  geom_col(width = 0.4, show.legend = FALSE)  +
  labs(x = "variables") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 15,hjust = 1))
# 树形图
#rpart.plot参考文档
install.packages('rpart.plot')
library('rpart.plot')
prp(fit_dt_cls_pruned,
    type = 2,
    extra = 104,
    tweak = 1,
    fallen.leaves = T,
    main = "Decision Tree")
########
#预测
#训练集预测概率
trainpredprob = predict(fit_dt_cls_pruned,newdata = traindata,type = "prob")
trainpredprob = ordered(factor(trainpredprob,levels = c("0","1")))
#训练集ROC
install.packages('pRoc')
library(pROC)
trainroc = roc(response = traindata$exang,predictor = trainpredprob[,2])
#训练集ROC曲线
plot(
  trainroc,
  print.auc = T,
  auc.polygon = T,
  grid = T,
  max.auc.polygon = T,
  auc.polygon.col = "skyblue",
  print.thres = T,
  legacy.axes = T,
  bty = "l")
#约登法则
bestp = trainroc$thresholds[which.max(trainroc$sensitivities + trainroc$specificities - 1)]
bestp
#训练集预测分类
trainpredlab = as.factor(ifelse(trainpredprob[,2] > bestp,"1","0"))
#训练集混淆矩阵
confusionMatrix(data = trainpredlab,  #预测类别
                reference = traindata$exang,   #实际类别
                positive = "1",
                mode = "everything")
#测试集预测概率
testpredprob = predict(fit_dt_cls_pruned,newdata = testdata,type = "prob")
#测试集预测分类
testpredlab = as.factor(ifelse(testpredprob[,2] > bestp,"1","0"))
#测试集混淆矩阵
confusionMatrix(data = testpredlab,  #预测类别
                reference = testdata$exang ,  #实际类别
                positive = "1",
                mode = "everything")
#测试集ROC
testroc = roc(response = testdata$exang,  #实际类别
              predictor = testpredprob[,2])  #预测概率
#训练集，测试集ROC曲线叠加
plot(trainroc,
     print.auc  = T,
     grid = c(0.1,0.2),
     auc.polygon = F,
     max.auc.polygon = T,
     main = "ROC",
     grid.col = c("green","red"))
plot(testroc,
     print.auc = T,
     print.auc.y = 0.4,
     add = T,
     col = "red")
legend("bottomright",
       legend = c("traindata","testdata"),
       col = c(par("fg"),"red"),
       lwd = 2,
       cex = 0.9)
