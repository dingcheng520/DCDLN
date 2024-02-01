clc;       %清除命令窗口的内容，对工作环境中的全部变量无任何影响 
clear all; %清除工作空间的所有变量，函数，和MEX文件
close all; %关闭所有的Figure窗口
%% Load and process data, build neural network and label training set
% load('Parkinson_pool') 
load('data_densenet121.mat')
% load('data_skin_densenet121.mat')
%load('data_rubbish1_densenet121.mat')
train_data = train_data_densenet121(2:end, :);
val_data = val_data(2:end, :);
X1 = [train_data;val_data];
v = val_data(2,:);
v_data = v(1:1024);
test_feature = val_data(:,1:1024);
test_labels = val_data(:,1025);

classificationLearner
% save('malaria_Bayesian.mat', 'Bayesian');
% save('malaria_LDA.mat', 'LDA');
% save('malaria_LinearSVM.mat', 'LinearSVM');
% save('malaria_GaussianSVM.mat', 'GaussianSVM');
% save('malaria_kNN.mat', 'kNN');
% save('malaria_DecisionTree.mat', 'DecisionTree');
% save('malaria_BaggingTree.mat', 'BaggingTree');
load('malaria_Bayesian.mat', 'Bayesian');
load('malaria_LDA.mat', 'LDA');
load('malaria_LinearSVM.mat', 'LinearSVM');
load('malaria_GaussianSVM.mat', 'GaussianSVM');
load('malaria_kNN.mat', 'kNN');
load('malaria_DecisionTree.mat', 'DecisionTree');
load('malaria_BaggingTree.mat', 'BaggingTree');
% load('rubbish_Bayesian.mat');
% load('rubbish_LDA.mat');                                                                       
% load('rubbish_LinearSVM.mat');
% load('rubbish_GaussianSVM.mat');
% load('rubbish_kNN.mat');
% load('rubbish_DecisionTree.mat');
% load('rubbish_BaggingTree.mat');
% 1. Naive Bayes
tic;
yfit_Bayesian=Bayesian.predictFcn(test_feature);
Bayesian_time = toc;
save('yfit_Bayesian.mat','yfit_Bayesian');
gaptest=yfit_Bayesian-test_labels;
Bayesian_acc=length(find(gaptest==0))./size(test_feature,1);
%Confusion matrix
cf_Bayesian=confusionmat(yfit_Bayesian,test_labels);
% Precision
TP=diag(cf_Bayesian);
FN=[];FP=[];
for i=1:size(cf_Bayesian,1)
    FN(i,1)=sum(cf_Bayesian(i,:))-cf_Bayesian(i,i);
    FP(i,1)=sum(cf_Bayesian(:,i))-cf_Bayesian(i,i);
end
mprecision=[];
for i=1:size(cf_Bayesian,1)
    mprecision(i)=TP(i)./(TP(i)+FP(i))./size(cf_Bayesian,1);
end
macro_precision_Bayesian=sum(mprecision);
wprecision=[];
for i=1:size(cf_Bayesian,1)
    wprecision(i)=TP(i)./(TP(i)+FP(i)).*sum(cf_Bayesian(i,:))./sum(sum(cf_Bayesian));
end
weighted_precision_Bayesian=sum(wprecision);
% Recall
mrecall=[];
for i=1:size(cf_Bayesian,1)
    mrecall(i)=TP(i)./(TP(i)+FN(i))./size(cf_Bayesian,1);
end
macro_recall_Bayesian=sum(mrecall);
wrecall=[];
for i=1:size(cf_Bayesian,1)
    wrecall(i)=TP(i)./(TP(i)+FN(i)).*sum(cf_Bayesian(i,:))./sum(sum(cf_Bayesian));
end
weighted_recall_Bayesian=sum(wrecall);
% F1-score
macro_f1_Bayesian=0;weighted_f1_Bayesian=0;
for i=1:size(cf_Bayesian,1)
    macro_f1_Bayesian=macro_f1_Bayesian+2.*mprecision(i).*mrecall(i)./(mprecision(i)+mrecall(i));
    weighted_f1_Bayesian=weighted_f1_Bayesian+2.*wprecision(i).*wrecall(i)./(wprecision(i)+wrecall(i));
end

[x,y,~,auc] = perfcurve(test_labels, yfit_Bayesian, 1);
plot(x, y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ' num2str(auc) ')']);

% 2. LDA
tic;
yfit_LDA=LDA.predictFcn(test_feature); 
LDA_time = toc;
save('yfit_LDA.mat',"yfit_LDA");
gaptest=yfit_LDA-test_labels;
LDA_acc=length(find(gaptest==0))./size(test_feature,1);
%Confusion matrix
cf_LDA=confusionmat(yfit_LDA,test_labels);
% Precision
TP=diag(cf_LDA);
FN=[];FP=[];
for i=1:size(cf_LDA,1)
    FN(i,1)=sum(cf_LDA(i,:))-cf_LDA(i,i);
    FP(i,1)=sum(cf_LDA(:,i))-cf_LDA(i,i);
end
mprecision=[];
for i=1:size(cf_LDA,1)
    mprecision(i)=TP(i)./(TP(i)+FP(i))./size(cf_LDA,1);
end
macro_precision_LDA=sum(mprecision);
wprecision=[];
for i=1:size(cf_LDA,1)
    wprecision(i)=TP(i)./(TP(i)+FP(i)).*sum(cf_LDA(i,:))./sum(sum(cf_LDA));
end
weighted_precision_LDA=sum(wprecision);
% Recall
mrecall=[];
for i=1:size(cf_LDA,1)
    mrecall(i)=TP(i)./(TP(i)+FN(i))./size(cf_LDA,1);
end
macro_recall_LDA=sum(mrecall);
wrecall=[];
for i=1:size(cf_LDA,1)
    wrecall(i)=TP(i)./(TP(i)+FN(i)).*sum(cf_LDA(i,:))./sum(sum(cf_LDA));
end
weighted_recall_LDA=sum(wrecall);
% F1-score
macro_f1_LDA=0;weighted_f1_LDA=0;
for i=1:size(cf_LDA,1)
    macro_f1_LDA=macro_f1_LDA+2.*mprecision(i).*mrecall(i)./(mprecision(i)+mrecall(i));
    weighted_f1_LDA=weighted_f1_LDA+2.*wprecision(i).*wrecall(i)./(wprecision(i)+wrecall(i));
end


% 3. QDA
% yfit_QDA=QDA.predictFcn(test_feature); 
% gaptest=yfit_QDA-test_labels;
% QDA_acc=length(find(gaptest==0))./size(test_feature,1);
% %Confusion matrix
% cf_QDA=confusionmat(yfit_QDA,test_labels);
% % Precision
% TP=diag(cf_QDA);
% FN=[];FP=[];
% for i=1:size(cf_QDA,1)
%     FN(i,1)=sum(cf_QDA(i,:))-cf_QDA(i,i);
%     FP(i,1)=sum(cf_QDA(:,i))-cf_QDA(i,i);
% end
% mprecision=[];
% for i=1:size(cf_QDA,1)
%     mprecision(i)=TP(i)./(TP(i)+FP(i))./size(cf_QDA,1);
% end
% macro_precision_QDA=sum(mprecision);
% wprecision=[];
% for i=1:size(cf_QDA,1)
%     wprecision(i)=TP(i)./(TP(i)+FP(i)).*sum(cf_QDA(i,:))./sum(sum(cf_QDA));
% end
% weighted_precision_QDA=sum(wprecision);
% % Recall
% mrecall=[];
% for i=1:size(cf_QDA,1)
%     mrecall(i)=TP(i)./(TP(i)+FN(i))./size(cf_QDA,1);
% end
% macro_recall_QDA=sum(mrecall);
% wrecall=[];
% for i=1:size(cf_QDA,1)
%     wrecall(i)=TP(i)./(TP(i)+FN(i)).*sum(cf_QDA(i,:))./sum(sum(cf_QDA));
% end
% weighted_recall_QDA=sum(wrecall);
% % F1-score
% macro_f1_QDA=0;weighted_f1_QDA=0;
% for i=1:size(cf_QDA,1)
%     macro_f1_QDA=macro_f1_QDA+2.*mprecision(i).*mrecall(i)./(mprecision(i)+mrecall(i));
%     weighted_f1_QDA=weighted_f1_QDA+2.*wprecision(i).*wrecall(i)./(wprecision(i)+wrecall(i));
% end


% 4. Linear SVM
tic;
yfit_LinearSVM=LinearSVM.predictFcn(test_feature);
LinearSVM_time = toc;
save('yfit_LinearSVM.mat',"yfit_LinearSVM");
gaptest=yfit_LinearSVM-test_labels;
LinearSVM_acc=length(find(gaptest==0))./size(test_feature,1);
%Confusion matrix
cf_LinearSVM=confusionmat(yfit_LinearSVM,test_labels);
% Precision
TP=diag(cf_LinearSVM);
FN=[];FP=[];
for i=1:size(cf_LinearSVM,1)
    FN(i,1)=sum(cf_LinearSVM(i,:))-cf_LinearSVM(i,i);
    FP(i,1)=sum(cf_LinearSVM(:,i))-cf_LinearSVM(i,i);
end
mprecision=[];
for i=1:size(cf_LinearSVM,1)
    mprecision(i)=TP(i)./(TP(i)+FP(i))./size(cf_LinearSVM,1);
end
macro_precision_LinearSVM=sum(mprecision);
wprecision=[];
for i=1:size(cf_LinearSVM,1)
    wprecision(i)=TP(i)./(TP(i)+FP(i)).*sum(cf_LinearSVM(i,:))./sum(sum(cf_LinearSVM));
end
weighted_precision_LinearSVM=sum(wprecision);
% Recall
mrecall=[];
for i=1:size(cf_LinearSVM,1)
    mrecall(i)=TP(i)./(TP(i)+FN(i))./size(cf_LinearSVM,1);
end
macro_recall_LinearSVM=sum(mrecall);
wrecall=[];
for i=1:size(cf_LinearSVM,1)
    wrecall(i)=TP(i)./(TP(i)+FN(i)).*sum(cf_LinearSVM(i,:))./sum(sum(cf_LinearSVM));
end
weighted_recall_LinearSVM=sum(wrecall);
% F1-score
macro_f1_LinearSVM=0;weighted_f1_LinearSVM=0;
for i=1:size(cf_LinearSVM,1)
    macro_f1_LinearSVM=macro_f1_LinearSVM+2.*mprecision(i).*mrecall(i)./(mprecision(i)+mrecall(i));
    weighted_f1_LinearSVM=weighted_f1_LinearSVM+2.*wprecision(i).*wrecall(i)./(wprecision(i)+wrecall(i));
end

% 5. Gaussian SVM
tic;
yfit_GaussianSVM=GaussianSVM.predictFcn(test_feature);
GaussianSVM_time = toc;
save('yfit_GaussianSVM.mat',"yfit_GaussianSVM");
gaptest=yfit_GaussianSVM-test_labels;
GaussianSVM_acc=length(find(gaptest==0))./size(test_feature,1);
%Confusion matrix
cf_GaussianSVM=confusionmat(yfit_GaussianSVM,test_labels);
% Precision
TP=diag(cf_GaussianSVM);
FN=[];FP=[];
for i=1:size(cf_GaussianSVM,1)
    FN(i,1)=sum(cf_GaussianSVM(i,:))-cf_GaussianSVM(i,i);
    FP(i,1)=sum(cf_GaussianSVM(:,i))-cf_GaussianSVM(i,i);
end
mprecision=[];
for i=1:size(cf_GaussianSVM,1)
    mprecision(i)=TP(i)./(TP(i)+FP(i))./size(cf_GaussianSVM,1);
end
macro_precision_GaussianSVM=sum(mprecision);
wprecision=[];
for i=1:size(cf_GaussianSVM,1)
    wprecision(i)=TP(i)./(TP(i)+FP(i)).*sum(cf_GaussianSVM(i,:))./sum(sum(cf_GaussianSVM));
end
weighted_precision_GaussianSVM=sum(wprecision);
% Recall
mrecall=[];
for i=1:size(cf_GaussianSVM,1)
    mrecall(i)=TP(i)./(TP(i)+FN(i))./size(cf_GaussianSVM,1);
end
macro_recall_GaussianSVM=sum(mrecall);
wrecall=[];
for i=1:size(cf_GaussianSVM,1)
    wrecall(i)=TP(i)./(TP(i)+FN(i)).*sum(cf_GaussianSVM(i,:))./sum(sum(cf_GaussianSVM));
end
weighted_recall_GaussianSVM=sum(wrecall);
% F1-score
macro_f1_GaussianSVM=0;weighted_f1_GaussianSVM=0;
for i=1:size(cf_GaussianSVM,1)
    macro_f1_GaussianSVM=macro_f1_GaussianSVM+2.*mprecision(i).*mrecall(i)./(mprecision(i)+mrecall(i));
    weighted_f1_GaussianSVM=weighted_f1_GaussianSVM+2.*wprecision(i).*wrecall(i)./(wprecision(i)+wrecall(i));
end

% 6. kNN
tic;
yfit_kNN=kNN.predictFcn(test_feature);
kNN_time = toc;
save('yfit_kNN.mat','yfit_kNN');
gaptest=yfit_kNN-test_labels;
kNN_acc=length(find(gaptest==0))./size(test_feature,1);
%Confusion matrix
cf_kNN=confusionmat(yfit_kNN,test_labels);
% Precision
TP=diag(cf_kNN);
FN=[];FP=[];
for i=1:size(cf_kNN,1)
    FN(i,1)=sum(cf_kNN(i,:))-cf_kNN(i,i);
    FP(i,1)=sum(cf_kNN(:,i))-cf_kNN(i,i);
end
mprecision=[];
for i=1:size(cf_kNN,1)
    mprecision(i)=TP(i)./(TP(i)+FP(i))./size(cf_kNN,1);
end
macro_precision_kNN=sum(mprecision);
wprecision=[];
for i=1:size(cf_kNN,1)
    wprecision(i)=TP(i)./(TP(i)+FP(i)).*sum(cf_kNN(i,:))./sum(sum(cf_kNN));
end
weighted_precision_kNN=sum(wprecision);
% Recall
mrecall=[];
for i=1:size(cf_kNN,1)
    mrecall(i)=TP(i)./(TP(i)+FN(i))./size(cf_kNN,1);
end
macro_recall_kNN=sum(mrecall);
wrecall=[];
for i=1:size(cf_kNN,1)
    wrecall(i)=TP(i)./(TP(i)+FN(i)).*sum(cf_kNN(i,:))./sum(sum(cf_kNN));
end
weighted_recall_kNN=sum(wrecall);
% F1-score
macro_f1_kNN=0;weighted_f1_kNN=0;
for i=1:size(cf_kNN,1)
    macro_f1_kNN=macro_f1_kNN+2.*mprecision(i).*mrecall(i)./(mprecision(i)+mrecall(i));
    weighted_f1_kNN=weighted_f1_kNN+2.*wprecision(i).*wrecall(i)./(wprecision(i)+wrecall(i));
end

% 7. Decision Tree
tic;
yfit_DecisionTree=DecisionTree.predictFcn(test_feature);
DecisionTree_time = toc;
save('yfit_DecisionTree.mat','yfit_DecisionTree');
gaptest=yfit_DecisionTree-test_labels;
DecisionTree_acc=length(find(gaptest==0))./size(test_feature,1);
%Confusion matrix
cf_DecisionTree=confusionmat(yfit_DecisionTree,test_labels);
% Precision
TP=diag(cf_DecisionTree);
FN=[];FP=[];
for i=1:size(cf_DecisionTree,1)
    FN(i,1)=sum(cf_DecisionTree(i,:))-cf_DecisionTree(i,i);
    FP(i,1)=sum(cf_DecisionTree(:,i))-cf_DecisionTree(i,i);
end
mprecision=[];
for i=1:size(cf_DecisionTree,1)
    mprecision(i)=TP(i)./(TP(i)+FP(i))./size(cf_DecisionTree,1);
end
macro_precision_DecisionTree=sum(mprecision);
wprecision=[];
for i=1:size(cf_DecisionTree,1)
    wprecision(i)=TP(i)./(TP(i)+FP(i)).*sum(cf_DecisionTree(i,:))./sum(sum(cf_DecisionTree));
end
weighted_precision_DecisionTree=sum(wprecision);
% Recall
mrecall=[];
for i=1:size(cf_DecisionTree,1)
    mrecall(i)=TP(i)./(TP(i)+FN(i))./size(cf_DecisionTree,1);
end
macro_recall_DecisionTree=sum(mrecall);
wrecall=[];
for i=1:size(cf_DecisionTree,1)
    wrecall(i)=TP(i)./(TP(i)+FN(i)).*sum(cf_DecisionTree(i,:))./sum(sum(cf_DecisionTree));
end
weighted_recall_DecisionTree=sum(wrecall);
% F1-score
macro_f1_DecisionTree=0;weighted_f1_DecisionTree=0;
for i=1:size(cf_DecisionTree,1)
    macro_f1_DecisionTree=macro_f1_DecisionTree+2.*mprecision(i).*mrecall(i)./(mprecision(i)+mrecall(i));
    weighted_f1_DecisionTree=weighted_f1_DecisionTree+2.*wprecision(i).*wrecall(i)./(wprecision(i)+wrecall(i));
end

% 8. Bagging Tree
tic;
yfit_BaggingTree=BaggingTree.predictFcn(test_feature);
BaggingTree_time = toc;
save('yfit_BaggingTree',"yfit_BaggingTree");
gaptest=yfit_BaggingTree-test_labels;
BaggingTree_acc=length(find(gaptest==0))./size(test_feature,1);
%Confusion matrix
cf_BaggingTree=confusionmat(yfit_BaggingTree,test_labels);
% Precision
TP=diag(cf_BaggingTree);
FN=[];FP=[];
for i=1:size(cf_BaggingTree,1)
    FN(i,1)=sum(cf_BaggingTree(i,:))-cf_BaggingTree(i,i);
    FP(i,1)=sum(cf_BaggingTree(:,i))-cf_BaggingTree(i,i);
end
mprecision=[];
for i=1:size(cf_BaggingTree,1)
    mprecision(i)=TP(i)./(TP(i)+FP(i))./size(cf_BaggingTree,1);
end
macro_precision_BaggingTree=sum(mprecision);
wprecision=[];
for i=1:size(cf_BaggingTree,1)
    wprecision(i)=TP(i)./(TP(i)+FP(i)).*sum(cf_BaggingTree(i,:))./sum(sum(cf_BaggingTree));
end
weighted_precision_BaggingTree=sum(wprecision);
% Recall
mrecall=[];
for i=1:size(cf_BaggingTree,1)
    mrecall(i)=TP(i)./(TP(i)+FN(i))./size(cf_BaggingTree,1);
end
macro_recall_BaggingTree=sum(mrecall);
wrecall=[];
for i=1:size(cf_BaggingTree,1)
    wrecall(i)=TP(i)./(TP(i)+FN(i)).*sum(cf_BaggingTree(i,:))./sum(sum(cf_BaggingTree));
end
weighted_recall_BaggingTree=sum(wrecall);
% F1-score
macro_f1_BaggingTree=0;weighted_f1_BaggingTree=0;
for i=1:size(cf_BaggingTree,1)
    macro_f1_BaggingTree=macro_f1_BaggingTree+2.*mprecision(i).*mrecall(i)./(mprecision(i)+mrecall(i));
    weighted_f1_BaggingTree=weighted_f1_BaggingTree+2.*wprecision(i).*wrecall(i)./(wprecision(i)+wrecall(i));
end

disp(['Bayesian_inference_time', num2str(Bayesian_time), 's']);
disp(['LDA_inference_time', num2str(LDA_time), 's']);
disp(['LinearSVM_inference_time', num2str(LinearSVM_time), 's']);
disp(['GaussianSVM_inference_time', num2str(GaussianSVM_time), 's']);
disp(['kNN_inference_time', num2str(kNN_time), 's']);
disp(['DecisionTree_inference_time', num2str(DecisionTree_time), 's']);
disp(['BaggingTree_inference_time', num2str(BaggingTree_time), 's']);

load('pred_Y.mat');
load('yfit_Bayesian.mat');
load('yfit_LDA');
load('yfit_LinearSVM');
load('yfit_GaussianSVM');
load('yfit_kNN');
load('yfit_DecisionTree');
load('yfit_BaggingTree');

figure();
[x,y,~,auc1] = perfcurve(test_labels, pred_Y_test, 1);
plot(x, y,'LineWidth',1.5,'LineStyle','-','Color','r');
hold on;
[x,y,~,auc2] = perfcurve(test_labels, yfit_Bayesian, 1);
plot(x, y,'LineWidth',1.5,'LineStyle','--');
hold on;
[x,y,~,auc3] = perfcurve(test_labels, yfit_LDA, 1);
plot(x, y,'LineWidth',1.5,'LineStyle','-.');
hold on;
% [x,y,~,auc4] = perfcurve(test_labels, yfit_LinearSVM, 1);
% plot(x, y,'LineWidth',1.5,'LineStyle','-');
% hold on;
[x,y,~,auc5] = perfcurve(test_labels, yfit_GaussianSVM, 1);
plot(x, y,'LineWidth',1.5,'LineStyle','--','Color','c');
hold on;
[x,y,~,auc6] = perfcurve(test_labels, yfit_kNN, 1);
plot(x, y,'LineWidth',1.5,'LineStyle','-.','Color','m');
hold on;
[x,y,~,auc7] = perfcurve(test_labels, yfit_DecisionTree, 1);
plot(x, y,'LineWidth',1.5,'LineStyle','-','Color','y');
hold on;
[x,y,~,auc8] = perfcurve(test_labels, yfit_BaggingTree, 1);
plot(x, y,'LineWidth',1.5,'LineStyle','--','Color','k');
hold on;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
set(gca,'FontName','Times New Roman','FontSize',12);
% title(['ROC Curve (AUC = ' num2str(auc1) ')']);
title(['ROC Curve']);
legend('DCDLN AUC=0.9750','Bayesian AUC=0.9628','LDA AUC=0.9683','GaussianSVM AUC=0.9622','KNN AUC=0.9698','DecisionTree AUC=0.9698','BaggingTree AUC=0.9707');






% 9. BPNN
% BPNN_output=csvread('veroutput.csv');
% for i=1:size(BPNN_output,1)
%     [yfit_BPNN(i,1),yfit_BPNN(i,2)]=max(BPNN_output(i,:));
% end
% gaptest=yfit_BPNN(:,2)-1-test_labels;
% BPNN_acc=length(find(gaptest==0))./size(test_feature,1);
% %Confusion matrix
% cf_BPNN=confusionmat(yfit_BPNN(:,2)-1,test_labels);
% % Precision
% TP=diag(cf_BPNN);
% FN=[];FP=[];
% for i=1:size(cf_BPNN,1)
%     FN(i,1)=sum(cf_BPNN(i,:))-cf_BPNN(i,i);
%     FP(i,1)=sum(cf_BPNN(:,i))-cf_BPNN(i,i);
% end
% mprecision=[];
% for i=1:size(cf_BPNN,1)
%     mprecision(i)=TP(i)./(TP(i)+FP(i))./size(cf_BPNN,1);
% end
% macro_precision_BPNN=sum(mprecision);
% wprecision=[];
% for i=1:size(cf_BPNN,1)
%     wprecision(i)=TP(i)./(TP(i)+FP(i)).*sum(cf_BPNN(i,:))./sum(sum(cf_BPNN));
% end
% weighted_precision_BPNN=sum(wprecision);
% % Recall
% mrecall=[];
% for i=1:size(cf_BPNN,1)
%     mrecall(i)=TP(i)./(TP(i)+FN(i))./size(cf_BPNN,1);
% end
% macro_recall_BPNN=sum(mrecall);
% wrecall=[];
% for i=1:size(cf_BPNN,1)
%     wrecall(i)=TP(i)./(TP(i)+FN(i)).*sum(cf_BPNN(i,:))./sum(sum(cf_BPNN));
% end
% weighted_recall_BPNN=sum(wrecall);
% % F1-score
% macro_f1_BPNN=0;weighted_f1_BPNN=0;
% for i=1:size(cf_BPNN,1)
%     macro_f1_BPNN=macro_f1_BPNN+2.*mprecision(i).*mrecall(i)./(mprecision(i)+mrecall(i));
%     weighted_f1_BPNN=weighted_f1_BPNN+2.*wprecision(i).*wrecall(i)./(wprecision(i)+wrecall(i));
% end