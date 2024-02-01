%二分类,加上训练集准确率

%% Clear environment variables
clc;       %清除命令窗口的内容，对工作环境中的全部变量无任何影响 
clear all; %清除工作空间的所有变量，函数，和MEX文件
close all; %关闭所有的Figure窗口
%% Load and process data, build neural network and label training set
load('data_densenet121.mat')
%load('data_rubbish1_densenet121.mat')
%load('weight_v_and_w_accurate.mat')
% Parkinson=Parkinson';
train_data = train_data_densenet121(2:end, :);
val_data = val_data(2:end, :);
X1 = [train_data;val_data];
X1 = X1';
X = X1(1:(end-1),:);
Y = X1(end,:);
Y(Y==0)=-1;
m=23558;n=27558;
%m=2637;n=3297;
%m=22564;n=25077;

% %-1到1归一化
% [A1,PS]=mapminmax(X(:,1:m));
% X= mapminmax('apply',X,PS);
%0均值标准化
[S,PCA_PS] = mapstd(X(:,1:m),0,1);
X = mapstd('apply',X,PCA_PS);
%主成分分析
[coeff,element]=PCA_extra(X(:,1:m)');
X=(X'*coeff(:,1:element))';
%0均值标准化
[S,PS] = mapstd(X(:,1:m),0,1);
X = mapstd('apply',X,PS);

%softsign  sigmoid  Relu  linear
g=@softsign;
g_deriv=@softsign_deriv;
f=@softsign;
f_deriv=@softsign_deriv;
p=@power_sigmoid;
% 添加全1偏置行
X=[X;ones(1,size(X,2))];
%Training set
X_train=X(:,1:m);
Y_train=Y(1:m);
%Test set
X_test=X(:,(m+1):n);
Y_test=Y((m+1):n);


%% init parameter
num_class=max(Y_test);
num_traindata=size(X_train,2);
num_testdata=size(X_test,2);
num_feature=size(X_train,1)-1;
num_hidden=round(sqrt(num_feature))+1;
alpha=0.17;         %保证最佳准确率（曲线平滑）下的最快学习率
MSE=[];
MSE_train=[];
all_accurate=[];
all_accurate_train=[];
%随机生产权重
%w1=0.4*random( 'uniform' ,-1, 1, num_hidden, num_feature );
w1=normrnd(0,1,[num_hidden, num_feature]);
b1=zeros( num_hidden,1);
% b1=random( 'uniform' ,0, 1, num_hidden,1 );
v=[w1,b1];
% v=best_wh.v_best;
%w2=0.4*random( 'uniform' ,-1, 1, 1, num_hidden );
w2=normrnd(0,1,[1, num_hidden]);
b2=zeros( 1,1);
% b2=random( 'uniform' ,0, 1, num_class, 1 );
w=[w2,b2];
% w=best_wh.w_best;
%记录最佳准确率及其权值
accurate_best=0;
Accuracy=0;
Precision=0;
Recall=0;
F1_measure=0;
all_Accuracy=[];
all_Precision=[];
all_Recall=[];
all_F1_measure=[];
%记录最佳均方误差及其权值
mse_best=10;
acc_best = 0;
all_mse_best=[];
all_error_dim1 = [];
all_error_dim10 = [];
all_error_dim100 = [];
% all_error_dim1000 = [];
% all_error_dim10000 = [];
% all_error_dim20000 = [];
all_error_dim500 = [];
all_error_dim1000 = [];
all_error_dim1500 = [];
%all_error_dim2000 = [];

%% The process of training
epochs=4000;
beta=0;
tic;
for epoch = 1:epochs
    epoch;
    H = g(v * X_train) ;  
    Y_pre=f(w*[H;ones(1,size(H,2))]);
    mse_train= 1 / num_traindata * sum((Y_pre - Y_train).^ 2);
    MSE_train=[MSE_train,mse_train];

%     regu=repmat(w'*w/size(X_train,1),size(X_train,1),1);
    error=Y_pre-Y_train;  %+beta*regu
    all_error_dim1 = [all_error_dim1,error(1)];
    all_error_dim10 = [all_error_dim10,error(10)];
    all_error_dim100 = [all_error_dim100,error(100)];
%     all_error_dim1000 = [all_error_dim1000,error(1000)];
%     all_error_dim10000 = [all_error_dim10000,error(10000)];
%     all_error_dim20000 = [all_error_dim20000,error(20000)];

    all_error_dim500 = [all_error_dim500,error(500)];
    all_error_dim1000 = [all_error_dim1000,error(1000)];
    all_error_dim1500 = [all_error_dim1500,error(1500)];
    %all_error_dim2000 = [all_error_dim2000,error(2000)];

    D2=(-alpha*p(error))./f_deriv(Y_pre);
    delta_w=D2*pinv([H;ones(1,size(H,2))]);
    w2=w(:,1:end-1);
    delta_H= pinv(w2) *D2;
    D1=delta_H./g_deriv(H);
    
    delta_v=D1*pinv(X_train);
    %隐动力学方程固定v求权值w
    w=w+delta_w;
    v=v+delta_v;

    % -----------------------预测收集每一个权值矩阵下的均方误差和准确率（之后删掉）---------------------------
    % 使用均方误差方法验证
    tic;
    H_test=g( v*X_test);
    pred_Y = f(w*[H_test;ones(1,size(H_test,2))]);   %+ b
    mse = 1 / num_testdata * sum((pred_Y - Y_test).^ 2);
    MSE=[MSE,mse];
    %使用四个指标进行验证
    [result nn]=score_binary(pred_Y',Y_test');
    Accuracy=result(1,1)
    Precision=result(1,2);
    Recall=result(1,3);
    F1_measure=result(1,4);
    time = toc; 
    disp(['inference time', num2str(time), ' s']);
    all_Accuracy=[all_Accuracy,Accuracy];
    all_Precision=[all_Precision,Precision];
    all_Recall=[all_Recall,Recall];
    all_F1_measure=[all_F1_measure,F1_measure];

    %选出均方误差最佳的权值；记录最佳准确率下预测结果
    if (mse<=mse_best)
       v_best_mse=v;
       w_best_mse=w;
       v_best_accurate=v;
       w_best_accurate=w;
       accurate_best=Accuracy;
       mse_best=mse;
       all_mse_best=[all_mse_best,mse_best];
       pred_Y_best=pred_Y;
       best_epoch=epoch;
    end
    if (Accuracy>=acc_best)
        acc_best = Accuracy;
        acc_stop = epoch;
    end

    % ---------------------------------------------------------
    if best_epoch<epoch-30
        save('pred_Y.mat','pred_Y')
       break;
    end
end
elapsed_time = toc;
disp(['training time', num2str(elapsed_time), ' s']);

% 假设我们有分类器的输出scores和对应的标签labels
[x,y,~,auc] = perfcurve(Y_test, pred_Y, 1);
plot(x, y,'LineWidth',1.5);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ' num2str(auc) ')']);

%% Graphical interface
DCDNN.tp = nn(1);
DCDNN.fp = nn(2);
DCDNN.fn = nn(4);
DCDNN.tn = nn(3);
save('confusion_matrix',"DCDNN");


figure();
plot(1:epoch,MSE,'LineWidth',1.5);
hold on;
plot(1:epoch,MSE_train,'LineStyle','--','Color','r','LineWidth',1.5);
xlabel('epoch');
ylabel('MSE');
legend('error of test','error of train');


figure();

subplot(2,2,1)
plot(1:acc_stop,all_Accuracy(1:acc_stop),'LineWidth',1.5);
xlabel('epoch'); 
ylabel('Accuracy');
subplot(2,2,2);
plot(1:acc_stop,all_Precision(1:acc_stop),'LineWidth',1.5);
xlabel('epoch') ;
ylabel('Precision');
subplot(2,2,3);
plot(1:acc_stop,all_Recall(1:acc_stop),'LineWidth',1.5);
xlabel('epoch') ;
ylabel('Recall');
subplot(2,2,4);
plot(1:acc_stop,all_F1_measure(1:acc_stop),'LineWidth',1.5);
xlabel('epoch') ;
ylabel('F1_measure');
set(gca,'FontName','Times New Roman');


figure();

subplot(2,2,1);
plot(1:epoch,all_Accuracy,'LineWidth',1.5);
xlabel('epoch'); 
ylabel('Accuracy');
subplot(2,2,2);
plot(1:epoch,all_Precision,'LineWidth',1.5);
xlabel('epoch') ;
ylabel('Precision');
subplot(2,2,3);
plot(1:epoch,all_Recall,'LineWidth',1.5);
xlabel('epoch') ;
ylabel('Recall');
subplot(2,2,4);
plot(1:epoch,all_F1_measure,'LineWidth',1.5);
xlabel('epoch') ;
ylabel('F1_measure');
set(gca,'FontName','Times New Roman');

figure();
subplot(2,3,1);
plot(1:epoch,all_error_dim1,'LineWidth',1.5);
xlabel('epoch'); 
ylabel('dimension1');
set(gca,'FontName','Times New Roman','FontSize',12);

subplot(2,3,2);
plot(1:epoch,all_error_dim10,'LineWidth',1.5);
xlabel('epoch') ;
ylabel('dimension10');
set(gca,'FontName','Times New Roman','FontSize',12);

subplot(2,3,3);
plot(1:epoch,all_error_dim100,'LineWidth',1.5);
xlabel('epoch') ;
ylabel('dimension100');
set(gca,'FontName','Times New Roman','FontSize',12);

subplot(2,3,4);
plot(1:epoch,all_error_dim500,'LineWidth',1.5);
xlabel('epoch') ;
ylabel('dimension500');
set(gca,'FontName','Times New Roman','FontSize',12);

subplot(2,3,5);
plot(1:epoch,all_error_dim1000,'LineWidth',1.5);
xlabel('epoch') ;
ylabel('dimension1000');
set(gca,'FontName','Times New Roman','FontSize',12);

subplot(2,3,6);
plot(1:epoch,all_error_dim1500,'LineWidth',1.5);
xlabel('epoch') ;
ylabel('dimension1500');
set(gca,'FontName','Times New Roman','FontSize',12);
sgtitle('Error convergence curve for some dimensions');

figure();
subplot(3,2,1);
plot(1:epoch,all_error_dim1,'LineWidth',1.5);
xlabel('epoch'); 
ylabel('dimension1');
set(gca,'FontName','Times New Roman','FontSize',10.5);

subplot(3,2,2);
plot(1:epoch,all_error_dim10,'LineWidth',1.5);
xlabel('epoch') ;
ylabel('dimension10');
set(gca,'FontName','Times New Roman','FontSize',10.5);

subplot(3,2,3);
plot(1:epoch,all_error_dim100,'LineWidth',1.5);
xlabel('epoch') ;
ylabel('dimension100');
set(gca,'FontName','Times New Roman','FontSize',10.5);

subplot(3,2,4);
plot(1:epoch,all_error_dim500,'LineWidth',1.5);
xlabel('epoch') ;
ylabel('dimension500');
set(gca,'FontName','Times New Roman','FontSize',10.5);

subplot(3,2,5);
plot(1:epoch,all_error_dim1000,'LineWidth',1.5);
xlabel('epoch') ;
ylabel('dimension1000');
set(gca,'FontName','Times New Roman','FontSize',10.5);

subplot(3,2,6);
plot(1:epoch,all_error_dim1500,'LineWidth',1.5);
xlabel('epoch') ;
ylabel('dimension1500');
set(gca,'FontName','Times New Roman','FontSize',10.5);
sgtitle('Error convergence curve for some dimensions');

figure();
plot(1:epoch,all_error_dim1,'LineStyle','-','Color','r');
hold on;
plot(1:epoch,all_error_dim10,'LineStyle','--','Color','g','Marker','*');
hold on;
plot(1:epoch,all_error_dim100,'LineStyle','-.','Color','b','Marker','+');
hold on;
plot(1:epoch,all_error_dim500,'LineStyle','--','Color','c','Marker','.');
hold on;
plot(1:epoch,all_error_dim1000,'LineStyle','-','Color','m','Marker','pentagram');
hold on;
plot(1:epoch,all_error_dim1500,'LineStyle','--','Color','y','Marker','diamond');
xlabel('epoch');
ylabel('six dimensions of error');
legend('first dimension of errors','10th dimension of errors','100th dimension of errors','500th dimension of errors','1000th dimension of errors','1500th dimension of errors');
set(gca,'FontName','Times New Roman');

figure();
plot(1:epoch,all_error_dim1,'LineStyle','-','Color','r');
hold on;
plot(1:epoch,all_error_dim10,'LineStyle','--','Color','g');
hold on;
plot(1:epoch,all_error_dim100,'LineStyle','-.','Color','b');
hold on;
plot(1:epoch,all_error_dim500,'LineStyle','--','Color','c');
hold on;
plot(1:epoch,all_error_dim1000,'LineStyle','-','Color','m');
hold on;
plot(1:epoch,all_error_dim1500,'LineStyle','--','Color','y');
xlabel('epoch');
ylabel('six dimensions of error');
legend('first dimension of errors','10th dimension of errors','100th dimension of errors','500th dimension of errors','1000th dimension of errors','1500th dimension of errors');
set(gca,'FontName','Times New Roman');

%%保存权值
v=v_best_accurate;w=w_best_accurate;
best_wh_2.v_best = v;
best_wh_2.w_best = w;
save('weight_v_and_w_accurate',"best_wh_2")
% v=v_best_mse;w=w_best_mse;
% save('weight_v_and_w_mse','v','w')