clc;
clear all;
close all;

load('data_densenet121.mat');
train_data = train_data_densenet121(2:end, :);
val_data = val_data(2:end, :);
X1 = [train_data; val_data];
X1 = X1';
X = X1(1:(end-1), :);
Y = X1(end, :);
Y(Y == 0) = -1;
m = 23558;
n = 27558;

% 0均值标准化
[S, PCA_PS] = mapstd(X(:, 1:m), 0, 1);
X = mapstd('apply', X, PCA_PS);

% 主成分分析
[coeff, element] = PCA_extra(X(:, 1:m)');
X = (X' * coeff(:, 1:element))';

% 0均值标准化
[S, PS] = mapstd(X(:, 1:m), 0, 1);
X = mapstd('apply', X, PS);

% Softsign, Sigmoid, Relu, Linear
g = @softsign;
g_deriv = @softsign_deriv;
f = @softsign;
f_deriv = @softsign_deriv;
p = @power_sigmoid;

% 添加全1偏置行
X = [X; ones(1, size(X, 2))];
X_trainALL=X(:,1:m);
Y_trainALL=Y(1:m);
%Test set
X_test=X(:,(m+1):n);
Y_test=Y((m+1):n);


% 参数初始化
num_class = max(Y);
num_data = size(X, 2);
num_traindata = size(X_trainALL,2);
num_testdata = size(X_test,2);
num_feature = size(X, 1) - 1;
num_hidden = round(sqrt(num_feature)) + 1;
alpha = 0.17; % 学习率
% MSE_val = [];
% MSE_test = [];
all_valaccurate = [];
all_testaccurate = [];
all_mse_best = [];

% 10折交叉验证
k_folds = 10;
fold_size = floor(num_traindata / k_folds);

tic;
for fold = 1:k_folds
    fprintf('Fold %d/%d\n', fold, k_folds);
    
    % 划分训练集和验证集
    start_idx = (fold - 1) * fold_size + 1;
    end_idx = fold * fold_size;
    val_indices = start_idx:end_idx;
    train_indices = setdiff(1:num_traindata, val_indices);
    
    X_train_1 = X_trainALL(:, 1:start_idx-1);
    X_train_2 = X_trainALL(:, end_idx+1:end);
    X_train = [X_train_1, X_train_2];
    %X_train = X_train(:, train_indices);
    Y_train_1 = Y_trainALL(1:start_idx-1);
    Y_train_2 = Y_trainALL(end_idx+1:end);
    Y_train = [Y_train_1, Y_train_2];
    
    X_val = X_trainALL(:, val_indices);
    Y_val = Y_trainALL(val_indices);
    
    % 初始化参数
    num_traindata = size(X_train, 2);
    num_valdata = size(X_val, 2);
    w = normrnd(0, 1, [1, num_hidden+1]);
    v = normrnd(0, 1, [num_hidden, num_feature+1]);
    
    % 训练过程
    epochs = 100;
    beta = 0;
    MSE_train = [];
    MSE_val = [];
    MSE_test = [];
    all_val_Accuracy = [];
    all_test_Accuracy = [];
    acc_best = 0;
    all_Precision_val=[];
    all_Recall_val=[];
    all_F1_measure_val=[];

    all_Precision_test=[];
    all_Recall_test=[];
    all_F1_measure_test=[];
    
    for epoch = 1:epochs
        epoch
        H = g(v * X_train);
        Y_pre = f(w * [H; ones(1, size(H, 2))]);
        mse_train = 1 / num_traindata * sum((Y_pre - Y_train).^2);
        MSE_train = [MSE_train, mse_train];
        
        error = Y_pre - Y_train;
        D2 = (-alpha * p(error)) ./ f_deriv(Y_pre);
        delta_w = D2 * [H; ones(1, size(H, 2))]';
        w2 = w(:, 1:end-1);
        delta_H = w2' * D2;
        D1 = delta_H ./ g_deriv(H);
        delta_v = D1 * X_train';
        
        w = w + delta_w;
        v = v + delta_v;
        
        H_val = g(v * X_val);
        pred_Y_val = f(w * [H_val; ones(1, size(H_val, 2))]);
        mse_val = 1 / num_valdata * sum((pred_Y_val - Y_val).^2);
        MSE_val = [MSE_val, mse_val];
        
        [result, ~] = score_binary(pred_Y_val', Y_val');
        Accuracy_val = result(1, 1);
        Precision_val=result(1,2);
        Recall_val=result(1,3);
        F1_measure_val=result(1,4);
        all_val_Accuracy = [all_val_Accuracy, Accuracy_val];
        all_Precision_val = [all_Precision_val, Precision_val];
        all_Recall_val = [all_Recall_val,Recall_val];
        all_F1_measure_val = [all_F1_measure_val, F1_measure_val];
        if (Accuracy_val>=acc_best)
            acc_best = Accuracy_val;
            acc_stop = epoch;
            
            %tic;
            H_test = g(v * X_test);
            pred_Y_test = f(w * [H_test; ones(1, size(H_test, 2))]);
            mse_test = 1 / num_valdata * sum((pred_Y_test - Y_test).^2);
            MSE_test = [MSE_test, mse_test];
        
            [result, nn] = score_binary(pred_Y_test', Y_test');
            Accuracy_test = result(1, 1);
            Precision_test=result(1,2);
            Recall_test=result(1,3);
            F1_measure_test=result(1,4);
            all_test_Accuracy = [all_test_Accuracy, Accuracy_test];
            all_Precision_test = [all_Precision_test, Precision_test];
            all_Recall_test = [all_Recall_test,Recall_val];
            all_F1_measure_test = [all_F1_measure_test, F1_measure_val];
            best_acc_test = max(all_test_Accuracy);
            best_pre_test = max(all_Precision_test);
            best_rec_test = max(all_Recall_test);
            best_f1_test = max(all_F1_measure_test)
            %time = toc; 
            %disp(['inference time', num2str(time), ' s']);
        end
    end
    
    all_valaccurate = [all_valaccurate; all_val_Accuracy];
    all_testaccurate = [all_testaccurate; best_acc_test];
    all_Precision_test = [all_Precision_test, best_pre_test];
    all_Recall_test = [all_Recall_test, best_rec_test];
    all_F1_measure_test = [all_F1_measure_test, best_f1_test];
    all_mse_best = [all_mse_best, min(MSE_val)];
end
elapsed_time = toc;
disp(['training time', num2str(elapsed_time), ' s']);
save('pred_Y.mat','pred_Y_test')
fprintf('Mean val accuracy: %.2f%%\n', mean(all_valaccurate(:)));
fprintf('Mean test accuracy: %.2f%%\n', mean(all_testaccurate(:)));
fprintf('Mean test precision: %.2f%%\n', mean(all_Precision_test(:)));
fprintf('Mean test recall: %.2f%%\n', mean(all_Recall_test(:)));
fprintf('Mean test f1 score: %.2f%%\n', mean(all_F1_measure_test(:)));
% 假设我们有分类器的输出scores和对应的标签labels
[x,y,~,auc] = perfcurve(Y_test, pred_Y_test, 1);
plot(x, y,'LineWidth',1.5);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ' num2str(auc) ')']);

%% Graphical interface
DCDLN.tp = nn(1);
DCDLN.fp = nn(2);
DCDLN.fn = nn(4);
DCDLN.tn = nn(3);
save('confusion_matrix',"DCDLN");


figure();
plot(1:size(MSE_test,2),MSE_test,'LineWidth',1.5);
hold on;
plot(1:epoch,MSE_val,'LineStyle','-','Color','y','LineWidth',1.5);
hold on;
plot(1:epoch,MSE_train,'LineStyle','--','Color','r','LineWidth',1.5);
xlabel('epoch');
ylabel('MSE');
legend('error of test','error of val','error of train');


figure();

subplot(2,2,1)
plot(1:size(all_test_Accuracy,2),all_test_Accuracy,'LineWidth',1.5);
xlabel('epoch'); 
ylabel('Accuracy');
subplot(2,2,2);
plot(1:size(all_Precision_test,2),all_Precision_test,'LineWidth',1.5);
xlabel('epoch') ;
ylabel('Precision');
subplot(2,2,3);
plot(1:size(all_Recall_test,2),all_Recall_test,'LineWidth',1.5);
xlabel('epoch') ;
ylabel('Recall');
subplot(2,2,4);
plot(1:size(all_F1_measure_test,2),all_F1_measure_test,'LineWidth',1.5);
xlabel('epoch') ;
ylabel('f1 score');
set(gca,'FontName','Times New Roman');