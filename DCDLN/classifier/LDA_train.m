load('data_densenet121.mat')
train_data = train_data_densenet121(2:end, :);
val_data = val_data(2:end, :);
X1 = [train_data;val_data];
v = val_data(2,:);
v_data = v(1:1024);
test_feature = val_data(:,1:1024);
test_labels = val_data(:,1025);


[trainedClassifier, validationAccuracy] = trainClassifier(train_data)
acc = validationAccuracy