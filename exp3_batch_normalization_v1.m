% The batch normalization improves the convergence of the training.
% Set the test-set to 1000, and comment the %batchNormalizationLayer in layers_v1 Then
% Set the test-set to 300, and comment the %batchNormalizationLayer in layers_v1
% What do we observe in the performance? 
% How the training and validation curve behave? 
% What happens if we set “epocs_factor” to 1?


testing_samples_list = {300};
for testing_sample = 1:length(testing_samples_list)
    % We defined val_size = train_size / 8. So,
    % Test = 1000 --> Train = 3500, Val = 500
    % Test = 300 --> Train = 4112, Val = 588
    [XTrain, YTrain, XValidation, YValidation, XTest, YTest] = load_data_given_test_set(testing_samples_list{testing_sample});

    imageSize = [28 28 1];
    augimds = augmentedImageDatastore(imageSize,XTrain,YTrain);

    architecture = 'v1_noBN';
    num_neurons = 32;
    num_epochs = 20;

    model = get_model(architecture, imageSize, num_neurons);
    
    epocs_factor = 1; % change it and observe performance
    options = trainingOptions('sgdm', 'MaxEpochs',20*epocs_factor, 'Shuffle',...
            'every-epoch', 'Verbose',true, 'Plots','training-progress', ...
            'ValidationData',{XValidation,YValidation}, ...
            'ValidationFrequency', 20 );

    net = trainNetwork(augimds,model,options);

    YPred = classify(net,XTest);

    accuracy = sum(YPred == YTest)/numel(YTest);
    
end