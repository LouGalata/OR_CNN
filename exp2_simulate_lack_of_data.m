% We are simulating the lack of data, by reducing the number of images in the training set to 1000, then to 500, then to 300.
% Please note that the “epocs_factor” is computed to compensate the limited number of samples.
% What do we observe in the performance? 
% How the training and validation curve behave? 
% What happens if we set “epocs_factor” to 1?

training_samples_list = {1000, 500, 300};
for training_sample = 1:length(training_samples_list)
    [XTrain, YTrain, XValidation, YValidation, XTest, YTest] = load_data(training_samples_list{training_sample});

    imageSize = [28 28 1];
    augimds = augmentedImageDatastore(imageSize,XTrain,YTrain);

    architecture = 'v1';
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