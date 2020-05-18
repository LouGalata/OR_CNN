% We are simulating the lack of data, by reducing the number of images in the training set to 1000, then to 500, then to 300.
% Please note that the â€œepocs_factorâ€? is computed to compensate the limited number of samples.
% What do we observe in the performance? 
% How the training and validation curve behave? 
% What happens if we set â€œepocs_factorâ€? to 1?

training_samples_list = {300};
for training_sample = 1:length(training_samples_list)
    [XTrain, YTrain, XValidation, YValidation, XTest, YTest] = load_data(training_samples_list{training_sample});

    imageSize = [28 28 1];
    augimds = augmentedImageDatastore(imageSize,XTrain,YTrain);

    architecture = 'v2';
    num_neurons = 32;
    num_epochs = 20;

    model = get_model(architecture, imageSize, num_neurons);

    %epochs_factor = 1; % change it and observe performance
    epochs_factor = 4000/training_samples_list{training_sample};
    options = trainingOptions('sgdm', 'MaxEpochs',cast(20*epochs_factor, 'int32'), 'Shuffle',...
            'every-epoch', 'Verbose',true, 'Plots','training-progress', ...
            'ValidationData',{XValidation,YValidation}, ...
            'ValidationFrequency', 20 );

    [net, info] = trainNetwork(augimds,model,options);
    save("results\v2_samples="+training_samples_list{training_sample}+"-ef="+epochs_factor+".mat", 'info')
   
    YPred = classify(net,XTest);

    accuracy = sum(YPred == YTest)/numel(YTest);
end