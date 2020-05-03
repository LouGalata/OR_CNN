% Execute for the first time the script, understand the plot curves and observe the computed accuracy.
% At this time the data augmentation is not activated, 
% the selected network is _v1, 
% the number of epocs is 20,
% the number of samples used for training is 4000.
% Observe the Test accuracy computed at the end, observe the validation accuracy on the plot.

training_sample = 4000;
[XTrain, YTrain, XValidation, YValidation, XTest, YTest] = load_data(training_sample);

imageSize = [28 28 1];
augimds = augmentedImageDatastore(imageSize,XTrain,YTrain);

architecture = 'v1';
num_neurons = 32;
num_epochs = 20;

model = get_model(architecture, imageSize, num_neurons);

epocs_factor = 1;
options = trainingOptions('sgdm', 'MaxEpochs',20*epocs_factor, 'Shuffle',...
        'every-epoch', 'Verbose',true, 'Plots','training-progress', ...
        'ValidationData',{XValidation,YValidation}, ...
        'ValidationFrequency', 20 );
    
net = trainNetwork(augimds,model,options);

YPred = classify(net,XTest);

accuracy = sum(YPred == YTest)/numel(YTest);
