% Set the test-set to 300, and replace layers_v2 as network to train in trainNetwork(...);
% How the training and validation performance change?


training_sample = 300;
[XTrain, YTrain, XValidation, YValidation, XTest, YTest] = load_data(training_sample);

imageSize = [28 28 1];
augimds = augmentedImageDatastore(imageSize,XTrain,YTrain);

architecture = 'v2';
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
