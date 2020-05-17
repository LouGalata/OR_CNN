% Define a data-augmentation strategy
% Set the test-set to 300, and activate the line (and comment the previous line)
% And in imageDataAugmenter(...) set 'RandRotation' only
% What do you observe?
% activate both 'RandRotation'and 'RandXTranslation'
% What do you observe?
% activate 'RandRotation', 'RandXTranslation' and ‘RandYTranslation'
% What do you observe?
% activate 'RandRotation', 'RandXTranslation' , ‘RandYTranslation', ‘RandYReflection’
% What do you observe?
% activate 'RandRotation', 'RandXTranslation' , ‘RandYTranslation', ‘RandYReflection’ and RandXReflection
% What do you observe?

training_sample = 300;
[XTrain, YTrain, XValidation, YValidation, XTest, YTest] = load_data(training_sample);


imageAugmenter_exp1 = imageDataAugmenter('RandRotation', [-20,20]);
imageAugmenter_exp2 = imageDataAugmenter('RandRotation', [-20,20], 'RandXTranslation', [-3 3]);
imageAugmenter_exp3 = imageDataAugmenter('RandRotation', [-20,20], 'RandXTranslation', [-3 3], 'RandYTranslation', [-3 3]);
imageAugmenter_exp4 = imageDataAugmenter('RandYReflection', true, 'RandRotation', [-20,20], 'RandXTranslation', [-3 3], 'RandYTranslation', [-3 3]);
imageAugmenter_exp5 = imageDataAugmenter('RandXReflection', true, 'RandYReflection', true, 'RandRotation', [-20,20], 'RandXTranslation', [-3 3], 'RandYTranslation', [-3 3]);

augmentation_strategy = containers.Map({1,2,3,4,5}, {imageAugmenter_exp1, imageAugmenter_exp2, imageAugmenter_exp3, imageAugmenter_exp4, imageAugmenter_exp5});

for key = keys(augmentation_strategy)
    imageSize = [28 28 1];
    augmend = augmentation_strategy(key{1});
    augimds = augmentedImageDatastore(imageSize,XTrain,YTrain,...
            'DataAugmentation',augmend);


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
end

