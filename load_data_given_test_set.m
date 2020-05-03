function [XTrain, YTrain, XValidation, YValidation, XTest, YTest] = load_data_given_test_set(test_set_samples)
    [XTrain,YTrain] = digitTrain4DArrayData;
    Testing_n_samples = test_set_samples;
    Dataset_n_samples= size(XTrain,4);
    epocs_factor=round(Dataset_n_samples/Testing_n_samples);
    Non_testing_samples=Dataset_n_samples-Testing_n_samples;

    idx = randperm(size(XTrain,4),Non_testing_samples);
    XTest = XTrain;
    YTest = YTrain;
    XValidation = XTrain(:,:,:,idx(1:round(Non_testing_samples/8)));
    YValidation = YTrain(idx(1:round(Non_testing_samples/8)));
    
    XTrain = XTrain(:,:,:,idx(round(Non_testing_samples/8)+1:Non_testing_samples));
    YTrain = YTrain(idx(round(Non_testing_samples/8)+1:Non_testing_samples));
    
    XTest(:,:,:,idx) = [];
    YTest(idx) = [];
    
end