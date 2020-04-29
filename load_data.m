function [XTrain, YTrain, XValidation, YValidation, XTest, YTest] = load_data()
    [XTrain,YTrain] = digitTrain4DArrayData;
    Training_n_samples=4000;
    Dataset_n_samples= size(XTrain,4);
    epocs_factor=round(Dataset_n_samples/Training_n_samples);
    Non_training_samples=Dataset_n_samples-Training_n_samples;

    idx = randperm(size(XTrain,4),Non_training_samples);
    XValidation = XTrain(:,:,:,idx(1:round(Non_training_samples/2)));
    XTest = XTrain(:,:,:,idx(round(Non_training_samples/2)+1:Non_training_samples));
    XTrain(:,:,:,idx) = [];
    YValidation = YTrain(idx(1:round(Non_training_samples/2)));
    YTest = YTrain(idx(round(Non_training_samples/2)+1:Non_training_samples));
    YTrain(idx) = [];
end