%unzip LogoDet-3K.zip
%data = load("vehicleDatasetGroundTruth.mat");
%logoDataset = data.vehicleDataset;");
%Load files here!

%% Generate training, validation, and test sets
rng("default");
shuffledIndices = randperm(height(logoDataset));
idx = floor(0.6 * length(shuffledIndices) );

trainingIdx = 1:idx;
trainingDataTbl = logoDataset(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = logoDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = logoDataset(shuffledIndices(testIdx),:);

%% Saving off data
save('Data.mat' , 'trainingDataTbl', 'validationDataTbl', 'testDataTbl');

%% Generate datastores
imgTrain = imageDatastore(trainingDataTbl{:,"imageFilename"});
boxTrain = boxLabelDatastore(trainingDataTbl(:,"logo"));

imgValidation = imageDatastore(validationDataTbl{:,"imageFilename"});
boxValidation = boxLabelDatastore(validationDataTbl(:,"logo"));

imgTest = imageDatastore(testDataTbl{:,"imageFilename"});
boxTest = boxLabelDatastore(testDataTbl(:,"logo"));

trainingData = combine(imgTrain,boxTrain);
validationData = combine(imgValidation,boxValidation);
testData = combine(imgTest,boxTest);

%% Validate training set
validateInputData(trainingData);
validateInputData(validationData);
validateInputData(testData);

%% Display example image
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,"Rectangle",bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%% Prep network
inputSize = [227 227 3];
className = "logo";
rng("default")
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 9;
[anchors,meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);
area = anchors(:, 1).*anchors(:,2);
[~,idx] = sort(area,"descend");

anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    anchors(7:9,:)
    };
detector = yolov4ObjectDetector("csp-darknet53-coco",className,anchorBoxes,InputSize=inputSize);
augmentedTrainingData = transform(trainingData,@augmentData);
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},"rectangle",data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,BorderSize=10)
options = trainingOptions("adam",...
    GradientDecayFactor=0.9,...
    SquaredGradientDecayFactor=0.999,...
    InitialLearnRate=0.001,...
    LearnRateSchedule="none",...
    MiniBatchSize=100,...
    L2Regularization=0.0005,...
    MaxEpochs=25,...
    BatchNormalizationStatistics="moving",...
    DispatchInBackground=true,...
    ResetInputNormalization=false,...
    Shuffle="every-epoch",...
    VerboseFrequency=20,...
    ValidationFrequency=1000,...
    CheckpointPath=tempdir,...
    ValidationData=validationData);
[detector,info] = trainYOLOv4ObjectDetector(augmentedTrainingData,detector,options);















