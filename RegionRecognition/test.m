clear;
clc;
 
%Rebecca's path
%filePath = "C:\Users\brelanre\OneDrive - Rose-Hulman Institute of Technology\Documents\MATLAB\CSSE463\Transportation";
%Nathan's path
%filePath = "C:\Users\kingnm\Documents\MATLAB\Image Recognition\Logo-Recognition\RegionRecognition\Transportation";
%Matthew's path
filePath = "E:\Logos\LogoDet-3K\Transportation";

imds = imageDatastore(filePath, "FileExtensions", [".jpg"], "IncludeSubfolders", true, "LabelSource", "foldernames");

numFiles = numel(imds.Files);
% Generate random permutation indices for shuffling
perm = randperm(numel(imds.Files));

% Shuffle the indices and use them to access the files
shuffledFiles = imds.Files(perm);

% Calculate the number of files for each split
numTrain = round(0.6 * numFiles);
numValidation = round(0.1 * numFiles);

% Partition the shuffled files manually
imDsTrain = imageDatastore(shuffledFiles(1:numTrain), "LabelSource", "foldernames");
imDsValidation = imageDatastore(shuffledFiles(numTrain+1:numTrain+numValidation), "LabelSource", "foldernames");
imDsTest = imageDatastore(shuffledFiles(numTrain+numValidation+1:end), "LabelSource", "foldernames");

bbds = fileDatastore(filePath, "ReadFcn", @readXML, "IncludeSubfolders", true, "FileExtensions", [".xml"]);

bboxes = readall(bbds);
bbLabels = getLowestLevelFolders(bbds.Files);

shuffledValues = bboxes(perm);
shuffledLabels = bbLabels(perm)';

trainTable = table(shuffledValues(1:numTrain));

bbDsTrain = boxLabelDatastore(trainTable);
bbDsValidation = boxLabelDatastore(table(shuffledValues(numTrain+1:numTrain+numValidation)));
bbDsTest = boxLabelDatastore(table(shuffledValues(numTrain+numValidation+1:end)));

%dataSource = groundTruthDataSource(imds.Files);
%gTruth = groundTruth(dataSource, table(shuffledLabels), table(shuffledValues));

%% Saving off data
save('RegionData.mat');

%% Load Data
load("RegionData.mat")

%% Generate datastores

trainingDs = combine(imDsTrain, bbDsTrain);
validationDs = combine(imDsValidation, bbDsValidation);
testDs = combine(imDsTest, bbDsTest);

%% Validate training set
validateInputData(trainingDs);
validateInputData(validationDs);
validateInputData(testDs);

%% Display example image
data = read(trainingDs);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,"Rectangle",bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%% Prep network and Augment Data
%Input needs to bea multiple of 32, so 224 instead of 227
inputSize = [224 224 3];

rng("default")
trainingDataForEstimation = transform(trainingDs,@(data)preprocessData(data,inputSize));
numAnchors = 9;
[anchors,meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);
area = anchors(:, 1).*anchors(:,2);
[~,idx] = sort(area,"descend");

anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    anchors(7:9,:)
    };
detector = yolov4ObjectDetector("csp-darknet53-coco",unique(shuffledLabels),anchorBoxes,InputSize=inputSize);
augmentedTrainingData = transform(trainingDs,@augmentData);
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
    MiniBatchSize=64,...
    L2Regularization=0.0005,...
    MaxEpochs=1,...
    BatchNormalizationStatistics="moving",...
    DispatchInBackground=true,...
    ResetInputNormalization=false,...
    Shuffle="every-epoch",...
    VerboseFrequency=20,...
    ValidationFrequency=100,...
    CheckpointPath=tempdir,...
    ValidationData=validationDs);


%% Train the Network
[detector,info] = trainYOLOv4ObjectDetector(augmentedTrainingData,detector,options);

%% Creates helper functions from example online
function data = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.

data = cell(size(A));
for ii = 1:size(A,1)
    I = A{ii,1};
    bboxes = A{ii,2};
    labels = A{ii,3};
    sz = size(I);

    if numel(sz) == 3 && sz(3) == 3
        I = jitterColorHSV(I,...
            contrast=0.0,...
            Hue=0.1,...
            Saturation=0.2,...
            Brightness=0.2);
    end
    
    % Randomly flip image.
    tform = randomAffine2d(XReflection=true,Scale=[1 1.1]);
    rout = affineOutputView(sz,tform,BoundsStyle="centerOutput");
    I = imwarp(I,tform,OutputView=rout);
    
    % Apply same transform to boxes.
    [bboxes,indices] = bboxwarp(bboxes,tform,rout,OverlapThreshold=0.25);
    labels = labels(indices);
    
    % Return original data only when all boxes are removed by warping.
    if isempty(indices)
        data(ii,:) = A(ii,:);
    else
        data(ii,:) = {I,bboxes,labels};
    end
end
end

function data = preprocessData(data,targetSize)
% Resize the images and scale the pixels to between 0 and 1. Also scale the
% corresponding bounding boxes.

%for ii = 1:size(data,1)
%    I = data{ii,1};
%    imgSize = size(I);
%    
%    bboxes = data{ii,2};
%    I = im2single(imresize(I,targetSize(1:2)));
%    scale = targetSize(1:2)./imgSize(1:2);
%    bboxes = bboxresize(bboxes,scale);
    
%    data(ii,1:2) = {I,bboxes};
%end

% Resize image and bounding boxes to the targetSize.
scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
boxEstimate=round(data{2});
boxEstimate(:,1)=max(boxEstimate(:,1),1);
boxEstimate(:,2)=max(boxEstimate(:,2),1);
data{2} = bboxresize(boxEstimate,scale);
end

function detector = downloadPretrainedYOLOv4Detector()
% Download a pretrained yolov4 detector.
if ~exist("yolov4CSPDarknet53VehicleExample_22a.mat", "file")
    if ~exist("yolov4CSPDarknet53VehicleExample_22a.zip", "file")
        disp("Downloading pretrained detector...");
        pretrainedURL = "https://ssd.mathworks.com/supportfiles/vision/data/yolov4CSPDarknet53VehicleExample_22a.zip";
        websave("yolov4CSPDarknet53VehicleExample_22a.zip", pretrainedURL);
    end
    unzip("yolov4CSPDarknet53VehicleExample_22a.zip");
end
pretrained = load("yolov4CSPDarknet53VehicleExample_22a.mat");
detector = pretrained.detector;
end

function validateInputData(ds)
% Validates the input images, bounding boxes and labels and displays the 
% paths of invalid samples. 

% Copyright 2021 The MathWorks, Inc.

% Path to images
info = ds.UnderlyingDatastores{1}.Files;

ds = transform(ds, @isValidDetectorData);
data = readall(ds);

validImgs = [data.validImgs];
validBoxes = [data.validBoxes];
validLabels = [data.validLabels];

msg = "";

if(any(~validImgs))
    imPaths = info(~validImgs);
    str = strjoin(imPaths, '\n');
    imErrMsg = sprintf("Input images must be non-empty and have 2 or 3 dimensions. The following images are invalid:\n") + str;
    msg = (imErrMsg + newline + newline);
end

%if(any(~validBoxes))
    %imPaths = info(~validBoxes);
    %str = strjoin(imPaths, '\n');
    %boxErrMsg = sprintf("Bounding box data must be M-by-4 matrices of positive integer values. The following images have invalid bounding box data:\n") ...
        %+ str;
    
    %msg = (msg + boxErrMsg + newline + newline);
%end

if(any(~validLabels))
    imPaths = info(~validLabels);
    str = strjoin(imPaths, '\n');
    labelErrMsg = sprintf("Labels must be non-empty and categorical. The following images have invalid labels:\n") + str;
    
    msg = (msg + labelErrMsg + newline);
end

if(~isempty(msg))
    error(msg);
end

end

function out = isValidDetectorData(data)
% Checks validity of images, bounding boxes and labels
for i = 1:size(data,1)
    I = data{i,1};
    boxes = data{i,2};
    labels = data{i,3};

    imageSize = size(I);
    mSize = size(boxes, 1);

    out.validImgs(i) = iCheckImages(I);
    out.validBoxes(i) = iCheckBoxes(boxes, imageSize);
    out.validLabels(i) = iCheckLabels(labels, mSize);
end

end

function valid = iCheckImages(I)
% Validates the input images.

valid = true;
if ndims(I) == 2
    nDims = 2;
else
    nDims = 3;
end
% Define image validation parameters.
classes        = {'numeric'};
attrs          = {'nonempty', 'nonsparse', 'nonnan', 'finite', 'ndims', nDims};
try
    validateattributes(I, classes, attrs);
catch
    valid = false;
end
end

function valid = iCheckBoxes(boxes, imageSize)
% Validates the ground-truth bounding boxes to be non-empty and finite.

valid = true;
% Define bounding box validation parameters.
classes = {'numeric'};
attrs   = {'nonempty', 'integer', 'nonnan', 'finite', 'positive', 'nonzero', 'nonsparse', '2d', 'ncols', 4};
try
    validateattributes(boxes, classes, attrs);
    % Validate if bounding box in within image boundary.
    validateattributes(boxes(:,1)+boxes(:,3)-1, classes, {'<=', imageSize(2)});
    validateattributes(boxes(:,2)+boxes(:,4)-1, classes, {'<=', imageSize(1)}); 
catch
    valid = false;
end
end

function valid = iCheckLabels(labels, mSize)
% Validates the labels.

valid = true;
% Define label validation parameters.
classes = {'categorical'};
attrs   = {'nonempty', 'nonsparse', '2d', 'ncols', 1, 'nrows', mSize};
try
    validateattributes(labels, classes, attrs);
catch
    valid = false;
end
end

function box = getBBox(path)
    xmlStruct = readstruct(path);
        
    values = xmlStruct.object.bndbox;
    width = values.xmax - values.xmin;
    height = values.ymax - values.ymin;

    box = [values.xmin values.ymin width height]';
end

function data = readXML(filename) 

    xml = readstruct(filename).object.bndbox;
    %xcenter = (xml.xmax + xml.xmin)/2;
    %ycenter = (xml.ymax + xml.ymin)/2;
    width = xml.xmax - xml.xmin;
    height = xml.ymax - xml.ymin;

    data = [xml.xmin xml.ymin width height];

end


function lowestLevelFolders = getLowestLevelFolders(filePaths)
    numFiles = numel(filePaths);
    lowestLevelFolders = cell(1, numFiles);

    for i = 1:numFiles
        [path, ~, ~] = fileparts(filePaths{i});
        [~, folderName, ~] = fileparts(path)
        %folderParts = strsplit(folderName, filesep);
        lowestLevelFolders{i} = folderName;
    end
end





