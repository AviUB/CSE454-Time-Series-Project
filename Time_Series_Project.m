
% 1) figure out the problem
%==========================================================================
% covered within report
%==========================================================================

% 2) initialization and loading
%==========================================================================
% clears just in case any garbage data is leftover from a previous run
clc
clear
close all

% load the data file into a numeric matrix
% synthetic_control contains 600 samples each with 60 time points, 
% divided into 6 classes
data = load("synthetic_control.data");
tps = size(data, 2);    % number of observations or time points
d_size = size(data, 1); % number of samples in the entire dataset
c_num = 6;              % number of classes in the dataset
c_size = d_size/c_num;  % number of samples in each class
%==========================================================================

% 3) preprocessing step
%==========================================================================
% standardization (transform dataset to get mean=0, std=1)
data = normalize(data);

% pull classes from the dataset, store them together in an array
% according to the UCI repo, the classes are organized as follows:
% 1-100 Normal
% 101-200 Cyclic
% 201-300 Increasing trend
% 301-400 Decreasing trend
% 401-500 Upward shift
% 501-600 Downward shift 
class_n = data(1:c_size, :);
class_c = data(c_size+1:2*c_size, :);
class_it = data(2*c_size+1:3*c_size, :);
class_dt = data(3*c_size+1:4*c_size, :);
class_us = data(4*c_size+1:5*c_size, :);
class_ds = data(5*c_size+1:6*c_size, :);
% can put them all into a cell array for easy access later
classes = {class_n, class_c, class_it, class_dt, class_us, class_ds};

% can also make a vector to represent each sample's class with 
% a corresponding number label for later use in classification analysis
data_clabels = zeros(d_size,1);
j=1;
for i=1:d_size
    data_clabels(i) = j;
    if mod(i,100)==0
        j=j+1;
    end
end
%==========================================================================

% 4a) sampling step (establising training and testing datasets)
%==========================================================================
% the testing dataset is the original dataset
% each class has 100 samples, a lot of data to look at all at once
% can make a smaller training dataset from these samples
% using 10 samples from each set
% put them all into a training dataset using a helper function
sam_idxs = [1,10,25,50,75,90,100];
data_t = get_training(classes, sam_idxs, tps);
t_size = size(data_t, 1); 
%==========================================================================

% 4b) representation step
%==========================================================================
% PAA (piecewise aggregate approximation)
c=10;    % number of windows for paa

% do paa on entire dataset (testing data)
paa_dataset = get_paa(data, d_size, tps, c);
% do paa on training dataset
paa_training = get_paa(data_t, t_size, tps, c);

% SAX (symbolic aggregate approximation)
% not implemented
%==========================================================================

% 5) analysis
%==========================================================================
% classifiction
%--------------------------------------------------------------------------
% calculate distances to determine classifications
% distances will be calculated with euclidean and manhatten methods
% go through entire dataset
euc=zeros(d_size,t_size);       man=zeros(d_size,t_size);
euc_PAA=zeros(d_size,t_size);   man_PAA=zeros(d_size,t_size);
for i=1:d_size
    % go through training dataset
    for j=1:t_size
        % make sure each datapoint gets the proper label
        c_label=mod(j, c_num);
        if c_label==0
            c_label=6;
        end
        euc(i,j)=euclidist(data(i,:), data_t(j,:));
        man(i,j)=manhadist(data(i,:), data_t(j,:));
        euc_PAA(i,j)=euclidist(paa_dataset(i,:), paa_training(j,:));
        man_PAA(i,j)=manhadist(paa_dataset(i,:), paa_training(j,:));
    end
end

% determine classifications with distances and a helper function
euc_classes = get_classifications(d_size, euc, data_clabels);
man_classes = get_classifications(d_size, man, data_clabels);
paa_euc_classes = get_classifications(d_size, euc_PAA, data_clabels);
paa_man_classes = get_classifications(d_size, man_PAA, data_clabels);

% confusion matrices for visualizing class predictions
% required installing "deep learning toolbox" add-on
% c_e = confusionmat(data_clabels, euc_classes);
% figure('Name', 'Euclidean'); confusionchart(c_e);
% c_m = confusionmat(data_clabels, man_classes);
% figure('Name', 'Manhattan'); confusionchart(c_m);
% c_e_paa = confusionmat(data_clabels, paa_euc_classes);
% figure('Name', 'Euclidean PAA'); confusionchart(c_e_paa);
% c_m_paa = confusionmat(data_clabels, paa_man_classes);
% figure('Name', 'Manhattan PAA'); confusionchart(c_m_paa);
%--------------------------------------------------------------------------

% plotting the data
%--------------------------------------------------------------------------
% WARNING: uncommenting all these plots will result in 94 figures printed
% comment out/in whichever plots need to be analyzed
% make an evenly spaced vector for plotting, 60 time points / observations 
t = linspace(1, tps, tps);

% % original data set and class subsets
% figure('Name', 'Synthetic Control Dataset');    plot(t, data);
% figure('Name', 'Normal');                       plot(t, class_n);
% figure('Name', 'Cyclic');                       plot(t, class_c);
% figure('Name', 'Increasing Trend');             plot(t, class_it);
% figure('Name', 'Decreasing Trend');             plot(t, class_dt);
% figure('Name', 'Upward Shift');                 plot(t, class_us);
% figure('Name', 'Downward Shift');               plot(t, class_ds);
% 
% % training set and individual training time series samples
% figure('Name', 'Training Set');                  plot(t, data_t);
% % 42 total samples in training set
% for i=1:t_size
%     figure('Name', 'Training Sample');           plot(t, data_t(i,:));
% end
% 
% % PAA plots for original (testing) data set (ugly mess)
% figure('Name', 'Dataset PAA');
% for i=1:c_size
%     paaplot(paa_dataset, data, c_size, tps,  c, i);
% end
% % PAA plots for entire training dataset and each training time series
% figure('Name', 'Training PAAs');
% for i=1:t_size
%     paaplot(paa_training, data_t, t_size, tps,  c, i);
% end
% for i=1:t_size
%     figure('Name', 'Training PAAs Time Series');
%     paaplot(paa_training, data_t, t_size, tps,  c, i);
% end
%--------------------------------------------------------------------------

%==========================================================================
% helper functions

% find the euclidian distance between two time series
% sqrt(sum((s[i]-r[i])^2))
% inputs: 2 time series, s and r
% output: distance between them
function d = euclidist(s, r)
    % for every observation in each series (r and s same length)
    sumD=0;
    for i = 1:length(s)
        d1=s(i)-r(i);
        sumD=sumD+(d1^2);
    end
    d=sqrt(sumD);
end

% find the manhattan distance between two time series
% d=sum(abs(s[i]-r[i]))
% inputs: 2 time series, s and r
% output: distance between them
function d = manhadist(s, r)
% for every observation in each series (r and s same length)
    d=0;
    for i=1:length(s)
        d1=s(i)-r(i);
        d=d+abs(d1);
    end
end

% get the training dataset from the original dataset (testing dataset)
% pull out a sample of each class from certain indices and put them
% into the training set in a class specified order
% inputs: classes cell array, array of sample indices, 
%         number of observations in the time series
% output: training dataset as a numerical matrix
function data_train = get_training(classes, sample_idxs, tps)
    % preallocate training datatset
    data_train=zeros(length(classes), tps);
    % index to place sample in the training dataset
    i=1;
    % once for each value in the sample index array
    for j=1:length(sample_idxs)
        % for each class
        for k=1:length(classes)
            % pull class subset from the classes cell array
            class=classes(k); class=class{1};
            % put sample into training dataset from its class
            data_train(i,:)=class(sample_idxs(j),:);
            i=i+1;
        end
    end
end

% get the classifications of a dataset based on distance calculations
% inputs: size of dataset, numerical matrix of distances for each sample, 
%         column vector of class labels
% outputs: an array of classification assignments for each sample
function classifications = get_classifications(set_size, dist, clabels)
    % preallocate classifications
    classifications=zeros(set_size,1);
    % determine classifications
    for i=1:length(classifications)
        % use the minimum distance to make class predictions
        [throwaway,classifications(i)]=min(dist(i,:));
    end
    % make sure class labels are correct
    for i=1:length(classifications)
        % change labels from 1-t_size to 1-6 so no extra classes are interpreted
        if classifications(i) > 6
            classifications(i)=mod(classifications(i),6);
            if classifications(i)==0
                classifications(i)=6;
            end
        end
    end
    % check how good the classification predictions are
    cnt=0;
    for i=1:length(classifications)
        % check if classification matches what it should be
        d=classifications(i) - clabels(i);
        if d==0
            cnt=cnt+1;
        end
    end
    % print classification accuracy into the console
    acc = cnt/length(classifications)
end
