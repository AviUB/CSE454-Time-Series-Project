
% 2) initialization and loading
%==========================================================================
% clears just in case any garbage data is leftover from a previous run
clc
clear

% load the data file into a numeric matrix
% synthetic_control contains 600 samples each with 60 times points, 
% divided into 6 classes
data = load("synthetic_control.data");
s = size(data);
tps = s(2);         % number of observations or time points
d_size = s(1);      % number of samples in the entire dataset
c_num = 6;          % number of classes in the dataset
c_size = d_size/c_num;  % number of samples in each class
%==========================================================================

% 3) preprocessing step
%==========================================================================
% normalization (scale values to be 0 to 1)
for i = 1:tps
    % data entry values range 0 to 100, divide by 100 to normalize
    data(:,i) = data(:,i) / 100;
end
 
% pull classes from the dataset 
% according to the UCI repo, the classes are organized as follows:
% 1-100 Normal
% 101-200 Cyclic
% 201-300 Increasing trend
% 301-400 Decreasing trend
% 401-500 Upward shift
% 501-600 Downward shift 
class_normal = data(1:c_size, :);
class_cyclic = data(c_size+1:2*c_size, :);
class_inc_trend = data(2*c_size+1:3*c_size, :);
class_dec_trend = data(3*c_size+1:4*c_size, :);
class_up_shift = data(4*c_size+1:5*c_size, :);
class_dn_shift = data(5*c_size+1:6*c_size,:);

% can also make a vector to represent each sample's class with 
% a corresponding number label for later use
data_clabels = zeros(1,d_size);
j=1;
for i=1:d_size
    data_clabels(1,i) = j;
    if mod(i,100)==0
        j=j+1;
    end
end
%==========================================================================

% 4a) sampling step 
%==========================================================================
% each class has 100 samples, a lot of data to look at all at once
% can make a smaller training dataset from these samples

% can take the mean of all samples for each class and use it as 
% a single sample of each class for training (600 samples to 6)
mean_cn_sample = get_mean_sample(class_normal, tps, c_size);    
mean_cc_sample = get_mean_sample(class_cyclic, tps, c_size);
mean_cit_sample = get_mean_sample(class_inc_trend, tps, c_size);   
mean_cdt_sample = get_mean_sample(class_dec_trend, tps, c_size);
mean_cu_sample = get_mean_sample(class_up_shift, tps, c_size);    
mean_cd_sample = get_mean_sample(class_dn_shift, tps, c_size);

% put them all into a training dataset 
data_t = zeros(c_num, tps);
data_t(1,:)=mean_cn_sample; data_t(2,:)=mean_cc_sample;
data_t(3,:)=mean_cit_sample; data_t(4,:)=mean_cdt_sample;
data_t(5,:)=mean_cu_sample; data_t(6,:)=mean_cd_sample;

t_size = size(data_t, 1); 
%==========================================================================

% 4b) representation step
%==========================================================================
% first is PAA (piecewise aggregate approximation)
c=5;    % number of windows for paa
% do paa on entire dataset
paa_d = get_paa(data, d_size, tps, c);
figure('Name', 'Dataset PAAs');
for i=1:d_size
    paaplot(paa_d, data , d_size, tps,  c, i);
end
% do paa on testing dataset
paa_t = get_paa(data_t, t_size, tps, c);
figure('Name', 'Testing PAAs');
for i=1:t_size
    paaplot(paa_t, data_t , t_size, tps,  c, i);
end
% plot paa for each testing sample individually
for i=1:t_size
    figure('Name', 'Testing PAA')
    paaplot(paa_t, data_t , t_size, tps,  c, i);
end
% do paa on each class subset
paa_n = get_paa(class_normal, c_size, tps, c);
figure('Name', 'Normal PAA');
for i=1:c_size
    paaplot(paa_n, class_normal , c_size, tps,  c, i);
end
paa_c = get_paa(class_cyclic, c_size, tps, c);
figure('Name', 'Cyclic PAA');
for i=1:c_size
    paaplot(paa_c, class_cyclic, c_size, tps,  c, i);
end
paa_it = get_paa(class_inc_trend, c_size, tps, c);
figure('Name', 'Increasing Trend PAA');
for i=1:c_size
    paaplot(paa_it, class_inc_trend, c_size, tps,  c, i);
end
paa_dt = get_paa(class_dec_trend, c_size, tps, c);
figure('Name', 'Decreasing Trend PAA');
for i=1:c_size
    paaplot(paa_dt, class_dec_trend, c_size, tps,  c, i);
end
paa_u = get_paa(class_up_shift, c_size, tps, c);
figure('Name', 'Upward Shift PAA');
for i=1:c_size
    paaplot(paa_u, class_up_shift, c_size, tps,  c, i);
end
paa_d = get_paa(class_dn_shift, c_size, tps, c);
figure('Name', 'Downward Shift PAA');
for i=1:c_size
    paaplot(paa_d, class_dn_shift, c_size, tps,  c, i);
end

% next is SAX (symbolic aggregate approximation)

%==========================================================================

% 5) analysis
%==========================================================================
% plot the data
% make an evenly spaced vector for plotting, 60 time points 
t = linspace(1, tps, tps);
% there are plots for the original data set, the class sets,
% the training sample sets,
% comment out/in whichever plots need to be analyzed
figure('Name', 'Synthetic Control Dataset');    plot(t, data);
figure('Name', 'Normal');                       plot(t, class_normal);
figure('Name', 'Cyclic');                       plot(t, class_cyclic);
figure('Name', 'Increasing Trend');             plot(t, class_inc_trend);
figure('Name', 'Decreasing Trend');             plot(t, class_dec_trend);
figure('Name', 'Upward Shift');                 plot(t, class_up_shift);
figure('Name', 'Downward Shift');               plot(t, class_dn_shift);
figure('Name', 'Normal Testing');               plot(t, mean_cn_sample);
figure('Name', 'Cyclic Testing');               plot(t, mean_cc_sample);
figure('Name', 'Increasing Trend Testing');     plot(t, mean_cit_sample);
figure('Name', 'Decreasing Trend Testing');     plot(t, mean_cdt_sample);
figure('Name', 'Upward Shift Testing');         plot(t, mean_cu_sample);
figure('Name', 'Downward Shift Testing');       plot(t, mean_cd_sample);

% calculate distances to determine classifications
% distances will be calculated with euclidean and manhatten methods
% go through entire dataset
for i=1:d_size
    % go through training dataset
    for j=1:t_size
        euc(i,j)=euclidist(data(i,:), data_t(j,:));
        man(i,j)=manhadist(data(i,:), data_t(j,:));
    end
end
% determine classifications
% euc and man should be same size
for i=1:size(euc, 1)
    % use the minimum distance to make class predictions
    [throwAway,euc_classifications(i)]=min(euc(i,:));
    [throwAway,man_classifications(i)]=min(man(i,:));
end
    
% check how good the classification predictions are
cnt_e=0; cnt_m=0;
for i=1:length(euc_classifications)
    % check if classification matches what it should be
    d=euc_classifications(i) - data_clabels(i);
    if d==0
        cnt_e=cnt_e+1;
    end
end
for i=1:length(man_classifications)
    % check if classification matches what it should be
    d=man_classifications(i) - data_clabels(i);
    if d==0
        cnt_m=cnt_m+1;
    end
end
% confusion matrices for visualizing class predictions
% required installing "deep learning toolbox" add-on
acc_e = cnt_e/length(euc_classifications)
c_e = confusionmat(data_clabels, euc_classifications);
figure('Name', 'Euclidean'); confusionchart(c_e);
acc_m = cnt_m/length(man_classifications)
c_m = confusionmat(data_clabels, man_classifications);
figure('Name', 'Manhattan'); confusionchart(c_m);
%==========================================================================
% helper functions

% find the euclidian distance between two time series
% sqrt(sum((s[i]-r[i])^2))
% input: 2 time series, s and r
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
% input: 2 time series, s and r
% output: distance between them
function d = manhadist(s, r)
% for every observation in each series (r and s same length)
    d=0;
    for i=1:length(s)
        d1=s(i)-r(i);
        d=d+abs(d1);
    end
end

% get a mean time series sample from a dataset
% inputs: dataset, number of timepoints, number of samples
% output: mean sample
function ms = get_mean_sample(d, nt, ns)
    ms = zeros(1, nt);
    for i = 1:nt
    % add together every sample at each time point
        for j = 1:ns
            ms(1,i) = ms(1,i) + d(j,i);
        end
        % divide total by number of time points
        ms(1,i) = ms(1,i) / nt;
    end
end