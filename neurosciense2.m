clc
clear all
close all

%% loading data

addpath('C:\Users\pouya\Desktop\neuros\project\data 3\data neurons');


load 2015090508.mat


%% visualization
%   

n = 11;  %number of trial



tsaccade = double(tsaccade);
onset = tsaccade(n,:);
onsett = num2str(onset); % to show on plot
stem(resp(n,:));   % ploting spikes on times
title(['trial ' , num2str(n) , ' / saccade at ' , onsett , '  ms']);  
set(gca , 'xlim' , [0 , 3000] , 'ylim' , [0 , 2]);
%
%% spkie count rate in trial n

T = onset;
ns_before_sac = sum(resp(n,1:onset));
spike_count_rate_before_saccade_for_trial_n = ns_before_sac/T*1000 %Hz

T2 = 3000-onset;

ns_after_sac = sum(resp(n,onset:3000));
spike_count_rate_after_saccade_for_trial_n = ns_after_sac/T2*1000


%
%% average firing rate
%
for trials = 1:length(resp(:,1)); %we average it on whole trials

    newonset = tsaccade(trials,:);
    number_of_trials = length(resp(:,1));
    summation_of_before(trials) = sum(resp(trials,1:newonset));
    summation_of_after(trials) = sum(resp(trials,newonset:3000));
    
    
end

summation_of_before_saccade = mean(summation_of_before);          %mean of summation of all spikes
summation_of_after_saccade = mean(summation_of_after);
average_fr_before_saccade = summation_of_before_saccade/number_of_trials*1000    % making <n>/T
average_fr_after_saccade = summation_of_after_saccade/number_of_trials*1000

%

%% making spike train smooth
%
n_spike_train = resp(n,:);
smoothing_bandwidth = 101 ;
smoothed = smooth(n_spike_train, smoothing_bandwidth);   %smoothing using bandwidth

time_bin = smoothing_bandwidth;

FR = smoothed/time_bin;                    %smooth firing rate
time = 1:1:length(resp(1,:));
figure
plot(time,FR);
title('firing rate');
xlabel('time(ms)');
ylabel('firing rate(Hz)');

%% plotig stimulus
plot(stimcode(1,:));
title('stimulus');
xlabel('time(ms)');
ylabel('location');

%% seperating after and before saccade stims

stim_before = zeros(length(conds),3000); % because the number of matrix columns are different and I we want them in a matrix
stim_after = zeros(length(conds),3000);
for ii = 1:length(conds);
    
    
    
    stim_before(ii,1:tsaccade(ii)) = stimcode(ii,1:tsaccade(ii));
    stim_after(ii,tsaccade(ii):3000) = stimcode(ii,tsaccade(ii):3000);
    
    
    
end


%%  tuning curve building without sepreation of saccade time
%{
    for iter = 1:81;
    
        
        
    [i,j] = find(stimcode == iter);
    A = resp(i,j);
    mean_sp(iter) = sum(A, 'all')/length(A);
    
    clear i    % it will be replaced by new stimulus(iteration)
    clear j
    clear A
    
    end

%% ploting tuning curve
x = 1:1:81 ;
x = x.';
%tuning_curve =  fit(x,mean_sp, 'smoothingspline');
tuning_curve = smoothdata(mean_sp,'gaussian',10);
figure
plot(tuning_curve);
title('tuning curve');
    
    
  %}
%% tuning curve building after saccade time sepreation

   for iter = 1:81;
    
        
    [l,k] = find(stim_after == iter);    
    [i,j] = find(stim_before == iter);
    A = resp(i,j);
    B = resp(l,k);
    mean_sp_before(iter) = sum(A, 'all')/length(A);
    mean_sp_after(iter) = sum(B, 'all')/length(B);
    clear i
    clear j
    clear A
    clear k
    clear l
    clear B
    end

%% ploting tuning curve after saccade time speration
%x = 1:1:81 ;
%x = x.';
%tuning_curve =  fit(x,mean_sp, 'smoothingspline');
tuning_curve1 = smoothdata(mean_sp_before,'gaussian',10);
tuning_curve2 = smoothdata(mean_sp_after,'gaussian',10);
figure
plot(tuning_curve1);
title('tuning curve for before saccade stimuluses');
figure
plot(tuning_curve2);
title('tuning curve for after saccade stimuluses');

%% seperating response before and after saccade
%
resp_before = zeros(length(conds),3000);
resp_after = zeros(length(conds),3000);

for sep = 1:length(conds);

resp_before(sep,1:tsaccade(sep)) = resp(sep,1:tsaccade(sep));
resp_after(sep,tsaccade(sep):3000) = resp(sep,tsaccade(sep):3000);

end
%}

%%  data sorting

training_stim_before = stim_before(1:3/4*size(stim_before,1),:);
test_stim_before = stim_before(floor(3/4*size(stim_before,1))+1:end,:);

training_resp_before = resp_before(1:3/4*size(resp_before,1),:);
test_resp_before = resp_before(floor(3/4*size(resp_before,1))+1:end,:);

training_stim_after = stim_after(1:3/4*size(stim_after,1),:);
test_stim_after = stim_after(floor(3/4*size(stim_after,1))+1:end,:);

training_resp_after = resp_after(1:3/4*size(resp_after,1),:);
test_resp_after = resp_after(floor(3/4*size(resp_after,1))+1:end,:);

%% feature extraction of training data


%for stims before saccade

for tt = 1:length(training_stim_before(:,1));

    
    from = 1;
    to = 100;
    
    for v = 1:30 ;
    
    if to <= length(training_stim_before(1,:)) ;  % from 0 to 3000 ms
        
        training_stim_before_section = training_stim_before(tt,from:to);
        
        training_feature_stim_before_mean(tt,v) = mean(training_stim_before_section);
        training_feature_stim_before_std(tt,v) = std(training_stim_before_section);
        training_feature_stim_before_var(tt,v) = var(training_stim_before_section);
        training_feature_stim_before_rms(tt,v) = rms(training_stim_before_section);
        training_feature_stim_before_thd(tt,v) = thd(training_stim_before_section);
        training_feature_stim_before_snr(tt,v) = snr(training_stim_before_section);
        training_feature_stim_before_sinad(tt,v) = sinad(training_stim_before_section);
        training_feature_stim_before_mf(tt,v) = meanfreq(training_stim_before_section);
        training_feature_stim_before_md(tt,v) = medfreq(training_stim_before_section);
        training_feature_stim_before_bp(tt,v) = bandpower(training_stim_before_section);
        
        from = from+100;
        to = to+100;
        
    end
    
    
    end
    
    clear from
    clear to
    

end


% for stims after saccade

for tt1 = 1:length(training_stim_after(:,1));

    
    from1 = 1;
    to1 = 100;
    
    for v1 = 1:30 ;
    
    if to1 <= length(training_stim_after(1,:)) ; 
        
        training_stim_after_section = training_stim_after(tt1,from1:to1);
        
        training_feature_stim_after_mean(tt1,v1) = mean(training_stim_after_section);
        training_feature_stim_after_std(tt1,v1) = std(training_stim_after_section);
        training_feature_stim_after_var(tt1,v1) = var(training_stim_after_section);
        training_feature_stim_after_rms(tt1,v1) = rms(training_stim_after_section);
        training_feature_stim_after_thd(tt1,v1) = thd(training_stim_after_section);
        training_feature_stim_after_snr(tt1,v1) = snr(training_stim_after_section);
        training_feature_stim_after_sinad(tt1,v1) = sinad(training_stim_after_section);
        training_feature_stim_after_mf(tt1,v1) = meanfreq(training_stim_after_section);
        training_feature_stim_after_md(tt1,v1) = medfreq(training_stim_after_section);
        training_feature_stim_after_bp(tt1,v1) = bandpower(training_stim_after_section);
        
        
        from1 = from1+100;
        to1 = to1+100;
        
    end
    
    
    end
    
    clear from1
    clear to1
    

end
    
%% feature extraction for test data

%for stims before saccade

for tt4 = 1:length(test_stim_before(:,1));

    
    from4 = 1;
    to4 = 100;
    
    for v4 = 1:30 ;
    
    if to4 <= length(test_stim_before(1,:)) ; 
        
        test_stim_before_section = test_stim_before(tt4,from4:to4);
        
        test_feature_stim_before_mean(tt4,v4) = mean(test_stim_before_section);
        test_feature_stim_before_std(tt4,v4) = std(test_stim_before_section);
        test_feature_stim_before_var(tt4,v4) = var(test_stim_before_section);
        test_feature_stim_before_rms(tt4,v4) = rms(test_stim_before_section);
        test_feature_stim_before_thd(tt4,v4) = thd(test_stim_before_section);
        test_feature_stim_before_snr(tt4,v4) = snr(test_stim_before_section);
        test_feature_stim_before_sinad(tt4,v4) = sinad(test_stim_before_section);
        test_feature_stim_before_mf(tt4,v4) = meanfreq(test_stim_before_section);
        test_feature_stim_before_md(tt4,v4) = medfreq(test_stim_before_section);
        test_feature_stim_before_bp(tt4,v4) = bandpower(test_stim_before_section);
        
        
        
        
        from4 = from4+100;
        to4 = to4+100;
        
    end
    
    
    end
    
    clear from4
    clear to4
    

end


% for stims after saccade

for tt5 = 1:length(test_stim_after(:,1));

    
    from5 = 1;
    to5 = 100;
    
    for v5 = 1:30 ;
    
    if to5 <= length(test_stim_after(1,:)) ; 
        
        test_stim_after_section = test_stim_after(tt5,from5:to5);
        
        test_feature_stim_after_mean(tt5,v5) = mean(test_stim_after_section);
        test_feature_stim_after_std(tt5,v5) = std(test_stim_after_section);
        test_feature_stim_after_var(tt5,v5) = var(test_stim_after_section);
        test_feature_stim_after_rms(tt5,v5) = rms(test_stim_after_section);
        test_feature_stim_after_thd(tt5,v5) = thd(test_stim_after_section);
        test_feature_stim_after_snr(tt5,v5) = snr(test_stim_after_section);
        test_feature_stim_after_sinad(tt5,v5) = sinad(test_stim_after_section);
        test_feature_stim_after_mf(tt5,v5) = meanfreq(test_stim_after_section);
        test_feature_stim_after_md(tt5,v5) = medfreq(test_stim_after_section);
        test_feature_stim_after_bp(tt5,v5) = bandpower(test_stim_after_section);
        
        
        from5 = from5+100;
        to5 = to5+100;
        
    end
    
    
    end
    
    clear from5
    clear to5
    

end
    

%% turning training response to mean segments

% for responses before saccade

for mm6 = 1:length(training_resp_before(:,1));

    from6 = 1;
    to6 = 100;    
    
for ll6 = 1:30;
    
    
    

 if to6 <= length(training_resp_before(1,:)) ;
        
        training_resp_before_section = training_resp_before(mm6,from6:to6);  % choosing the segment
        
        training_resp_before_segmented(mm6 , ll6) = mean(training_resp_before_section); % replacing mean of segment in set
       


        
        from6 = from6+100;
        to6 = to6+100;
        
 end
end

end

% for responses after saccade

for mm7 = 1:length(training_resp_after(:,1));

    from7 = 1;
    to7 = 100;    
    
for ll7 = 1:30;
    
    
    

 if to7 <= length(training_resp_after(1,:)) ; 
        
        training_resp_after_section = training_resp_after(mm7,from7:to7);  
        
        training_resp_after_segmented(mm7 , ll7) = mean(training_resp_after_section);
       


        
        from7 = from7+100;
        to7 = to7+100;
        
 end
end

end

%% turning test responses to segments

% for responses before saccade

for mm8 = 1:length(test_resp_before(:,1));

    from8 = 1;
    to8 = 100;    
    
for ll8 = 1:30;
    
    
    

 if to8 <= length(test_resp_before(1,:)) ; 
        
        test_resp_before_section = test_resp_before(mm8,from8:to8);
        
        test_resp_before_segmented(mm8 , ll8) = mean(test_resp_before_section);
       


        
        from8 = from8+100;
        to8 = to8+100;
        
 end
end

end

% for responses after saccade

for mm9 = 1:length(test_resp_after(:,1));

    from9 = 1;
    to9 = 100;    
    
for ll9 = 1:30;
    
    
    

 if to9 <= length(test_resp_after(1,:)) ; 
        
        test_resp_after_section = test_resp_after(mm9,from9:to9);
        
        test_resp_after_segmented(mm9 , ll9) = mean(test_resp_after_section);
       


        
        from9 = from9+100;
        to9 = to9+100;
        
 end
end

end


%% segmenting stims
%{
% for stimuluses bafore saccade 

for mm10 = 1:length(training_stim_before(:,1));

    from10 = 1;
    to10 = 100;    
    
for llt = 1:30;
    
    
    

 if to10 <= length(training_stim_before(1,:)) ; 
        
        training_stim_before_section = training_stim_before(mm10,from10:to10);
        
        training_stim_before_segmented(mm10 , llt) = mean(training_stim_before_section);
       


        
        from10 = from10+100;
        to10 = to10+100;
        
 end
end

end

for mm11 = 1:length(test_stim_before(:,1));

    from11 = 1;
    to11 = 100;    
    
for llt2 = 1:30;
    
    
    

 if to11 <= length(test_stim_before(1,:)) ; 
        
        test_stim_before_section = test_stim_before(mm11,from11:to11);
        
        test_stim_before_segmented(mm11 , llt2) = mean(test_stim_before_section);
       


        
        from11 = from11+100;
        to11 = to11+100;
        
 end
end

end
%}


%% making data ready for classsification

clear U
clear G
clear L

for L = 1:length(training_feature_stim_after_mean(:,1));

    Z1(1,L*30-29:L*30) = training_feature_stim_after_mean(L,:);
    Z2(1,L*30-29:L*30) = training_feature_stim_after_std(L,:);
    Z3(1,L*30-29:L*30) = training_feature_stim_after_var(L,:);
    Z4(1,L*30-29:L*30) = training_feature_stim_after_rms(L,:);
    Z5(1,L*30-29:L*30) = training_feature_stim_after_thd(L,:);
    Z6(1,L*30-29:L*30) = training_feature_stim_after_snr(L,:);
    Z7(1,L*30-29:L*30) = training_feature_stim_after_sinad(L,:);
    Z8(1,L*30-29:L*30) = training_feature_stim_after_mf(L,:);
    Z9(1,L*30-29:L*30) = training_feature_stim_after_md(L,:);
    Z10(1,L*30-29:L*30) = training_feature_stim_after_bp(L,:);
    
    Ya1(1,L*30-29:L*30) = training_resp_after_segmented(L,:);
    
end



for U = 1:length(training_feature_stim_before_mean(:,1));

Q1(1,U*30-29:U*30) = training_feature_stim_before_mean(U,:);
Q2(1,U*30-29:U*30) = training_feature_stim_before_std(U,:);
Q3(1,U*30-29:U*30) = training_feature_stim_before_var(U,:);
Q4(1,U*30-29:U*30) = training_feature_stim_before_rms(U,:);
Q5(1,U*30-29:U*30) = training_feature_stim_before_thd(U,:);
Q6(1,U*30-29:U*30) = training_feature_stim_before_snr(U,:);
Q7(1,U*30-29:U*30) = training_feature_stim_before_sinad(U,:);
Q8(1,U*30-29:U*30) = training_feature_stim_before_mf(U,:);
Q9(1,U*30-29:U*30) = training_feature_stim_before_md(U,:);
Q10(1,U*30-29:U*30) = training_feature_stim_before_bp(U,:);


Y1(1,U*30-29:U*30) = training_resp_before_segmented(U,:);


end

%train

Q1 = Q1.' ;
Q2 = Q2.' ;
Q3 = Q3.' ;
Q4 = Q4.' ;
Q5 = Q5.' ;
Q6 = Q6.' ;
Q7 = Q7.' ;
Q8 = Q8.' ;
Q9 = Q9.' ;
Q10 = Q10.' ;

Z1 = Z1.' ;
Z2 = Z2.' ;
Z3 = Z3.' ;
Z4 = Z4.' ;
Z5 = Z5.' ;
Z6 = Z6.' ;
Z7 = Z7.' ;
Z8 = Z8.' ;
Z9 = Z9.' ;
Z10 = Z10.';
xtrain(:,1) = Q1;
xtrain(:,2) = Q2;
xtrain(:,3) = Q3;
xtrain(:,4) = Q4;
xtrain(:,5) = Q5;
xtrain(:,6) = Q6;
xtrain(:,7) = Q7;
xtrain(:,8) = Q8;
xtrain(:,9) = Q9;
xtrain(:,10) = Q10;

xatrain(:,1) = Z1;
xatrain(:,2) = Z2;
xatrain(:,3) = Z3;
xatrain(:,4) = Z4;
xatrain(:,5) = Z5;
xatrain(:,6) = Z6;
xatrain(:,7) = Z7;
xatrain(:,8) = Z8;
xatrain(:,9) = Z9;
xatrain(:,10) = Z10;

Ya1 = Ya1.';
Y1 = Y1.';
ytrain = Y1;
yatrain = Ya1;
% test

for X = 1:length(test_feature_stim_after_mean(:,1));
Z11(1,X*30-29:X*30) = test_feature_stim_after_mean(X,:);
Z22(1,X*30-29:X*30) = test_feature_stim_after_std(X,:);
Z33(1,X*30-29:X*30) = test_feature_stim_after_var(X,:);
Z44(1,X*30-29:X*30) = test_feature_stim_after_rms(X,:);
Z55(1,X*30-29:X*30) = test_feature_stim_after_thd(X,:);
Z66(1,X*30-29:X*30) = test_feature_stim_after_snr(X,:);
Z77(1,X*30-29:X*30) = test_feature_stim_after_sinad(X,:);
Z88(1,X*30-29:X*30) = test_feature_stim_after_mf(X,:);
Z99(1,X*30-29:X*30) = test_feature_stim_after_md(X,:);
Z1010(1,X*30-29:X*30) = test_feature_stim_after_bp(X,:);

Ya11(1,X*30-29:X*30) = test_resp_after_segmented(X,:);

end
%
for G = 1:length(test_feature_stim_before_mean(:,1));
Q11(1,G*30-29:G*30) = test_feature_stim_before_mean(G,:);
Q22(1,G*30-29:G*30) = test_feature_stim_before_std(G,:);
Q33(1,G*30-29:G*30) = test_feature_stim_before_var(G,:);
Q44(1,G*30-29:G*30) = test_feature_stim_before_rms(G,:);
Q55(1,G*30-29:G*30) = test_feature_stim_before_thd(G,:);
Q66(1,G*30-29:G*30) = test_feature_stim_before_snr(G,:);
Q77(1,G*30-29:G*30) = test_feature_stim_before_sinad(G,:);
Q88(1,G*30-29:G*30) = test_feature_stim_before_mf(G,:);
Q99(1,G*30-29:G*30) = test_feature_stim_before_md(G,:);
Q1010(1,G*30-29:G*30) = test_feature_stim_before_bp(G,:);


Y11(1,G*30-29:G*30) = test_resp_before_segmented(G,:);
end
%}

Q11 = Q11.' ;
Q22 = Q22.' ;
Q33 = Q33.' ;
Q44 = Q44.' ;
Q55 = Q55.' ;
Q66 = Q66.' ;
Q77 = Q77.' ;
Q88 = Q88.' ;
Q99 = Q99.';
Q1010 = Q1010.';
Z11 = Z11.' ;
Z22 = Z22.' ;
Z33 = Z33.' ;
Z44 = Z44.' ;
Z55 = Z55.' ;
Z66 = Z66.' ;
Z77 = Z77.' ;
Z88 = Z88.';
Z99 = Z99.';
Z1010 = Z1010.';

xtest(:,1) = Q11;
xtest(:,2) = Q22;
xtest(:,3) = Q33;
xtest(:,4) = Q44;
xtest(:,5) = Q55;
xtest(:,6) = Q66;
xtest(:,7) = Q77;
xtest(:,8) = Q88;
xtest(:,9) = Q99;
xtest(:,10) = Q1010;
xatest(:,1) = Z11;
xatest(:,2) = Z22;
xatest(:,3) = Z33;
xatest(:,4) = Z44;
xatest(:,5) = Z55;
xatest(:,6) = Z66;
xatest(:,7) = Z77;
xatest(:,8) = Z88;
xatest(:,9) = Z99;
xatest(:,10) = Z1010;
Y11 = Y11.';
Ya11 = Ya11.';
ytest = Y11;
yatest = Ya11;
%%
%% replace Nan and inf with median

nansetTrain = isnan(xtrain);
nansetTraina = isnan(xatrain);
infsetTrain = isinf(xtrain);
infsetTraina = isinf(xatrain);
featurespp = xtrain;
featuresppa = xatrain;
[b1,c1] = find(nansetTrain==1);
[b2,c2] = find(infsetTrain==1);
[ba1,ca1] = find(nansetTraina==1);
[ba2,ca2] = find(infsetTraina==1);

m1 = nanmedian(xtrain);
ma1 = nanmedian(xatrain);
for i1 = 1:length(b1)
    featurespp(b1(i1),c1(i1)) = m1(c1(i1));
end
for ia1 = 1:length(ba1)
    featuresppa(ba1(ia1),ca1(ia1)) = ma1(ca1(ia1));
end


for ia2 = 1:length(ba2)
    featuresppa(ba2(ia2),ca2(ia2)) = ma1(ca2(ia2));
end
for i2 = 1:length(b2)
    featurespp(b2(i2),c2(i2)) = m1(c2(i2));
end

xtrain = featurespp;
xatrain = featuresppa;
% doing the same on test data
nansetTest = isnan(xtest);
nansetTesta = isnan(xatest);
infsetTest = isinf(xtest);
infsetTesta = isinf(xatest);
featurespt = xtest;
featurespta = xatest;
[e1,r1] = find(nansetTest==1);
[e2,r2] = find(infsetTest==1);
[ea1,ra1] = find(nansetTesta==1);
[ea2,ra2] = find(infsetTesta==1);

for it1 = 1:length(e1)
    featurespt(e1(it1),r1(it1)) = m1(c1(it1));
end
for it1 = 1:length(e2)
    featurespt(e2(it1),r2(it1)) = m1(c2(it1));
end

for ita1 = 1:length(ea1)
    featurespta(ea1(ita1),ra1(ita1)) = ma1(ca1(ita1));
end
for ita2 = 1:length(ea2)
    featurespta(ea2(ita2),ra2(ita2)) = ma1(ca2(ita2));
end

xtest = featurespt;
xatest = featurespta;
%% PCA feature selection
%
[coeff,scoreTrain,~,~,explained,mu] = pca(xtrain);
[coeffa,scoreTraina,~,~,explaineda,mua] = pca(xatrain);
%This code returns four outputs: coeff, scoreTrain, explained, and mu. Use explained (percentage of total variance explained) to find the number of components required to explain at least 98.5% variability. Use coeff (principal component coefficients) and mu (estimated means of XTrain) to apply the PCA to a test data set. Use scoreTrain (principal component scores) instead of XTrain when you train a model.

%Display the percent variability explained by the principal components.

%explained

idx = find(cumsum(explained)>99.99,1);
idxa = find(cumsum(explaineda)>99.99,1);

% Train a classification tree using the first two components.
scoretrain = scoreTrain(:,1:idx);
scoretraina = scoreTraina(:,1:idxa);



% To use the trained model for the test set, you need to transform the test data set by using the PCA obtained from the training data set. Obtain the principal component scores of the test data set by subtracting mu from XTest and multiplying by coeff. Only the scores for the first two components are necessary, so use the first two coefficients coeff(:,1:idx).

scoretest = (xtest-mu)*coeff(:,1:idx);
scoretesta = (xatest-mua)*coeffa(:,1:idxa);
% Pass the trained model mdl and the transformed test data set scoreTest to the predict function to predict ratings for the test set.
%}

xtrain = scoretrain;
xtest = scoretest;

xatrain = scoretraina;
xatest = scoretesta;


%% excluding zeros
%{
i12 = find(ytrain ~= 0);
i13 = find(yatrain ~= 0);


xtrain = xtrain(i12,:);
xatrain = xatrain(i13,:);
ytrain = ytrain(i12,:);
yatrain = yatrain(i13,:);

%}
 %% training model & validation
% K-fold
%{
%K = 3;
K = 10;
%K = 20;
n_run = 3;
accuracy = zeros(K,n_run);
% 10_fold
for i_run=1:n_run
    indices = crossvalind('Kfold',ytrain,K);
    
    for i_fold = 1:K
        Val = indices==i_fold;
        train = ~Val;
        featureTrain = xtrain(train,:);
        featureVal = xtrain(Val,:);
        
        % Classification with KNN
        Model = fitcknn(featureTrain,ytrain(train));
        class = predict(Model, featureVal);
        %accuracy(i_fold,i_run) = 100*length(find(class == ytrain(Val)))/length(ytrain(Val));
        
        %tree decision on validation
        
       %modeltree = fitctree(featureTrain,ytrain(train),'MinLeafSize',1249);
        %preclass = predict(modeltree, featureVal);
        
        
        
    end    
      
end

for i_runa=1:n_run
    indices = crossvalind('Kfold',yatrain,K);
    
    for i_folda = 1:K
        Val = indices==i_folda;
        traina = ~Val;
        featureTraina = xatrain(traina,:);
        featureVala = xatrain(Val,:);
        
        % Classification with KNN
        Modela = fitcknn(featureTraina,yatrain(traina));
        classa = predict(Modela, featureVala);
        %accuracy(i_fold,i_run) = 100*length(find(class == ytrain(Val)))/length(ytrain(Val));
        
        %tree decision on validation
        
       %modeltreea = fitctree(featureTraina,yatrain(traina),'MinLeafSize',1249);
        %preclassa = predict(modeltreea, featureVala);
        
        
        
    end    
      
end

%}

%% one trial prediction for plotting
%
xtest = xtest(1:1:30,:);
xatest = xatest(1:1:30,:);
ytest = ytest(1:1:30,:);
yatest = yatest(1:1:30,:);
%}

%% classification resp without validation

% KNN
%
knnModel = fitcknn(xtrain,ytrain);
%,'NumNeighbors',24,'distance','mahalanobis');

class = predict(knnModel,xtest);
n = sum(class==ytest);
accuracy_of_KNN_before_saccade = n/length(xtest)*100

%%
%{
knnModela = fitcknn(xatrain,yatrain);%,'NumNeighbors',118,'distance','chebychev');
%,'OptimizeHyperparameters','auto');
%

classa = predict(knnModela,xatest);
na = sum(classa==yatest);
accuracy_of_KNN_after_saccade = na/length(xatest)*100
%}

%% NN classification (works only on matlab 2022)
%{
NNmodel = fitcnet(xtrain,ytrain);
classNN = predict(NNmodel,xtest);
accuracyOfNN = sum(classNN==ytest)/length(xtest)*100


%}

%%   KNN test after validation

%{

class = predict(Model, xtest);
n111 = sum(class==ytest);
accuracyofKNNBefore = n111/length(ytest)*100


classa = predict(Modela, xatest);
n111a = sum(classa==yatest);
accuracyofKNNAfter = n111a/length(yatest)*100
%}

%%  Tree classifier
%
classtree = fitrtree(xtrain,ytrain);
%,'MinLeafSize',1249);

cc = predict(classtree,xtest);
% ,'OptimizeHyperparameters','auto'
accuracyofTreeBefore = sum(predict(classtree,xtest) == ytest)/length(ytest)*100

classtreeafter = fitctree(xatrain,yatrain);%,'MinLeafSize',1092);

cca = predict(classtreeafter,xatest);
% ,'OptimizeHyperparameters','auto'
accuracyofTreeAfter = sum(predict(classtreeafter,xatest) == yatest)/length(yatest)*100

%}

%% tree classifier after using K-fold cross-validation
%{
tree_pre = predict(modeltree, xtest);
n2 = sum(tree_pre==ytest);
accuracyofTree = n2/length(ytest)*100
%}

%% ploting results of jj prediction
%

classtreeafter = fitrtree(xatrain,yatrain);%,'MinLeafSize',1092);

cca = predict(classtreeafter,xatest);


yy1 = ytest.' ;
yya1 = yatest.';

yy2 = cc.';
yya2 = cca.';
%yy2 = class.';
%yya2 = classa.';

new_smoothing_bandwidth = 4 ;
yy1 = smooth(yy1, new_smoothing_bandwidth);
yy2 = smooth(yy2, new_smoothing_bandwidth);
yya1 = smooth(yya1, new_smoothing_bandwidth);
yya2 = smooth(yya2, new_smoothing_bandwidth);

figure; hold on
title('prediction and ground true')
plot(yy1)
plot(yy2)
plot(yya1)
plot(yya2)

legend('ture value before saccade','predicted value before saccade',...
    'ture value after saccade','predicted value after saccade');

%}


%% correlation calculation
%
correlationOfbeforeSaccadePrediction = sum(xcorr( yy1 , yy2 ))
correlationOfAfterSaccadePrediction  = sum(xcorr( yya1 , yya2 ))

%% ISI & Cv

%} 