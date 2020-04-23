clear; clc; close all;
load('variables_matrix'); load('frequency'); load('setting_conditions');
load('results3')

temperature(:,1) = Lizzy_temperatures_matrix(:,16);
sample_length = length(temperature);
t = 1:sample_length;
wn = Lizzy_fnewall;

subplot(2,1,1)
plot(t,temperature)
xlabel('Observations'); ylabel('Temperature (C)');
xline(3476,'--');
xline(1201,'--r');
xline(1500,'--r');
subplot(2,1,2)
plot(t,wn)
xlabel('Observations'); ylabel('Natural Frequencies (Hz)');
xline(3476,'--');
xline(1201,'--r');
xline(1500,'--r');

[threshold1,threshold2] = monte_carlo_threshold();

%% standardisation of natural frequencies

for i=1:4
    wn_std(:,i) = (wn(:,i) - mean(wn(:,i)))./std(wn(:,i));
end

figure
plot(t,wn_std)
xlabel('Observations'); ylabel('Standardized Natural Frequencies (Hz)');
xline(3476,'--','LineWidth', 2);

%% PCA of natural frequencies

tic % start timer
[coeff, pca_score] = pca(wn_std); % possibly macplot
elapsed_time.pca(1) = toc;

figure
gscatter(pca_score(:,1),pca_score(:,2),general_conditions,'rgb','...',8)
xlabel('1st Principal Component','FontSize',14); 
ylabel('2nd Principal Component','FontSize',14);
legend('Location','southoutside','FontSize',12)

%% tSNE with different random walk

rng('default'); tic
[score.tsne{1},loss.tsne{1},grad.tsne{1}] = tsne_grad(pca_score); elapsed_time.tsne(1) = toc;
for i =2:4
    tic
    [score.tsne{i},loss.tsne{i},grad.tsne{i}] = tsne_grad(pca_score); elapsed_time.tsne(i) = toc;
end

%% tSNE figures

figure
for i = 1:4
    subplot(2,2,i)
    gscatter(score.tsne{i}(:,1),score.tsne{i}(:,2),general_conditions,...
        'rgb','...');
    xlabel('Dimension 1'); ylabel('Dimension 2'); legend('off')
    title(['tSNE sample ',num2str(i)])
end

%% tSNE with different perplexity

%perplexity = [1,2,5,10,20,30,50,75,100];
perplexity = [1,10,30,50,100,150,200,250,300];
j=1;

for i = perplexity 
        rng('default'); tic
        [score.tsneperp{j},loss.tsneperp(j),grad.tsneperp(j)] ...
            = tsne_grad(pca_score,'Perplexity',i); ...
            elapsed_time.tsneperp(j) = toc;
        j = j+1;
end

%% tSNE perplexity figures

j=1;
figure
for i = perplexity
    subplot(3,3,j)
    gscatter(score.tsneperp{j}(:,1),score.tsneperp{j}(:,2),general_conditions,...
        'rgb','.ox');
    title(['tSNE Perplexity of ',num2str(i)])
    xlabel('Dimension 1'); ylabel('Dimension 2'); legend('off')
    j = j+1;
end

figure
subplot(2,1,1)
plot(perplexity,loss.tsneperp,'-x')
xlabel('Perplexity'); ylabel('KL divergnce'); legend('off')
subplot(2,1,2)
plot(perplexity,grad.tsneperp,'-x')
xlabel('Perplexity'); ylabel('Gradient of descent'); legend('off')

%% tSNE with different perplexity2222222

%perplexity = [1,2,5,10,20,30,50,75,100];
perplexity = [2,10,30,50,100,150,200,250,300];
j=1;

for i = perplexity 
        rng('default'); tic
        [score.tsneperp2{j},loss.tsneperp2(j),grad.tsneperp2(j)] ...
            = tsne_grad(pca_score,'Perplexity',i); ...
            elapsed_time.tsneperp2(j) = toc;
        j = j+1;
end

%% tSNE perplexity figures222222

j=1;
figure
for i = perplexity
    subplot(3,3,j)
    gscatter(score.tsneperp2{j}(:,1),score.tsneperp2{j}(:,2),general_conditions,...
        'rgb','...');
    title(['tSNE Perplexity of ',num2str(i)])
    xlabel('Dimension 1'); ylabel('Dimension 2'); legend('off')
    j = j+1;
end

figure
subplot(2,1,1)
plot(perplexity,loss.tsneperp2,'-x')
xlabel('Perplexity'); ylabel('KL divergnce'); legend('off')
subplot(2,1,2)
plot(perplexity,grad.tsneperp2,'-x')
xlabel('Perplexity'); ylabel('Gradient of descent'); legend('off')

%% Number of iterations sensitivity

iterations = [10,25,50,75,100,250,500,750,1000];
j=1;

for i = iterations
    opts = statset('MaxIter',i);
    rng('default'); tic
    [score.tsneitr{j},loss.tsneitr(j),grad.tsneitr(j)] ...
        = tsne_grad(pca_score,'NumPrint',100,'Verbose',1,'Options',opts); ...
        elapsed_time.tsne_itr(j) = toc;
    j = j+1;
end

%% tSNE iterations figures

j=1;
figure
for i = iterations
    subplot(3,3,j)
    gscatter(score.tsneitr{j}(:,1),score.tsneitr{j}(:,2),general_conditions,...
        'rgb','.ox');
    title(['tSNE ',num2str(i),' iterations'])
    xlabel('Dimension 1'); ylabel('Dimension 2'); legend('off')
    j = j+1;
end

figure
subplot(2,1,1)
plot(iterations,loss.tsneitr,'-x')
xlabel('Iterations'); ylabel('KL divergnce'); legend('off')
subplot(2,1,2)
plot(iterations,grad.tsneitr,'-x')
xlabel('Iterations'); ylabel('Gradient of descent'); legend('off')


%% Exaggeration sensitivity test

exaggeration = [1,1.5,2,4,6,10,15,20,25];
j = 1;

for i = exaggeration
    rng('default'); tic
    [score.tsneexag{j},loss.tsneexag(j),grad.tsneexag(j)] ...
        = tsne_grad(pca_score,'NumPrint',100,'Verbose',1,'Exaggeration',i); ...
        elapsed_time.tsne_exag(j) = toc;
    j = j+1;
end

%% tSNE exaggeration figures

j=1;
figure
for i = exaggeration 
    subplot(3,3,j)
    gscatter(score.tsneexag{j}(:,1),score.tsneexag{j}(:,2),general_conditions,...
        'rgb','.ox');
    title(['tSNE exaggeration of ',num2str(i)])
    xlabel('Dimension 1'); ylabel('Dimension 2'); legend('off')
    j = j+1;
end

figure
subplot(2,1,1)
plot(exaggeration,loss.tsneexag,'-x')
xlabel('Exaggeration'); ylabel('KL divergnce'); legend('off')
subplot(2,1,2)
plot(exaggeration,grad.tsneexag,'-x')
xlabel('Exaggeration'); ylabel('Gradient of descent'); legend('off')

%% Learning rate sensitivity test

epsilon = [2,10,20,50,100,250,500,750,1000];
j = 1;

for i = epsilon
    rng('default'); tic
    [score.tsnelr{j},loss.tsnelr(j),grad.tsnelr(j)] ...
        = tsne_grad(pca_score,'NumPrint',100,'Verbose',1,'LearnRate',i); ...
        elapsed_time.tsne_lr(j) = toc;
    j = j+1;
end

%% Learning rate figures

j=1;
figure
for i = epsilon 
    subplot(3,3,j)
    gscatter(score.tsnelr{j}(:,1),score.tsnelr{j}(:,2),general_conditions,...
        'rgb','.ox');
    title(['tSNE epsilon=',num2str(i)])
    xlabel('Dimension 1'); ylabel('Dimension 2'); legend('off')
    j = j+1;
end

figure
subplot(2,1,1)
plot(epsilon,loss.tsnelr,'-x')
xlabel('Learning rate'); ylabel('KL divergnce'); legend('off')
subplot(2,1,2)
plot(epsilon,grad.tsnelr,'-x')
xlabel('Learning rate'); ylabel('Gradient of descent'); legend('off')

%% Sensitivity final embedding or tuned

rng('default'); tic
[score.tsnefinal,loss.tsnefinal,grad.tsnefinal] = tsne_grad(wn,...
    'NumPrint',100,'Verbose',1,'Perplexity',200,'Exaggeration',4,'LearnRate',500);
elapsed_time.tsnefinal = toc;

%% Final t-sne figure

figure
gscatter(score.tsnefinal(:,1),score.tsnefinal(:,2),general_conditions,...
        'rgb','.ox');
title('Final/tuned t-SNE embedding')
xlabel('Dimension 1'); ylabel('Dimension 2'); legend('off')

%% Discussion 1

figure
subplot(1,3,1)
gscatter(pca_score(:,1),pca_score(:,2),general_conditions,'kkk','...')
xlabel('1st Principal Component','FontSize',14); 
ylabel('2nd Principal Component','FontSize',14);
title('PCA (Linear)','FontSize',14)
legend('off')

subplot(1,3,2)
gscatter(score.tsne{1}(:,1),score.tsne{1}(:,2),general_conditions,'kkk','...')
xlabel('Dimension 1','FontSize',14); 
ylabel('Dimension 2','FontSize',14);
title('Typical t-SNE (Nonlinear)','FontSize',14)
legend('off')

subplot(1,3,3)
gscatter(score.tsnefinal(:,1),score.tsnefinal(:,2),general_conditions,'kkk','...')
xlabel('Dimension 1','FontSize',14); 
ylabel('Dimension 2','FontSize',14);
title('Tuned t-SNE (Nonlinear)','FontSize',14)
legend('off')

%% Discussion 2

score.gmm{1} = pca_score(:,1:2);
score.gmm{2} = score.tsne{1};
score.gmm{3} = score.tsnefinal;

titlegmm{1} = 'PCA';
titlegmm{2} = 'Typical t-SNE';
titlegmm{3} = 'Tuned t-SNE';

figure
for sample = [1,2,3]
    
    GMModel_1 = fitgmdist(score.gmm{sample},2);
    GMModel_2 = fitgmdist(score.gmm{sample},3);

    subplot(3,2,sample*2-1)
    gscatter(score.gmm{sample}(:,1),score.gmm{sample}(:,2),general_conditions);
    h = gca;
    hold on
    gmPDF = @(x1,x2)arrayfun(@(x1,x2)pdf(GMModel_1,[x1(:) x2(:)]),x1,x2);
    fcontour(gmPDF,[h.XLim h.YLim],'MeshDensity',100)
    title([titlegmm{sample},' - 2 comps'],'FontSize',10)
    xlabel('Dimension 1');
    ylabel('Dimension 2'); legend('off')
    hold off
    subplot(3,2,sample*2)
    gscatter(score.gmm{sample}(:,1),score.gmm{sample}(:,2),general_conditions);
    h = gca;
    hold on
    gmPDF = @(x1,x2)arrayfun(@(x1,x2)pdf(GMModel_2,[x1(:) x2(:)]),x1,x2);
    fcontour(gmPDF,[h.XLim h.YLim],'MeshDensity',100)
    title([titlegmm{sample},' - 3 comps'],'FontSize',10)
    xlabel('Dimension 1');
    ylabel('Dimension 2'); legend('off')
    hold off
end

%% Mahal distance

score.gmm{1} = pca_score(:,1:2);
score.gmm{2} = score.tsne{1};
score.gmm{3} = score.tsnefinal;

titlegmm{1} = 'PCA';
titlegmm{2} = 'Typical t-SNE';
titlegmm{3} = 'Tuned t-SNE';

figure
for sample = [1,2,3]
    novelty_index = mahal(score.gmm{sample},score.gmm{sample}(1:1000,:));
    subplot(3,1,sample)
    gscatter(t,novelty_index,general_conditions)
    title([titlegmm{sample}],'FontSize',12)
    xlabel('Observations')
    ylabel('Mahal sq-distance'); legend('off')
    %yline(threshold1)
    yline(threshold2)
    
end

%% CDF

x = icdf('Chisquare',novelty_index,2);
y = cdf('Chisquare',novelty_index,2);
pd = 1-chi2pdf(novelty_index,2);

%% PCA mahal

figure
novelty_index = mahal(pca_score,pca_score(1:1000,:));
gscatter(t,novelty_index,general_conditions)

%% Independent component analysis ICA of natural frequencies

tic
[ica_score,U,icapca_score,V] = pcaica(wn',4); elapsed_time.ica = toc;

%% tSNE of ICA

tic
score.tsneica = tsne(ica_score','Algorithm','exact'); elapsed_time.tsneica(1) = toc;

%% Plotting of tSNE ICA figures

figure

subplot(1,2,1)
gscatter(ica_score(1,:),ica_score(2,:),general_conditions);
title('ICA of natural frequencies','FontSize',11);
xlabel('1st Independent Component'); ylabel('2nd Independent Component');
legend('off')

subplot(1,2,2)
gscatter(score.tsneica(:,1),score.tsneica(:,2),general_conditions);
title('t-SNE embedding of ICA','FontSize',11);
xlabel('Dimension 1'); ylabel('Dimension 2');
legend('off')

%% save results
save_results()