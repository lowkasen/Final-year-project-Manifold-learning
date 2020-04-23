file_name = 'results3.mat';

save(file_name,'elapsed_time','-append')
save(file_name,'grad','-append')
save(file_name,'score','-append')
save(file_name,'loss','-append')

% save(file_name,'elapsed_time','-append')
% save(file_name,'ica_score','-append')
% save(file_name,'icapca_score','-append')
% 
% save(file_name,'lle1_score','-append')
% save(file_name,'lle2_score','-append')
% save(file_name,'lle3_score','-append')
% save(file_name,'lle_scores','-append')
% 
% save(file_name,'pca_score','-append')
% 
% save(file_name,'tsne_scores','-append')
% save(file_name,'tsne_perp1','-append')
% save(file_name,'tsne_perp2','-append')
% save(file_name,'tsne_perp3','-append')
% save(file_name,'tsne_perp4','-append')
% save(file_name,'tsne_perp5','-append')
% save(file_name,'tsne_perp6','-append')
% 
% save(file_name,'tsne_itr_scores','-append')
% save(file_name,'tsne_lr_scores','-append')
% save(file_name,'tsne_mom_scores','-append')
% 
% save(file_name,'tsneica_scores','-append')