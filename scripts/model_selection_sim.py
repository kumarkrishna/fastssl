import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os, glob
from linclab_utils import plot_utils
from tqdm import tqdm
from sklearn.metrics import r2_score

plot_utils.linclab_plt_defaults(font="Arial",fontdir=os.path.expanduser('~')+"/Projects/fonts") 	# Run locally, not from cluster

plot_abs = False
flag_debug = False
calc_new_alpha = True
R2_thresh = 0.95
dataset_ssl = 'cifar10'
dataset_classifier = 'cifar10'
ckpt_dir = 'checkpoints_design_hparams_{}'.format(dataset_ssl)

def stringer_get_powerlaw(ss, trange):
    # COPIED FROM Stringer+Pachitariu 2018b github repo! (https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/blob/master/python/utils.py)
    ''' fit exponent to variance curve'''
    logss = np.log(np.abs(ss))
    y = logss[trange][:,np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:,np.newaxis], np.ones((nt,1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:,np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()
    
    allrange = np.arange(0, ss.size).astype(int) + 1
    x = np.concatenate((-np.log(allrange)[:,np.newaxis], np.ones((ss.size,1))), axis=1)
    ypred = np.exp((x * b).sum(axis=1))
    alpha = b[0]
    max_range = 500 if len(ss)>=512 else len(ss) - 10   # subtracting 10 here arbitrarily because we want to avoid the last tail!
    fit_R2 = r2_score(y_true=logss[trange[0]:max_range],y_pred=np.log(np.abs(ypred))[trange[0]:max_range])
    try:
        fit_R2_range = r2_score(y_true=logss[trange],y_pred=np.log(np.abs(ypred))[trange])
    except:
        fit_R2_range = None
    return alpha, ypred, fit_R2, fit_R2_range

def plot_alpha_fit(data_dict,trange,lamda_val,pdim_val):
	eigen = data_dict['eigenspectrum']
	al,ypred,r2,r2_range = stringer_get_powerlaw(eigen,trange)
	print(r2,r2_range)
	plt.loglog(np.arange(1,1+200),eigen[:200])
	plt.loglog(np.arange(1,1+200),ypred[:200])
	plt.title("lamda = {:.2e}, pdim = {}, alpha={:.3f}".format(lamda_val,pdim_val,al))
	plt.show()

def retrieve_from_dict(sequence_dict,idx):
	return sequence_dict[idx]

def simulate_model_selection(pdim_arr,lamda_arr,alpha_dict,accuracy_dict):
	alpha_sequence_dict = {}
	pdim_sequence_dict = {}
	lamda_sequence_dict = {}
	accuracy_sequence_dict = {}
	model_idx = 0
	for pidx,pdim in enumerate(pdim_arr):
		for lidx,lamda in enumerate(lamda_arr):
			try:
				if type(alpha_dict[pdim][lamda]) is list:
					curr_alpha = np.mean(np.array(alpha_dict[pdim][lamda]))
				else:
					curr_alpha = alpha_dict[pdim][lamda]
				if type(accuracy_dict[pdim][lamda]) is list:
					curr_accuracy = np.mean(np.array(accuracy_dict[pdim][lamda]))
				else:
					curr_accuracy = accuracy_dict[pdim][lamda]
				alpha_sequence_dict[model_idx] = curr_alpha
				accuracy_sequence_dict[model_idx] = curr_accuracy
				lamda_sequence_dict[model_idx] = lamda
				pdim_sequence_dict[model_idx] = pdim
				model_idx += 1
			except:
				pass
	
	total_models = model_idx
	best_accuracy = np.array([val for key,val in accuracy_sequence_dict.items()]).max()

	hparam_search_repeats = 1000
	parallel_models = 10
	eval_prob_arr_repeats = []
	total_linear_evals_repeats = []
	num_correct_search = 0 
	search_tolerance = 1 	# setting tolerance to 1%
	error_in_best_model_search = 0
	max_error_in_best_model_search = 0

	for ridx in tqdm(range(hparam_search_repeats)):
		np.random.seed(ridx)
		random_shuffle_order = np.random.permutation(total_models)
		alpha_min = 0 
		alpha_max = 10000
		total_linear_evals = []
		alpha_log_arr = []
		accuracy_log_arr = []
		eval_prob_arr = []
		search_step_indices = np.arange(parallel_models,total_models,parallel_models)
		if search_step_indices[-1]!= total_models:
			search_step_indices = np.append(search_step_indices,total_models)
		search_step_indices = np.insert(search_step_indices,0,0)
		for m in np.arange(1,len(search_step_indices)):
			model_indices = random_shuffle_order[search_step_indices[m-1]:search_step_indices[m]]
			curr_alpha_arr = np.array([retrieve_from_dict(alpha_sequence_dict,i) for i in model_indices])
			compare_idx = np.logical_and(curr_alpha_arr>=alpha_min,curr_alpha_arr<=alpha_max) 	# check which models have alpha in range [alpha_min, alpha_max]
			# good models indices
			good_model_indices = model_indices[compare_idx]
			# add good model alphas to alpha_log_arr
			alpha_log_arr.extend(curr_alpha_arr[compare_idx])
			# add good model accuracies to accuracy_log_arr
			accuracy_log_arr.extend([retrieve_from_dict(accuracy_sequence_dict,i) for i in good_model_indices])
			# Count of linear evals increased
			if len(total_linear_evals)==0:
				total_linear_evals.append(compare_idx.sum())
			else:
				total_linear_evals.append(total_linear_evals[-1]+compare_idx.sum())
			eval_prob_arr.append(compare_idx.sum()/parallel_models)
			# breakpoint()
			# update alpha_min and alpha_max
			# Step 1: find the top k models, k = parallel_models//2 here but can be set to some other number
			curr_accuracy_arr_np = np.array(accuracy_log_arr)
			sort_idx = np.argsort(curr_accuracy_arr_np)
			top_k_models = sort_idx[-int(parallel_models//2):]
			# Step 2: Sort the alpha log array
			curr_alpha_arr_np = np.array(alpha_log_arr)
			alpha_sort_idx = np.argsort(curr_alpha_arr_np)
			# Step 3: Go over the sorted alpha array and find alpha_min and alpha_max
			top_model_flag_arr = [accuracy_log_arr[idx] in curr_accuracy_arr_np[top_k_models] for idx in alpha_sort_idx]
			# breakpoint()
			true_idx = [idx for (idx,item) in enumerate(top_model_flag_arr) if item==True]
			if true_idx[0]!=0:
				alpha_min = alpha_log_arr[alpha_sort_idx[true_idx[0]]]
			if true_idx[-1]!=len(top_model_flag_arr)-1:
				alpha_max = alpha_log_arr[alpha_sort_idx[true_idx[-1]]]
		eval_prob_arr_repeats.append(eval_prob_arr)
		total_linear_evals_repeats.append(total_linear_evals)
		best_model_accuracy = np.array(accuracy_log_arr).max()
		num_correct_search += (best_accuracy-best_model_accuracy)<=search_tolerance
		error_in_best_model_search += (best_accuracy-best_model_accuracy)
		if (best_accuracy-best_model_accuracy)>max_error_in_best_model_search:
			max_error_in_best_model_search = (best_accuracy-best_model_accuracy)

	# print(eval_prob_arr,total_linear_evals)
	# print(np.array(accuracy_log_arr).max())
	# all_accuracies = np.array([val for key,val in accuracy_sequence_dict.items()])
	# print(all_accuracies.max())
	eval_prob_arr_repeats = np.array(eval_prob_arr_repeats)
	total_linear_evals_repeats = np.array(total_linear_evals_repeats)
	avg_linear_evals = total_linear_evals_repeats.mean()
	mean_eval_prob_arr_repeats = np.mean(eval_prob_arr_repeats,axis=0)
	std_eval_prob_arr_repeats = np.std(eval_prob_arr_repeats,axis=0)
	mean_total_linear_evals_repeats = np.mean(total_linear_evals_repeats,axis=0)
	std_total_linear_evals_repeats = np.std(total_linear_evals_repeats,axis=0)

	print("Probability of correct model accuracy:",num_correct_search/hparam_search_repeats)
	print("Mean error in best accuracy search:",error_in_best_model_search/hparam_search_repeats)
	print("Max error in best accuracy search:",max_error_in_best_model_search)
	print("Expected Number of linear evals:",avg_linear_evals)

	plt.figure()
	plt.errorbar(x=np.arange(1,1+mean_eval_prob_arr_repeats.shape[0]),y=mean_eval_prob_arr_repeats,yerr=std_eval_prob_arr_repeats,label='Empirical probability')
	plt.plot(np.arange(1,1+mean_eval_prob_arr_repeats.shape[0]),1./np.arange(1,1+mean_eval_prob_arr_repeats.shape[0]),label=r'$\frac{1}{steps}$ decay',ls='--')
	plt.grid('on')
	plt.xlabel('Sequential steps')
	plt.ylabel('Linear eval probability')
	plt.legend()

	plt.figure()
	plt.errorbar(x=np.arange(1,1+mean_total_linear_evals_repeats.shape[0]),y=mean_total_linear_evals_repeats,yerr=std_total_linear_evals_repeats,label=r'$\alpha$-based filtering')
	plt.plot(np.arange(1,1+mean_total_linear_evals_repeats.shape[0]),parallel_models*np.arange(1,1+mean_total_linear_evals_repeats.shape[0]),label='No filtering')
	plt.grid('on')
	plt.xlabel('Sequential steps')
	plt.ylabel('Total number of linear evals')
	plt.legend()
	plt.show()



def plot_colorplot(pdim_arr,lamda_arr,attribute_dict,attribute_label,vmin=None,vmax=None,cmap='seismic'):
	plotting_arr = np.empty((len(pdim_arr),len(lamda_arr)))
	plotting_arr[:] = np.nan
	for pidx,pdim in enumerate(pdim_arr):
		for lidx,lamda in enumerate(lamda_arr):
			try:
				if type(attribute_dict[pdim][lamda]) is list:
					plotting_arr[pidx,lidx] = np.mean(np.array(attribute_dict[pdim][lamda]))
				else:
					plotting_arr[pidx,lidx] = attribute_dict[pdim][lamda]
			except:
				pass
	plt.figure()
	plt.imshow(plotting_arr,cmap=cmap,vmin=vmin,vmax=vmax,origin='lower')
	plt.colorbar()
	plt.title(attribute_label)
	plt.yticks(np.arange(len(pdim_arr)),pdim_arr)
	plt.xticks(np.arange(0,len(lamda_arr),2),lamda_arr[np.arange(0,len(lamda_arr),2)])
	plt.gca().set_xticklabels(FormatStrFormatter('%.2e').format_ticks(lamda_arr[np.arange(0,len(lamda_arr),2)]))
	plt.ylabel('Projector Dimensionality')
	plt.xlabel(r'$\lambda$')

def plot_scatterplot(attribute1_dict,attribute1_label,attribute2_dict,attribute2_label,filter_dict=None,vmin=None,vmax=None,hmin=None,hmax=None):
	plt.figure()
	markers_arr = ['o','^','x','s','d','v','*']
	for pidx,pdim in enumerate(attribute1_dict.keys()):
		attr1_arr = []
		attr1_err_arr = []
		attr2_arr = []
		attr2_err_arr = []
		for lidx,lamda in enumerate(attribute1_dict[pdim].keys()):
			if filter_dict is not None and np.nanmean(np.array(filter_dict[pdim][lamda]))<R2_thresh: continue
			if type(attribute1_dict[pdim][lamda]) is list:
				attr1_arr.append(np.mean(np.array(attribute1_dict[pdim][lamda])))
				attr1_err_arr.append(np.std(np.array(attribute1_dict[pdim][lamda])))
			else:
				attr1_arr.append(attribute1_dict[pdim][lamda])
				attr1_err_arr.append(0)
			if type(attribute2_dict[pdim][lamda]) is list:
				attr2_arr.append(np.mean(np.array(attribute2_dict[pdim][lamda])))
				attr2_err_arr.append(np.std(np.array(attribute2_dict[pdim][lamda])))
			else:
				attr2_arr.append(attribute2_dict[pdim][lamda])
				attr2_err_arr.append(0)
	
		# plt.scatter(attr1_arr,attr2_arr,marker=markers_arr[pidx%len(markers_arr)],label='pdim={}'.format(int(pdim)))
		plt.errorbar(x=attr1_arr,y=attr2_arr,xerr=attr1_err_arr,yerr=attr2_err_arr,marker=markers_arr[pidx%len(markers_arr)],ls='',label='pdim={}'.format(int(pdim)))
	plt.title('{} vs {}'.format(attribute1_label,attribute2_label))
	plt.xlabel('{}'.format(attribute1_label))
	plt.ylabel('{}'.format(attribute2_label))
	plt.legend()
	plt.ylim([vmin,vmax])
	plt.xlim([hmin,hmax])


files = glob.glob(os.path.join(ckpt_dir,'*'))
sorting_order = [i[0] for i in sorted(enumerate(files),key=lambda x: float(os.path.basename(x[1]).split('pdim_')[-1].split('_')[0])+float(os.path.basename(x[1]).split('lambd_')[-1].split('_')[0]))]
files_sorted = [files[idx] for idx in sorting_order]

lamda_arr = {}
pdim_arr = {}
accuracy_dict = {}
best_accuracy_dict = {}
alpha_dict = {}
SSL_loss_dict ={}
R2_dict = {}
R2_100_dict = {}
alpha_correction_due_to_minibatches = 0.0 #0.1
for fidx,file in enumerate(tqdm(files_sorted)):
	try:
		lamda_val = float(os.path.basename(file).split('lambd_')[-1].split('_')[0])
		pdim_val = float(os.path.basename(file).split('pdim_')[-1].split('_')[0])
		# if lamda_val <= 1e-6: continue
		lamda_arr[lamda_val]=True
		pdim_arr[pdim_val]=True
		if pdim_val not in accuracy_dict.keys():
			accuracy_dict[pdim_val] = {}
		if pdim_val not in best_accuracy_dict.keys():
			best_accuracy_dict[pdim_val] = {}
		if pdim_val not in alpha_dict.keys():
			alpha_dict[pdim_val] = {}
		if pdim_val not in SSL_loss_dict.keys():
			SSL_loss_dict[pdim_val] = {}
		if pdim_val not in R2_dict.keys():
			R2_dict[pdim_val] = {}
		if pdim_val not in R2_100_dict.keys():
			R2_100_dict[pdim_val] = {}
		SSL_fname = os.path.join(file,'results_{}_early_alpha_ssl_100.npy'.format(dataset_ssl))
		if os.path.exists(SSL_fname): 
			SSL_file = np.load(SSL_fname,allow_pickle=True).item()
			SSL_loss_dict[pdim_val][lamda_val] = np.log(SSL_file['train_loss'][-1])/pdim_val
		else:
			SSL_loss_dict[pdim_val][lamda_val] = np.nan

		linear_files = glob.glob(os.path.join(file,'results_{}_early_alpha_linear_200*'.format(dataset_classifier)))
		for linear_fname in linear_files:
			# linear_fname = os.path.join(file,'results_{}_early_alpha_linear_200.npy'.format(dataset))
			linear_dict = np.load(linear_fname,allow_pickle=True).item()
			if lamda_val not in alpha_dict[pdim_val].keys():
				alpha_dict[pdim_val][lamda_val] = []
				R2_dict[pdim_val][lamda_val] = []
				R2_100_dict[pdim_val][lamda_val] = []
				accuracy_dict[pdim_val][lamda_val] = []
				best_accuracy_dict[pdim_val][lamda_val] = []

			if (calc_new_alpha and linear_dict['R2_100']<R2_thresh) or dataset_classifier!=dataset_ssl:
				eigen = linear_dict['eigenspectrum']
				if dataset_ssl!=dataset_classifier:
					range_init = 11
				else:
					range_init = 5
				alpha,ypred,R2,r2_range = stringer_get_powerlaw(eigen,np.arange(range_init,50))
				if r2_range<R2_thresh:
					alpha,ypred,R2,r2_range = stringer_get_powerlaw(eigen,np.arange(range_init,20))
				R2_100 = r2_range
			else:
				alpha = linear_dict['alpha']
				R2 = linear_dict['R2']
				R2_100 = linear_dict['R2_100']
			if plot_abs:
				alpha_dict[pdim_val][lamda_val].append(np.abs(1-alpha))
			else:
				alpha_dict[pdim_val][lamda_val].append(alpha)
			R2_dict[pdim_val][lamda_val].append(R2)
			R2_100_dict[pdim_val][lamda_val].append(R2_100)
			test_acc_arr = np.array(linear_dict['test_acc_1'])
			# breakpoint()

			final_test_acc = test_acc_arr[-1]
			best_test_acc = test_acc_arr.max()
			accuracy_dict[pdim_val][lamda_val].append(final_test_acc)
			best_accuracy_dict[pdim_val][lamda_val].append(best_test_acc)
		if flag_debug: 
			plot_alpha_fit(linear_dict,trange=np.arange(3,100),lamda_val=lamda_val,pdim_val=pdim_val)
			breakpoint()
		
	except:
		breakpoint()
		pass


lamda_arr = np.sort(np.array(list(lamda_arr.keys())))
pdim_arr = np.sort(np.array(list(pdim_arr.keys())))
print(lamda_arr)
simulate_model_selection(pdim_arr=pdim_arr,lamda_arr=lamda_arr,alpha_dict=alpha_dict,accuracy_dict=accuracy_dict)
# plot_colorplot(pdim_arr,lamda_arr,accuracy_dict,"Final accuracy",cmap='coolwarm',vmin=70,vmax=85 if dataset_ssl!=dataset_classifier in dataset_classifier else 90)
# plot_colorplot(pdim_arr,lamda_arr,alpha_dict,r"$|1-\alpha|$" if plot_abs else r"$\alpha$",cmap='coolwarm_r',vmax=1.2 if plot_abs else 2.2)
# plot_colorplot(pdim_arr,lamda_arr,R2_100_dict,r"$R^2$ (top 100)",cmap='coolwarm',vmin=0.90)
# plot_colorplot(pdim_arr,lamda_arr,SSL_loss_dict,"SSL loss (log)",cmap='coolwarm_r')#,vmax=6000)
# plot_scatterplot(alpha_dict,r"$|1-\alpha|$" if plot_abs else r"$\alpha$",accuracy_dict,"Final accuracy",filter_dict=R2_100_dict,vmin=70,vmax=85 if dataset_ssl!=dataset_classifier in dataset_classifier else 90,
# 																																hmax=2 if 'stl' in dataset_classifier else 2.2)
# if plot_abs:
# 	plt.axvline(x=0.0,color='k',ls='--')
# 	lim_min,lim_max = plt.xlim()
# 	if lim_min>-0.05: lim_min = -0.05
# else:
# 	plt.axvline(x=1.0,color='k',ls='--')
# 	plt.axvline(x=0.8,color='k',ls=':')
# 	lim_min,lim_max = plt.xlim()
# 	if lim_min>0.95: lim_min = 0.95
# plt.xlim([lim_min,lim_max])
# plt.show()



# accuracy_arr = np.array(accuracy_arr)
# alpha_mean_arr = np.array(alpha_mean_arr)
# alpha_std_arr = np.array(alpha_std_arr)
# plt.figure()
# if plot_abs:
# 	plt.errorbar(x=np.abs(1-alpha_mean_arr),y=accuracy_arr,xerr=alpha_std_arr,ls='',marker='o')
# 	plt.xlabel(r'|1-$\alpha$|')
# 	plt.axvline(x=0.0,color='k',ls=':')
# else:
# 	plt.errorbar(x=alpha_mean_arr,y=accuracy_arr,xerr=alpha_std_arr,ls='',marker='o')
# 	plt.xlabel(r'$\alpha$')
# 	plt.axvline(x=1.0,color='k',ls=':')
# plt.ylabel('Test accuracy')
# plt.title('BarlowTwins trained features on {}'.format(dataset))
# plt.grid('on')
# plt.ylim([81,84])

# plt.figure()
# plt.plot(lamda_arr,accuracy_arr,ls='--',marker='o')
# plt.xlabel(r'$\lambda$')
# plt.ylabel('Test accuracy')
# plt.xscale('log')
# plt.title('BarlowTwins trained features on {}'.format(dataset))
# plt.grid('on')
# plt.show()

