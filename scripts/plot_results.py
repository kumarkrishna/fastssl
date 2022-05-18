import numpy as np
import matplotlib.pyplot as plt
import os, glob
from linclab_utils import plot_utils

plot_utils.linclab_plt_defaults(font="Arial",fontdir=os.path.expanduser('~')+"/Projects/fonts") 	# Run locally, not from cluster

plot_abs = False
dataset = 'cifar10'
ckpt_dir = 'checkpoints'

files = glob.glob(os.path.join(ckpt_dir,'*'))
sorting_order = [i[0] for i in sorted(enumerate(files),key=lambda x: float(os.path.basename(x[1]).split('lambd_')[-1].split('_')[0]))]
files_sorted = [files[idx] for idx in sorting_order]

lamda_arr = []
accuracy_arr = []
alpha_mean_arr = []
alpha_std_arr = []
alpha_correction_due_to_minibatches = 0.0 #0.1
for fidx,file in enumerate(files_sorted[:-1]):
	try:
		lamda_val = float(os.path.basename(file).split('lambd_')[-1].split('_')[0])
		res_fname = os.path.join(file,'results_{}_early_alpha_linear_200.npy'.format(dataset))
		alpha_dict_fname = os.path.join(file,'results_{}_full_early_alpha_linear_200.npy'.format(dataset))
		res_dict = np.load(res_fname,allow_pickle=True).item()
		alpha_dict = np.load(alpha_dict_fname,allow_pickle=True).item()
		test_acc_arr = np.array(res_dict['test_acc_1'])
		alpha_arr = res_dict['alpha_arr']
		R2_arr = res_dict['R2_100_arr']
		# breakpoint()

		final_test_acc = test_acc_arr[-1]
		best_test_acc = test_acc_arr.max()
		reliable_R2_idx = R2_arr>0
		reliable_R2_arr = R2_arr[reliable_R2_idx]
		reliable_alpha_arr = alpha_arr[reliable_R2_idx]
		
		# alpha_mean = np.average(reliable_alpha_arr,weights=reliable_R2_arr)
		# alpha_variance = np.average((reliable_alpha_arr-alpha_mean)**2,weights=reliable_R2_arr) 	# NOTE: this is a biased estimate of variance
		# alpha_std = np.sqrt(alpha_variance)
		# alpha_mean = np.mean(alpha_arr)
		# alpha_std = np.std(alpha_arr)
		alpha_mean = alpha_dict['alpha']
		alpha_std = 0
		print(reliable_R2_arr.shape,alpha_mean)
		lamda_arr.append(lamda_val)
		accuracy_arr.append(final_test_acc)
		alpha_mean_arr.append(alpha_mean+alpha_correction_due_to_minibatches)
		alpha_std_arr.append(alpha_std)
	except:
		pass
lamda_arr = np.array(lamda_arr)
accuracy_arr = np.array(accuracy_arr)
alpha_mean_arr = np.array(alpha_mean_arr)
alpha_std_arr = np.array(alpha_std_arr)
plt.figure()
if plot_abs:
	plt.errorbar(x=np.abs(1-alpha_mean_arr),y=accuracy_arr,xerr=alpha_std_arr,ls='',marker='o')
	plt.xlabel(r'|1-$\alpha$|')
	plt.axvline(x=0.0,color='k',ls=':')
else:
	plt.errorbar(x=alpha_mean_arr,y=accuracy_arr,xerr=alpha_std_arr,ls='',marker='o')
	plt.xlabel(r'$\alpha$')
	plt.axvline(x=1.0,color='k',ls=':')
plt.ylabel('Test accuracy')
plt.title('BarlowTwins trained features on {}'.format(dataset))
plt.grid('on')
plt.ylim([81,84])

plt.figure()
plt.plot(lamda_arr,accuracy_arr,ls='--',marker='o')
plt.xlabel(r'$\lambda$')
plt.ylabel('Test accuracy')
plt.xscale('log')
plt.title('BarlowTwins trained features on {}'.format(dataset))
plt.grid('on')
plt.show()

