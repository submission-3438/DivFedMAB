import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad
from sklearn.metrics import pairwise_distances


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.rng = np.random.default_rng()

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))

        test_accuracies = []
        acc_10quant = []
        acc_20quant = []
        test_acc_var = []
        train_accuracies = []
        train_losses = []
        num_sampled = []
        
        
        norm_diff_div_total_clients = []  # New list to store norm of difference divided by total number of clients
        all_grads_sum = []
        sui_grads_sum = []
        diff_norm = []
        # ---- r_norm history: shape (num_rounds, num_clients) ----
        self.r_norm_history = np.zeros((self.num_rounds, len(self.clients)))

        # shape: (num_rounds, num_clients)
        self.mu_history = np.zeros((self.num_rounds, len(self.clients)))


        
        
        client_sets_all = np.zeros([self.num_rounds, self.clients_per_round], dtype=int)
        diff_grad = np.zeros([self.num_rounds, len(self.clients)])
        
        N_i = np.array( [0] * len(self.clients) )
        prev_grad_sum =  np.array( [0] * len(self.clients) )
        reward =  np.zeros(len(self.clients))
        lcb =  np.zeros(len(self.clients))
        
        
        
        #C_i = np.array( [0] * len(self.clients) )
       #marg_util_sum = np.array( [0] * len(self.clients) )
        
        
        uu =[]
        aa = []
        
        
        for i in range(self.num_rounds):
            #print("for loop i is ",i)
            allcl_models = []
            for cc in self.clients:
                clmodel = cc.get_params()
                allcl_models.append(clmodel)
            # test model
            if i % self.eval_every == 0:
                stats = self.test()  # have set the latest model for all clients
                stats_train = self.train_error_and_loss()
                
                test_accuracies.append(np.sum(stats[3]) * 1.0 / np.sum(stats[2]))
                acc_10quant.append(np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.1))
                acc_20quant.append(np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.2))
                test_acc_var.append(np.var([i/j for i,j in zip(stats[3], stats[2])]))
                train_accuracies.append(np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2]))
                train_losses.append(np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2]))
              
                
                
                tqdm.write('At round {} per-client-accuracy: {}'.format(i, [i/j for i,j in zip(stats[3], stats[2])]))
                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))  # testing accuracy
                tqdm.write('At round {} acc. 10th: {}'.format(i, np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.1)))  # testing accuracy variance
                tqdm.write('At round {} acc. 20th: {}'.format(i, np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.2)))  # testing accuracy variance
                tqdm.write('At round {} acc. variance: {}'.format(i, np.var([i/j for i,j in zip(stats[3], stats[2])])))  # testing accuracy variance
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
                uu.append(np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2]) )
                #tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])))
                #aa.append ( np.var([i/j for i,j in zip(stats[3], stats[2])]))
                
            
           
            #print('The value of self.clientsel_algo is ',self.clientsel_algo)
            if self.clientsel_algo == 'submodular':
                print('entered fedavg submodular')
                #if i % self.m_interval == 0: # Moved the condition inside the function
                if i == 0 or self.clients_per_round == 1:  # at the first iteration or when m=1, collect gradients from all clients
                    self.all_grads = np.asarray(self.show_grads()[:-1])
                    self.norm_diff = pairwise_distances(self.all_grads, metric="euclidean")
                    np.fill_diagonal(self.norm_diff, 0)
                    
                #print('will enter cl_submod ')
           
                indices, selected_clients, all_grad,N_i,reward, prev_grad_sum= self.select_cl_submod(i, self.clients_per_round, N_i,reward,prev_grad_sum, True)
              


                

                all_grads_sum.append(np.sum(self.all_grads, axis=0))
                sui_grads_sum.append(np.sum(self.all_grads[indices], axis=0))
                diff = all_grads_sum[-1] - sui_grads_sum[-1]
                diff_norm.append(np.linalg.norm(diff) / len(self.clients))
                norm_diff_div_total_clients.append(diff_norm[-1])  # Append to the new list

                
                
                
                
              
                
                
                active_clients = selected_clients # Dropping clients don't apply in this case
                #print('done cl_submod ')
                if i == 0:
                    diff_grad[i] = np.zeros(len(all_grad))
                else:
                    diff_grad[i] = np.linalg.norm(all_grad - old_grad, axis=1)
                old_grad = all_grad.copy()
            elif self.clientsel_algo == 'lossbased':
                print('Power of choice')
                
                if i % self.m_interval == 0:
                    lprob = stats_train[2]/np.sum(stats_train[2], axis=0)
                    #d=100
                    subsample = 0.1
                    #d = max(self.clients_per_round, int(subsample * len(self.clients)))
                    d = len(self.clients)
                    lvals = self.rng.choice(stats_train[4], size=d, replace = False, p=lprob)
                    Mlist = [np.where(stats_train[4] == i)[0][0] for i in lvals]
                    lossvals = np.asarray(stats_train[4]) #All loss values
                    sel_losses = lossvals[Mlist]
                    idx_losses = np.argpartition(sel_losses, self.clients_per_round)
                    values = sel_losses[idx_losses[-self.clients_per_round:]]
                    
                    listvalues = values.tolist()
                    listlossvals = lossvals.tolist()
                    indices = [listlossvals.index(i) for i in listvalues] 
                
                #indices = np.argsort(stats_train[4], axis=0)[::-1][:self.clients_per_round]
                selected_clients = np.asarray(self.clients)[indices]
                np.random.seed(i)
                active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1-self.drop_percent)), replace=False)
            else:
                print('Uniform doing ')
                indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
                np.random.seed(i)
                active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1-self.drop_percent)), replace=False)
                
           
            
            
            #print('Client set is ', indices)
            
            #tqdm.write('At round {} num. clients sampled: {}'.format(i, len(indices)))
            tqdm.write('At round {} num. clients sampled: {}'.format(i, len(indices)))
            num_sampled.append(len(indices))
            csolns = []  # buffer for receiving client solutions
            
            glob_copy = np.append(self.latest_model[0].flatten(), self.latest_model[1])

            for idx, c in enumerate(active_clients.tolist()):  # simply drop the slow devices
                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally
                soln, stats, grads = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
                #print("Shape of grads", np.shape(grads))
                
                # gather solutions from client
                csolns.append(soln)

                if self.clientsel_algo == 'submodular':
                    self.all_grads[indices[idx]] = grads
                
                # Update server's view of clients' models (only for the selected clients)
                #c.updatevec = (glob_copy - np.append(c.get_params()[0].flatten(), c.get_params()[1]))*0.01
                c.updatevec = np.append(c.get_params()[0].flatten(), c.get_params()[1])

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # update models
            if self.clientsel_algo == 'submodular':
                self.norm_diff[indices] = pairwise_distances(self.all_grads[indices], self.all_grads, metric="euclidean")
                self.norm_diff[:, indices] = self.norm_diff[indices].T
                self.latest_model = self.aggregate(csolns)
                #self.latest_model = self.aggregate_submod(csolns, gammas)
            elif self.clientsel_algo == 'lossbased':
                self.latest_model = self.aggregate_simple(csolns)
            else:
                self.latest_model = self.aggregate(csolns)







        #tqdm.write('training losses are: {}'.format(uu))
        tqdm.write('training losses are: {}'.format(aa))
       
        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} per-client-accuracy: {}'.format(i, [i/j for i,j in zip(stats[3], stats[2])]))
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} acc. variance: {}'.format(self.num_rounds, np.var([i/j for i,j in zip(stats[3], stats[2])])))
        tqdm.write('At round {} acc. 10th: {}'.format(self.num_rounds, np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.1)))
        tqdm.write('At round {} acc. 20th: {}'.format(self.num_rounds, np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.2)))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
        tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])))

        # ---- save r_norm history after training ----
        # print("\n===== r_norm HISTORY (comma-separated, client-wise) =====")

        # num_clients = len(self.clients)
        # num_rounds = self.num_rounds

        # for cid in range(num_clients):
        #     values = self.r_norm_history[:, cid]
        #     csv_line = ",".join(map(str, values))
        #     print(f"Client {cid}: {csv_line}")

        print("\n===== MU HISTORY (comma-separated, client-wise) =====")

        num_clients = len(self.clients)

        for cid in range(num_clients):
            values = self.mu_history[:, cid]
            csv_line = ",".join(map(str, values))
            print(f"Client {cid}: {csv_line}")











    #    if self.clientsel_algo == 'submodular':
   #         np.save('./results/sent140/psubmod_select_client_sets_all_%s_epoch%d_numclient%d_m%d.npy' % (self.clientsel_algo, self.num_epochs, self.clients_per_round, self.m_interval), client_sets_all)
  #          np.save('./results/sent140/psubmod_client_diff_grad_all_%s_epoch%d_numclient%d_m%d.npy' % (self.clientsel_algo, self.num_epochs, self.clients_per_round, self.m_interval), diff_grad)
 #       elif self.clientsel_algo == 'lossbased':
#            np.save('./results/sent140/powerofchoice_select_client_sets_all_%s_epoch%d_numclient%d_m%d.npy' % (self.clientsel_algo, self.num_epochs, self.clients_per_round, self.m_interval), client_sets_all)

        #print('Number of samples', stats_train[2])

        # save_dir = "./results/"
        # result_path = os.path.join(save_dir,'submodular.csv')
        # print('Writing Statistics to file')
        # with open(result_path, 'wb') as f:
        #     np.savetxt(f, np.c_[test_accuracies, train_accuracies, train_losses, num_sampled], delimiter=",")
        
        
    
        
        
        